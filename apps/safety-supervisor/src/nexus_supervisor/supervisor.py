"""The guardrail, as a decision function over one timestep.

Kept separate from the HTTP surface so it can be tested without a running controller — the interesting
behaviour is the *sequence* of decisions across timesteps (engage, hold, release), and that is miserable
to exercise through a network.

The order of operations is deliberate and is the safety argument in miniature:

1. **Observe the network first.** Drift and QoS are properties of what already happened; reading them
   after acting would mean the fallback decision lags the evidence by a timestep.
2. **Check the proposal before applying it.** An unrepairable action never reaches the simulator.
3. **Fall back rather than invent.** When the supervisor cannot honestly repair an action, it hands the
   timestep to a deterministic allocator instead of guessing what the policy meant.
4. **Record everything, including corrections.** A silently repaired action is a policy defect nobody
   ever learns about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nexus_baselines.allocator import StaticAllocator
from nexus_safety.bounds import ActionBounds, StarvationTracker, check_action
from nexus_safety.drift import DriftDetector, DriftReference
from nexus_safety.fallback import (
    FallbackController,
    FallbackReason,
    QosSample,
    QosThresholds,
)
from nexus_safety.proposal import ProposedAction
from nexus_safety.shadow import ShadowComparator, ShadowThresholds
from nexus_safety.violation import Violation
from nexus_telemetry.client import get_telemetry
from nexus_types.action import AllocationAction
from nexus_types.network_state import NetworkState

_telemetry = get_telemetry("nexus.supervisor")

SUPERVISOR_ISSUER = "supervisor:fallback"
"""Provenance stamped on an action the supervisor produced itself.

Never reuses the controller's label. Relabelling a fallback allocation as a policy decision would make
every downstream explanation — the dashboard, the agent layer's answers, the decision log — attribute the
baseline's behaviour to the policy.
"""


@dataclass(frozen=True, slots=True)
class SupervisedAction:
    """What the supervisor decided for one timestep, and why."""

    action: AllocationAction
    violations: tuple[Violation, ...] = ()
    fallback_engaged: bool = False
    fallback_reason: FallbackReason = FallbackReason.NONE
    drifting: bool = False
    drifted_features: tuple[str, ...] = field(default_factory=tuple)

    @property
    def was_corrected(self) -> bool:
        return bool(self.violations)


class SafetySupervisor:
    """Independent guardrail over the controller's actions.

    Independent in the sense that matters: it holds its own bounds, its own thresholds and its own
    fallback allocator, and it never asks the controller whether an action is acceptable. A guardrail
    configured by the thing it guards is not a guardrail (v2 spec, Decision 9).
    """

    def __init__(
        self,
        bounds: ActionBounds | None = None,
        thresholds: QosThresholds | None = None,
        drift_reference: DriftReference | None = None,
        shadow_thresholds: ShadowThresholds | None = None,
    ) -> None:
        self._bounds = bounds or ActionBounds()
        self._fallback = FallbackController(thresholds)
        self._starvation = StarvationTracker(self._bounds)
        self._drift = DriftDetector(drift_reference or DriftReference())
        self._allocator: StaticAllocator | None = None
        self._supervised = 0
        self._corrected = 0
        self._violation_counts: dict[str, int] = {}
        self._shadow = ShadowComparator(shadow_thresholds)

    def supervise(
        self,
        state: NetworkState,
        proposal: ProposedAction | None,
        shadow_proposal: ProposedAction | None = None,
        shadow_configured: bool = False,
    ) -> SupervisedAction:
        """Decide the action actually applied for one timestep.

        ``proposal`` is ``None`` when the controller could not be reached. That is treated as an immediate
        fallback rather than a breach to accumulate: there is no policy output to be patient about, and
        the pre-supervisor behaviour — stepping unmanaged, every flow at full demand — was a silent
        degradation dressed up as a log line.

        ``shadow_proposal`` is a candidate policy's action. It is checked and recorded and **never
        applied** — there is no branch below in which it can reach the returned action. That is the
        guarantee shadow mode rests on, and it is enforced by the absence of a code path rather than by a
        flag.
        """
        self._supervised += 1
        if shadow_configured:
            self._observe_shadow(state, proposal, shadow_proposal)
        self._drift.observe(state)
        self._fallback.observe(_sample(state), state.timestep)
        drift = self._drift.report()

        if proposal is None:
            self._fallback.engage_now(
                FallbackReason.CONTROLLER_UNREACHABLE,
                "No action from the controller; running the deterministic allocator.",
                state.timestep,
            )
            return self._fallback_action(state, (), drift.is_drifting, drift.drifted_features)

        checked = check_action(proposal, state, self._bounds)
        violations = (*checked.violations, *self._starvation.observe(proposal))
        self._count(violations)

        if checked.requires_fallback:
            self._fallback.engage_now(
                FallbackReason.UNSAFE_ACTION,
                "; ".join(v.detail for v in checked.violations) or "Unrepairable action.",
                state.timestep,
            )
            _telemetry.warning(
                "Unsafe action refused — deterministic allocator engaged",
                {"timestep": state.timestep, "violations": [v.model_dump(mode="json") for v in violations]},
            )
            return self._fallback_action(state, violations, drift.is_drifting, drift.drifted_features)

        if self._fallback.engaged:
            # The action was fine, but the fallback is still holding the network for its own reason —
            # a QoS breach that has not yet cleared its cooldown. Applying the policy's action here
            # would defeat the cooldown and reintroduce the oscillation it exists to prevent.
            return self._fallback_action(state, violations, drift.is_drifting, drift.drifted_features)

        if violations:
            self._corrected += 1
            _telemetry.info(
                "Action corrected within bounds",
                {"timestep": state.timestep, "violations": [v.model_dump(mode="json") for v in violations]},
            )

        return SupervisedAction(
            action=checked.action,
            violations=violations,
            fallback_engaged=False,
            fallback_reason=FallbackReason.NONE,
            drifting=drift.is_drifting,
            drifted_features=drift.drifted_features,
        )

    def _observe_shadow(
        self,
        state: NetworkState,
        incumbent: ProposedAction | None,
        candidate: ProposedAction | None,
    ) -> None:
        """Record the candidate's behaviour beside the incumbent's.

        A timestep with no incumbent to compare against is skipped rather than scored: divergence from
        nothing is not a number, and counting it as agreement would flatter a candidate during exactly
        the periods when the primary controller was struggling.
        """
        if candidate is None:
            self._shadow.record_failure()
            return
        if incumbent is None:
            return
        self._shadow.record(incumbent, candidate, check_action(candidate, state, self._bounds).violations)

    def shadow_report(self) -> dict[str, object]:
        """The candidate's readiness verdict, for the promotion decision."""
        report = self._shadow.report()
        return {
            "verdict": report.verdict.value,
            "reason": report.reason,
            "samples": report.samples,
            "failures": report.failures,
            "failure_rate": report.failure_rate,
            "mean_divergence": report.mean_divergence,
            "max_divergence": report.max_divergence,
            "violations": [v.model_dump(mode="json") for v in report.violations[:20]],
        }

    def status(self) -> dict[str, object]:
        """Policy-level observability: what the guardrail has seen and done.

        Service metrics (latency, error rate, uptime) say nothing about whether a *policy* is behaving.
        These are the numbers that do — violation rates, correction rate, fallback activations, drift —
        and they are the v2 specification's §8.4 list.
        """
        drift = self._drift.report()
        return {
            **self._fallback.status(),
            "supervised_timesteps": self._supervised,
            "corrected_timesteps": self._corrected,
            "correction_rate": self._corrected / self._supervised if self._supervised else 0.0,
            "violations_by_kind": dict(sorted(self._violation_counts.items())),
            "starved_flows": list(self._starvation.starved_flows),
            "drift": {
                "established": bool(drift.per_feature_sigma),
                "max_sigma": drift.max_sigma,
                "warn_sigma": drift.warn_sigma,
                "drifting": drift.is_drifting,
                "drifted_features": list(drift.drifted_features),
                "samples": drift.samples,
            },
            "recent_transitions": [
                {
                    "engaged": t.engaged,
                    "reason": t.reason.value,
                    "detail": t.detail,
                    "timestep": t.timestep,
                }
                for t in self._fallback.transitions[-10:]
            ],
        }

    def _fallback_action(
        self,
        state: NetworkState,
        violations: tuple[Violation, ...],
        drifting: bool,
        drifted_features: tuple[str, ...],
    ) -> SupervisedAction:
        return SupervisedAction(
            action=self._ensure_allocator(state)
            .allocate(state)
            .model_copy(update={"issued_by": SUPERVISOR_ISSUER}),
            violations=violations,
            fallback_engaged=True,
            fallback_reason=self._fallback.reason,
            drifting=drifting,
            drifted_features=drifted_features,
        )

    def _ensure_allocator(self, state: NetworkState) -> StaticAllocator:
        """Derive the deterministic allocator from the network as observed, once.

        From received state, never from configuration — the supervisor holds no model of the network, for
        the same reason the controller does not (ADR-0006).
        """
        if self._allocator is None:
            mean_capacity = (
                sum(link.capacity_mbps for link in state.links) / len(state.links) if state.links else 0.0
            )
            self._allocator = StaticAllocator(equal_rate_mbps=mean_capacity / max(1, len(state.flows)))
        return self._allocator

    def _count(self, violations: tuple[Violation, ...]) -> None:
        for violation in violations:
            self._violation_counts[violation.kind.value] = (
                self._violation_counts.get(violation.kind.value, 0) + 1
            )


def _sample(state: NetworkState) -> QosSample:
    """Read the observed outcome off the state the simulator sent.

    Derived rather than requested: asking the simulator for metrics would put another call in the fast
    path, and everything needed is already in the state that arrived.
    """
    if not state.links or not state.flows:
        return QosSample(p95_latency_ms=0.0, drop_rate=0.0, fairness=1.0)

    latencies = sorted(link.latency_ms for link in state.links)
    index = max(0, min(len(latencies) - 1, round(0.95 * (len(latencies) - 1))))

    demand = sum(flow.demand_mbps for flow in state.flows)
    dropped = sum(flow.dropped_mbps for flow in state.flows)

    allocations = [flow.allocated_mbps for flow in state.flows]
    total = sum(allocations)
    squares = sum(value * value for value in allocations)
    # Jain's fairness index. Defined as 1 for an all-zero allocation: nothing is being shared unfairly,
    # and reporting 0 there would trip the fairness threshold on an idle network.
    fairness = (total * total) / (len(allocations) * squares) if squares > 0.0 else 1.0

    return QosSample(
        p95_latency_ms=latencies[index],
        drop_rate=dropped / demand if demand > 0.0 else 0.0,
        fairness=fairness,
    )
