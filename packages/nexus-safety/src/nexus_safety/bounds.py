"""Hard action bounds — limits the policy physically cannot exceed.

The rule from the v2 specification, stated as plainly as it deserves: *never trust the learned policy
alone*. These checks run **outside the model**, on the action it produced, against the network it will be
applied to. They hold whether the policy is well-trained, diverged, adversarially prompted, or replaced by
something entirely different tomorrow.

Two design commitments:

- **Correct where correction is unambiguous, fall back where it is not.** Clamping a share of 1.9 to 1.0
  preserves the policy's evident intent. There is no comparable "obvious" repair for a NaN, so that
  escalates. Guessing at intent is how a guardrail becomes a second, unaccountable policy.
- **Everything is reported, including what was corrected.** A silently repaired action is a policy defect
  nobody ever learns about — and the fallback-activation rate is the clearest failure signal the system
  has (v2 spec §8.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from nexus_safety.proposal import ProposedAction
from nexus_safety.violation import Violation, ViolationKind
from nexus_types.action import AllocationAction
from nexus_types.network_state import FlowState, NetworkState

DEFAULT_MIN_SHARE = 0.0
DEFAULT_MAX_SHARE = 1.0


@dataclass(frozen=True, slots=True)
class ActionBounds:
    """The limits every action is checked against.

    Deliberately *not* read from the policy, the checkpoint, or any A1 policy. A guardrail configured by
    the thing it guards is not a guardrail — this comes from the supervisor's own configuration.
    """

    min_share: float = DEFAULT_MIN_SHARE
    max_share: float = DEFAULT_MAX_SHARE

    starvation_share: float = 0.01
    """At or below this, a flow counts as starved for the window check. Not zero: a policy can hold a flow
    at 0.001 indefinitely and starve it just as thoroughly as at exactly zero."""

    starvation_window: int = 50
    """Consecutive timesteps of starvation before it is a violation. A few steps of near-zero is a
    legitimate allocation decision under contention; fifty is a policy that has written a flow off."""

    require_known_paths: bool = True
    """Reject routes that are not among a flow's candidate paths."""


@dataclass(frozen=True, slots=True)
class BoundsCheck:
    """The result of checking one action: what was wrong, and the safe action to apply instead.

    ``action`` is a validated :class:`AllocationAction` — the supervisor inspects raw output and emits
    something the rest of the system can trust. That asymmetry is the whole point of the boundary.
    """

    action: AllocationAction
    """The corrected, validated action. Equivalent to the proposal when nothing was wrong."""

    violations: tuple[Violation, ...]

    @property
    def is_clean(self) -> bool:
        return not self.violations

    @property
    def requires_fallback(self) -> bool:
        """Whether the action is unsafe in a way clamping cannot repair.

        Non-finite shares only. Everything else has an unambiguous correction, and falling back for a
        single out-of-range share would hand the network to the baseline over a rounding error.
        """
        return any(v.kind is ViolationKind.NON_FINITE_SHARE for v in self.violations)


class StarvationTracker:
    """Consecutive-timestep starvation counts, per flow.

    Stateful because the bound is about behaviour over time, and deliberately separate from the stateless
    checks so that :func:`check_action` stays a pure function of (action, state, bounds).
    """

    def __init__(self, bounds: ActionBounds) -> None:
        self._bounds = bounds
        self._consecutive: dict[str, int] = {}

    def observe(self, action: ProposedAction) -> tuple[Violation, ...]:
        """Record one timestep and report flows that have now been starved for the whole window."""
        violations: list[Violation] = []
        for flow_id, share in action.share_by_flow_id.items():
            if math.isfinite(share) and share <= self._bounds.starvation_share:
                self._consecutive[flow_id] = self._consecutive.get(flow_id, 0) + 1
            else:
                self._consecutive[flow_id] = 0

            if self._consecutive[flow_id] == self._bounds.starvation_window:
                # Reported once, on the step the window closes — not every step thereafter, which would
                # bury every other finding under one stuck flow.
                violations.append(
                    Violation(
                        kind=ViolationKind.STARVATION,
                        subject=flow_id,
                        detail=(
                            f"{flow_id} has been at or below share {self._bounds.starvation_share} for "
                            f"{self._bounds.starvation_window} consecutive timesteps. A policy that "
                            f"permanently starves a flow to improve an aggregate has found a degenerate "
                            f"optimum."
                        ),
                    )
                )
        return tuple(violations)

    def forget(self, flow_id: str) -> None:
        self._consecutive.pop(flow_id, None)

    @property
    def starved_flows(self) -> tuple[str, ...]:
        """Flows currently past the starvation window. Reported on the supervisor's status endpoint."""
        return tuple(
            sorted(
                flow_id
                for flow_id, count in self._consecutive.items()
                if count >= self._bounds.starvation_window
            )
        )


def check_action(action: ProposedAction, state: NetworkState, bounds: ActionBounds) -> BoundsCheck:
    """Check a proposed action against the network it would be applied to, and return a safe version.

    Takes a :class:`ProposedAction` rather than an ``AllocationAction`` deliberately — see that module.
    The returned action is validated, so everything downstream of the supervisor still gets the strong
    type.

    Pure: same proposal, same state, same bounds ⇒ same result. The time-dependent starvation bound lives
    in :class:`StarvationTracker` precisely to keep this so.
    """
    known_flows: dict[str, FlowState] = {flow.flow_id: flow for flow in state.flows}
    violations: list[Violation] = []
    shares: dict[str, float] = {}

    for flow_id, share in action.share_by_flow_id.items():
        if flow_id not in known_flows:
            violations.append(
                Violation(
                    kind=ViolationKind.UNKNOWN_FLOW,
                    subject=flow_id,
                    detail=(
                        f"Action allocates to {flow_id}, which is not in the network. Dropped. This "
                        f"usually means the policy is serving a topology it was not trained on."
                    ),
                )
            )
            continue

        if not math.isfinite(share):
            # No safe repair exists: 0.0 starves the flow, 1.0 floods it, and both are inventions. The
            # supervisor escalates to the fallback allocator instead of choosing on the policy's behalf.
            violations.append(
                Violation(
                    kind=ViolationKind.NON_FINITE_SHARE,
                    subject=flow_id,
                    detail=(
                        f"{flow_id} was allocated a non-finite share ({share!r}). The usual cause is a "
                        f"diverged policy, not a decoding bug. Escalating to the fallback allocator — "
                        f"there is no honest way to repair this value."
                    ),
                )
            )
            continue

        if share < bounds.min_share or share > bounds.max_share:
            clamped = min(max(share, bounds.min_share), bounds.max_share)
            violations.append(
                Violation(
                    kind=ViolationKind.SHARE_OUT_OF_RANGE,
                    subject=flow_id,
                    detail=(
                        f"{flow_id} share {share:.4f} is outside "
                        f"[{bounds.min_share}, {bounds.max_share}]; clamped to {clamped}. Check that the "
                        f"policy's action space matches this MDP_VERSION."
                    ),
                    corrected_to=clamped,
                )
            )
            shares[flow_id] = clamped
            continue

        shares[flow_id] = share

    for flow_id in known_flows:
        if flow_id not in action.share_by_flow_id:
            # Absent is not zero. Filling it with zero would be the supervisor making an allocation
            # decision, so it takes the flow's current share of its own demand instead — the closest
            # thing to "carry on as you were" available without a network model.
            violations.append(
                Violation(
                    kind=ViolationKind.MISSING_FLOW,
                    subject=flow_id,
                    detail=(
                        f"{flow_id} exists in the network but received no allocation. Held at its "
                        f"current rate rather than zeroed — an omission is not a decision to starve it."
                    ),
                    corrected_to=_current_share(known_flows[flow_id]),
                )
            )
            shares[flow_id] = _current_share(known_flows[flow_id])

    paths, path_violations = _check_paths(action, known_flows, bounds)
    violations.extend(path_violations)

    if any(v.kind is ViolationKind.NON_FINITE_SHARE for v in violations):
        # A validated action cannot be built from these shares, and must not be: the caller escalates to
        # the fallback allocator. An empty allocation stands in so the type is inhabited; `requires_fallback`
        # is what callers actually branch on.
        return BoundsCheck(
            action=AllocationAction(share_by_flow_id={}, issued_by=action.issued_by),
            violations=tuple(violations),
        )

    return BoundsCheck(
        action=AllocationAction(
            share_by_flow_id=shares,
            path_by_flow_id=paths,
            issued_by=action.issued_by,
        ),
        violations=tuple(violations),
    )


def _check_paths(
    action: ProposedAction,
    known_flows: dict[str, FlowState],
    bounds: ActionBounds,
) -> tuple[dict[str, tuple[str, ...]], list[Violation]]:
    """Drop routes that are not among a flow's candidates, keeping the flow on its default path."""
    if not bounds.require_known_paths:
        return dict(action.path_by_flow_id), []

    accepted: dict[str, tuple[str, ...]] = {}
    violations: list[Violation] = []

    for flow_id, path in action.path_by_flow_id.items():
        flow = known_flows.get(flow_id)
        if flow is None:
            continue  # already reported as UNKNOWN_FLOW
        if path in flow.candidate_paths:
            accepted[flow_id] = path
            continue
        violations.append(
            Violation(
                kind=ViolationKind.INVALID_PATH,
                subject=flow_id,
                detail=(
                    f"{flow_id} was routed over {list(path)}, which is not one of its candidate paths. "
                    f"Dropped; the flow keeps its default route. Applying it would have carried traffic "
                    f"over links the topology does not connect."
                ),
            )
        )
    return accepted, violations


def _current_share(flow: FlowState) -> float:
    """A flow's present allocation as a fraction of its demand, clamped into range.

    Used only to hold an omitted flow steady. Reads the state the simulator sent rather than any model of
    the network — the supervisor, like the controller, holds none (ADR-0006).
    """
    if flow.demand_mbps <= 0.0:
        return 0.0
    return min(max(flow.allocated_mbps / flow.demand_mbps, DEFAULT_MIN_SHARE), DEFAULT_MAX_SHARE)
