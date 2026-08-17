"""The supervisor as a sequence of decisions, not a set of checks.

Per `tasks/lessons.md`, component tests in isolation are what missed the read-path-mutation and dying-loop
bugs in v1. The guardrail's interesting behaviour is entirely across timesteps — engage, hold through
recovery, release — so that is what is exercised here.

The one property worth stating up front, because everything else follows from it: **the supervisor always
returns an action.** There is no path where the simulator gets nothing. That path is what the service
exists to replace — the control loop used to catch a malformed action, log it, and step unmanaged with
every flow at full demand.
"""

from __future__ import annotations

import math

from nexus_safety.bounds import ActionBounds
from nexus_safety.fallback import FallbackReason, QosThresholds
from nexus_safety.proposal import ProposedAction
from nexus_safety.violation import ViolationKind
from nexus_supervisor.supervisor import SUPERVISOR_ISSUER, SafetySupervisor
from nexus_types.network_state import FlowState, LinkState, NetworkState
from nexus_types.traffic_class import TrafficClass

_FAST = QosThresholds(
    max_p95_latency_ms=100.0,
    consecutive_breaches_to_engage=3,
    consecutive_healthy_to_release=5,
)


def _state(timestep: int = 0, latency_ms: float = 5.0, allocated: float = 40.0) -> NetworkState:
    return NetworkState(
        timestep=timestep,
        links=(
            LinkState(
                link_id="n0->n1",
                source="n0",
                target="n1",
                capacity_mbps=100.0,
                effective_capacity_mbps=100.0,
                allocated_mbps=allocated,
                queue_mb=1.0,
                utilization=allocated / 100.0,
                latency_ms=latency_ms,
            ),
        ),
        flows=tuple(
            FlowState(
                flow_id=flow_id,
                source="n0",
                target="n1",
                traffic_class=TrafficClass.BULK,
                demand_mbps=100.0,
                allocated_mbps=allocated,
                dropped_mbps=0.0,
                path=("n0->n1",),
                candidate_paths=(("n0->n1",),),
                latency_ms=latency_ms,
            )
            for flow_id in ("f0", "f1")
        ),
    )


def _ok(**shares: float) -> ProposedAction:
    return ProposedAction(share_by_flow_id=dict(shares) or {"f0": 0.5, "f1": 0.5})


# --- The policy drives when it behaves --------------------------------------------------------------


def test_a_clean_action_is_applied_untouched_and_stays_the_controllers() -> None:
    decision = SafetySupervisor(thresholds=_FAST).supervise(_state(), _ok(f0=0.5, f1=0.6))

    assert decision.action.share_by_flow_id == {"f0": 0.5, "f1": 0.6}
    assert not decision.fallback_engaged
    assert decision.action.issued_by == "controller"


def test_a_correctable_action_is_corrected_without_falling_back() -> None:
    decision = SafetySupervisor(thresholds=_FAST).supervise(_state(), _ok(f0=1.4, f1=0.5))

    assert decision.action.share_by_flow_id["f0"] == 1.0
    assert not decision.fallback_engaged
    assert decision.was_corrected


# --- The guardrail takes over when it must ----------------------------------------------------------


def test_a_diverged_policy_hands_the_timestep_to_the_deterministic_allocator() -> None:
    supervisor = SafetySupervisor(thresholds=_FAST)

    decision = supervisor.supervise(_state(), _ok(f0=math.nan, f1=0.5))

    assert decision.fallback_engaged
    assert decision.fallback_reason is FallbackReason.UNSAFE_ACTION
    assert all(math.isfinite(s) for s in decision.action.share_by_flow_id.values())


def test_a_fallback_action_is_never_labelled_as_the_policys() -> None:
    # Relabelling it would make the dashboard, the decision log and the agent's explanations all
    # attribute the baseline's behaviour to the policy.
    supervisor = SafetySupervisor(thresholds=_FAST)

    decision = supervisor.supervise(_state(), _ok(f0=math.inf, f1=0.5))

    assert decision.action.issued_by == SUPERVISOR_ISSUER


def test_an_unreachable_controller_produces_a_deliberate_allocation_not_silence() -> None:
    # The pre-supervisor behaviour was to step with no action at all — every flow at full demand, which
    # is a silent degradation dressed up as a log line.
    supervisor = SafetySupervisor(thresholds=_FAST)

    decision = supervisor.supervise(_state(), None)

    assert decision.fallback_engaged
    assert decision.fallback_reason is FallbackReason.CONTROLLER_UNREACHABLE
    assert decision.action.share_by_flow_id  # something was actually allocated


def test_a_sustained_qos_breach_engages_even_though_every_action_is_in_bounds() -> None:
    # Action bounds constrain what the policy may ask for; QoS thresholds constrain what the network
    # actually does. A policy can emit perfect shares and still drive the network into the ground.
    supervisor = SafetySupervisor(thresholds=_FAST)

    decisions = [
        supervisor.supervise(_state(step, latency_ms=500.0), _ok(f0=0.5, f1=0.5)) for step in range(3)
    ]

    assert not decisions[0].fallback_engaged
    assert decisions[-1].fallback_engaged
    assert decisions[-1].fallback_reason is FallbackReason.QOS_BREACH


# --- Hysteresis across the sequence -----------------------------------------------------------------


def test_the_policy_does_not_get_the_network_back_the_moment_metrics_recover() -> None:
    # The fallback fixed the metric. Handing straight back is how the system oscillates.
    supervisor = SafetySupervisor(thresholds=_FAST)
    for step in range(3):
        supervisor.supervise(_state(step, latency_ms=500.0), _ok(f0=0.5, f1=0.5))

    recovered = supervisor.supervise(_state(10, latency_ms=5.0), _ok(f0=0.5, f1=0.5))

    assert recovered.fallback_engaged


def test_a_clean_action_during_the_cooldown_is_still_not_applied() -> None:
    # The action is fine; the fallback is holding for its own reason. Applying the policy's action here
    # would defeat the cooldown entirely.
    supervisor = SafetySupervisor(thresholds=_FAST)
    for step in range(3):
        supervisor.supervise(_state(step, latency_ms=500.0), _ok(f0=0.5, f1=0.5))

    held = supervisor.supervise(_state(10, latency_ms=5.0), _ok(f0=0.9, f1=0.9))

    assert held.action.issued_by == SUPERVISOR_ISSUER
    assert held.action.share_by_flow_id != {"f0": 0.9, "f1": 0.9}


def test_control_returns_after_the_full_cooldown() -> None:
    supervisor = SafetySupervisor(thresholds=_FAST)
    for step in range(3):
        supervisor.supervise(_state(step, latency_ms=500.0), _ok(f0=0.5, f1=0.5))

    for step in range(10, 15):
        decision = supervisor.supervise(_state(step, latency_ms=5.0), _ok(f0=0.5, f1=0.5))

    assert not decision.fallback_engaged
    assert decision.action.issued_by == "controller"


# --- Observability ----------------------------------------------------------------------------------


def test_status_reports_the_numbers_service_metrics_cannot() -> None:
    # Uptime and error rate say nothing about whether a policy is behaving.
    supervisor = SafetySupervisor(thresholds=_FAST)
    supervisor.supervise(_state(0), _ok(f0=1.5, f1=0.5))
    supervisor.supervise(_state(1), _ok(f0=math.nan, f1=0.5))

    status = supervisor.status()
    by_kind = status["violations_by_kind"]

    assert status["supervised_timesteps"] == 2
    assert status["corrected_timesteps"] == 1
    assert status["activation_count"] == 1
    assert isinstance(by_kind, dict)
    assert by_kind[ViolationKind.SHARE_OUT_OF_RANGE.value] == 1
    assert by_kind[ViolationKind.NON_FINITE_SHARE.value] == 1


def test_drift_reports_nothing_rather_than_something_invented_without_a_reference() -> None:
    # Reporting "no drift" against a reference nobody established is worse than reporting nothing.
    supervisor = SafetySupervisor(thresholds=_FAST)
    supervisor.supervise(_state(0), _ok(f0=0.5, f1=0.5))

    drift = supervisor.status()["drift"]

    assert isinstance(drift, dict)
    assert drift["established"] is False
    assert drift["drifting"] is False


def test_a_starving_policy_is_reported_even_while_every_action_is_in_bounds() -> None:
    supervisor = SafetySupervisor(bounds=ActionBounds(starvation_window=3), thresholds=_FAST)

    for step in range(3):
        supervisor.supervise(_state(step), _ok(f0=0.0, f1=0.9))

    assert supervisor.status()["starved_flows"] == ["f0"]
