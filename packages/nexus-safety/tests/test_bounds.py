"""Hard action bounds.

The premise being tested throughout: these run on what the policy *produced*, not on something already
proven valid. ``AllocationAction`` rejects out-of-range and non-finite shares in its own validator, so a
guardrail that inspected one would only ever see actions that were already fine.
"""

from __future__ import annotations

import math

import pytest

from nexus_safety.bounds import ActionBounds, StarvationTracker, check_action
from nexus_safety.proposal import ProposedAction
from nexus_safety.violation import ViolationKind
from nexus_types.network_state import FlowState, LinkState, NetworkState
from nexus_types.traffic_class import TrafficClass


def _state(flow_ids: tuple[str, ...] = ("f0", "f1")) -> NetworkState:
    return NetworkState(
        timestep=10,
        links=(
            LinkState(
                link_id="n0->n1",
                source="n0",
                target="n1",
                capacity_mbps=100.0,
                effective_capacity_mbps=100.0,
                allocated_mbps=50.0,
                queue_mb=1.0,
                utilization=0.5,
                latency_ms=5.0,
            ),
        ),
        flows=tuple(
            FlowState(
                flow_id=flow_id,
                source="n0",
                target="n1",
                traffic_class=TrafficClass.BULK,
                demand_mbps=100.0,
                allocated_mbps=40.0,
                dropped_mbps=0.0,
                path=("n0->n1",),
                candidate_paths=(("n0->n1",),),
                latency_ms=5.0,
            )
            for flow_id in flow_ids
        ),
    )


def _proposal(**shares: float) -> ProposedAction:
    return ProposedAction(share_by_flow_id=dict(shares))


def test_a_clean_action_passes_through_unchanged() -> None:
    checked = check_action(_proposal(f0=0.5, f1=0.7), _state(), ActionBounds())

    assert checked.is_clean
    assert checked.action.share_by_flow_id == {"f0": 0.5, "f1": 0.7}


def test_an_out_of_range_share_is_clamped_not_refused() -> None:
    # Clamping preserves the policy's evident intent. Falling back over a rounding error would hand the
    # network to the baseline far too readily.
    checked = check_action(_proposal(f0=1.9, f1=0.5), _state(), ActionBounds())

    assert checked.action.share_by_flow_id["f0"] == 1.0
    assert not checked.requires_fallback
    assert checked.violations[0].kind is ViolationKind.SHARE_OUT_OF_RANGE
    assert checked.violations[0].corrected_to == 1.0


def test_a_negative_share_is_clamped_to_zero() -> None:
    checked = check_action(_proposal(f0=-0.4, f1=0.5), _state(), ActionBounds())

    assert checked.action.share_by_flow_id["f0"] == 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_share_escalates_because_no_repair_is_honest(bad: float) -> None:
    # 0.0 starves the flow, 1.0 floods it; both are inventions. The supervisor must not choose on the
    # policy's behalf, so this is the one violation that forces the deterministic allocator.
    checked = check_action(_proposal(f0=bad, f1=0.5), _state(), ActionBounds())

    assert checked.requires_fallback
    assert checked.violations[0].kind is ViolationKind.NON_FINITE_SHARE


def test_an_unknown_flow_is_dropped_and_named() -> None:
    checked = check_action(_proposal(f0=0.5, f1=0.5, f99=0.5), _state(), ActionBounds())

    assert "f99" not in checked.action.share_by_flow_id
    assert checked.violations[0].subject == "f99"
    assert checked.violations[0].kind is ViolationKind.UNKNOWN_FLOW


def test_an_omitted_flow_is_held_at_its_current_rate_not_zeroed() -> None:
    # Absent is not a decision to starve. Zeroing it would make the supervisor an allocator.
    checked = check_action(_proposal(f0=0.5), _state(), ActionBounds())

    assert checked.violations[0].kind is ViolationKind.MISSING_FLOW
    assert checked.action.share_by_flow_id["f1"] == pytest.approx(0.4)  # 40 allocated / 100 demand


def test_a_route_off_the_candidate_paths_is_dropped() -> None:
    # Applying it would carry traffic over links the topology does not connect.
    proposal = ProposedAction(
        share_by_flow_id={"f0": 0.5, "f1": 0.5},
        path_by_flow_id={"f0": ("n0->n9", "n9->n1")},
    )

    checked = check_action(proposal, _state(), ActionBounds())

    assert "f0" not in checked.action.path_by_flow_id
    assert checked.violations[0].kind is ViolationKind.INVALID_PATH


def test_a_valid_route_survives() -> None:
    proposal = ProposedAction(share_by_flow_id={"f0": 0.5, "f1": 0.5}, path_by_flow_id={"f0": ("n0->n1",)})

    checked = check_action(proposal, _state(), ActionBounds())

    assert checked.is_clean
    assert checked.action.path_by_flow_id["f0"] == ("n0->n1",)


def test_the_emitted_action_is_always_a_validated_allocation() -> None:
    # The supervisor inspects raw output and emits something the rest of the system can trust.
    checked = check_action(_proposal(f0=5.0, f1=-5.0), _state(), ActionBounds())

    assert all(0.0 <= share <= 1.0 for share in checked.action.share_by_flow_id.values())


def test_provenance_is_preserved_through_correction() -> None:
    # A corrected action is still the controller's decision. Relabelling it would misattribute behaviour.
    proposal = ProposedAction(share_by_flow_id={"f0": 1.5, "f1": 0.5}, issued_by="controller")

    assert check_action(proposal, _state(), ActionBounds()).action.issued_by == "controller"


def test_checking_is_pure() -> None:
    state, bounds, proposal = _state(), ActionBounds(), _proposal(f0=1.5, f1=0.2)

    first = check_action(proposal, state, bounds)
    second = check_action(proposal, state, bounds)

    assert first.action.share_by_flow_id == second.action.share_by_flow_id
    assert len(first.violations) == len(second.violations)


# --- Starvation, which is about behaviour over time -------------------------------------------------


def test_a_briefly_throttled_flow_is_not_starved() -> None:
    # A few steps of near-zero is a legitimate allocation decision under contention.
    tracker = StarvationTracker(ActionBounds(starvation_window=5))

    for _ in range(4):
        assert tracker.observe(_proposal(f0=0.0, f1=0.8)) == ()


def test_a_permanently_zeroed_flow_is_reported_once_the_window_closes() -> None:
    tracker = StarvationTracker(ActionBounds(starvation_window=3))

    violations = [tracker.observe(_proposal(f0=0.0, f1=0.8)) for _ in range(3)]

    assert violations[0] == violations[1] == ()
    assert violations[2][0].kind is ViolationKind.STARVATION
    assert violations[2][0].subject == "f0"


def test_starvation_is_reported_once_not_every_subsequent_step() -> None:
    # Otherwise one stuck flow buries every other finding.
    tracker = StarvationTracker(ActionBounds(starvation_window=2))
    for _ in range(2):
        tracker.observe(_proposal(f0=0.0))

    assert tracker.observe(_proposal(f0=0.0)) == ()
    assert tracker.starved_flows == ("f0",)


def test_recovery_resets_the_starvation_count() -> None:
    tracker = StarvationTracker(ActionBounds(starvation_window=3))
    tracker.observe(_proposal(f0=0.0))
    tracker.observe(_proposal(f0=0.0))

    tracker.observe(_proposal(f0=0.9))

    assert tracker.observe(_proposal(f0=0.0)) == ()
    assert tracker.starved_flows == ()


def test_a_near_zero_share_counts_as_starvation() -> None:
    # A policy can hold a flow at 0.001 indefinitely and starve it as thoroughly as at exactly zero.
    tracker = StarvationTracker(ActionBounds(starvation_window=2, starvation_share=0.01))

    tracker.observe(_proposal(f0=0.005))

    assert tracker.observe(_proposal(f0=0.005))[0].kind is ViolationKind.STARVATION
