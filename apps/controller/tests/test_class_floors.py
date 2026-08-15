"""Class-floor enforcement tests.

Regression: ``class_floor_mbps`` was accepted, validated as a real constraint, and advertised in ADR-0002,
SECURITY.md and CONTEXT.md — while no code read it. A floor-only constraint was a sanctioned no-op.
"""

from __future__ import annotations

import pytest

from nexus_controller.class_floors import apply_class_floors, merge_floors
from nexus_types.action import AllocationAction
from nexus_types.network_state import FlowState, LinkState, NetworkState
from nexus_types.traffic_class import TrafficClass


def _flow(flow_id: str, demand: float, traffic_class: TrafficClass) -> FlowState:
    return FlowState(
        flow_id=flow_id,
        source="n0",
        target="n1",
        traffic_class=traffic_class,
        demand_mbps=demand,
        allocated_mbps=0.0,
        dropped_mbps=0.0,
        path=("n0->n1",),
        latency_ms=1.0,
    )


def _state(*flows: FlowState) -> NetworkState:
    link = LinkState(
        link_id="n0->n1",
        source="n0",
        target="n1",
        capacity_mbps=500.0,
        effective_capacity_mbps=500.0,
        allocated_mbps=0.0,
        drained_mbps=0.0,
        queue_mb=0.0,
        utilization=0.0,
        latency_ms=1.0,
    )
    return NetworkState(timestep=1, links=(link,), flows=flows)


def _granted(action: AllocationAction, state: NetworkState, traffic_class: TrafficClass) -> float:
    return sum(
        flow.demand_mbps * action.share_by_flow_id[flow.flow_id]
        for flow in state.flows
        if flow.traffic_class is traffic_class
    )


def test_a_starved_class_is_raised_to_its_floor() -> None:
    state = _state(
        _flow("f1", 100.0, TrafficClass.LATENCY_SENSITIVE),
        _flow("f2", 100.0, TrafficClass.BULK),
    )
    throttled = AllocationAction(share_by_flow_id={"f1": 0.1, "f2": 1.0})

    result = apply_class_floors(throttled, state, {TrafficClass.LATENCY_SENSITIVE: 50.0})

    assert _granted(result, state, TrafficClass.LATENCY_SENSITIVE) == pytest.approx(50.0, rel=1e-3)


def test_a_class_already_above_its_floor_is_untouched() -> None:
    state = _state(_flow("f1", 100.0, TrafficClass.LATENCY_SENSITIVE))
    action = AllocationAction(share_by_flow_id={"f1": 0.9})

    result = apply_class_floors(action, state, {TrafficClass.LATENCY_SENSITIVE: 50.0})

    assert result.share_by_flow_id["f1"] == pytest.approx(0.9)


def test_other_classes_are_never_lowered_by_a_floor() -> None:
    # A floor is a guarantee for one class, not a cap on the others.
    state = _state(
        _flow("f1", 100.0, TrafficClass.LATENCY_SENSITIVE),
        _flow("f2", 100.0, TrafficClass.BULK),
    )
    action = AllocationAction(share_by_flow_id={"f1": 0.1, "f2": 1.0})

    result = apply_class_floors(action, state, {TrafficClass.LATENCY_SENSITIVE: 50.0})

    assert result.share_by_flow_id["f2"] == 1.0


def test_a_floor_cannot_manufacture_demand() -> None:
    # Asking for 200 Mb/s of a class that only wants 40 must not push shares above 1.0.
    state = _state(_flow("f1", 40.0, TrafficClass.BACKGROUND))
    action = AllocationAction(share_by_flow_id={"f1": 0.1})

    result = apply_class_floors(action, state, {TrafficClass.BACKGROUND: 200.0})

    assert result.share_by_flow_id["f1"] == pytest.approx(1.0)


def test_a_floor_on_an_absent_class_is_a_no_op() -> None:
    state = _state(_flow("f1", 100.0, TrafficClass.BULK))
    action = AllocationAction(share_by_flow_id={"f1": 0.5})

    result = apply_class_floors(action, state, {TrafficClass.LATENCY_SENSITIVE: 50.0})

    assert result.share_by_flow_id == {"f1": 0.5}


def test_a_class_with_no_demand_is_left_alone() -> None:
    state = _state(_flow("f1", 0.0, TrafficClass.BULK))
    action = AllocationAction(share_by_flow_id={"f1": 0.0})

    result = apply_class_floors(action, state, {TrafficClass.BULK: 50.0})

    assert result.share_by_flow_id["f1"] == 0.0


def test_no_floors_returns_the_action_unchanged() -> None:
    state = _state(_flow("f1", 100.0, TrafficClass.BULK))
    action = AllocationAction(share_by_flow_id={"f1": 0.5})

    assert apply_class_floors(action, state, {}) is action


def test_floors_from_several_constraints_take_the_strongest_per_class() -> None:
    merged = merge_floors(
        [
            {TrafficClass.LATENCY_SENSITIVE: 40.0, TrafficClass.BULK: 10.0},
            {TrafficClass.LATENCY_SENSITIVE: 90.0},
        ]
    )

    assert merged == {TrafficClass.LATENCY_SENSITIVE: 90.0, TrafficClass.BULK: 10.0}


def test_shares_stay_within_bounds_after_a_floor() -> None:
    state = _state(
        _flow("f1", 10.0, TrafficClass.BULK),
        _flow("f2", 1000.0, TrafficClass.BULK),
    )
    action = AllocationAction(share_by_flow_id={"f1": 0.01, "f2": 0.01})

    result = apply_class_floors(action, state, {TrafficClass.BULK: 900.0})

    assert all(0.0 <= s <= 1.0 for s in result.share_by_flow_id.values())
