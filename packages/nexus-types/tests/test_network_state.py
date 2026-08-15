from __future__ import annotations

import pytest
from pydantic import ValidationError

from nexus_types.network_state import FlowState, LinkState, NetworkState
from nexus_types.traffic_class import TrafficClass


def _link(link_id: str = "n1->n2", utilization: float = 0.5) -> LinkState:
    return LinkState(
        link_id=link_id,
        source=link_id.split("->")[0],
        target=link_id.split("->")[1],
        capacity_mbps=100.0,
        effective_capacity_mbps=100.0,
        allocated_mbps=utilization * 100.0,
        queue_mb=0.0,
        utilization=utilization,
        latency_ms=1.0,
    )


def _flow(demand: float = 100.0, allocated: float = 50.0, dropped: float = 0.0) -> FlowState:
    return FlowState(
        flow_id="f1",
        source="n1",
        target="n2",
        traffic_class=TrafficClass.BULK,
        demand_mbps=demand,
        allocated_mbps=allocated,
        dropped_mbps=dropped,
        path=("n1->n2",),
        latency_ms=1.0,
    )


def test_state_is_frozen_so_handed_out_snapshots_stay_valid() -> None:
    state = NetworkState(timestep=0, links=(_link(),), flows=(_flow(),))

    with pytest.raises(ValidationError):
        state.timestep = 1  # type: ignore[misc]  # the point of the test is that this is illegal


def test_utilization_above_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _link(utilization=1.2)


def test_negative_queue_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LinkState(
            link_id="n1->n2",
            source="n1",
            target="n2",
            capacity_mbps=100.0,
            effective_capacity_mbps=100.0,
            allocated_mbps=0.0,
            queue_mb=-1.0,
            utilization=0.0,
            latency_ms=1.0,
        )


def test_satisfaction_ratio_is_allocated_over_requested() -> None:
    assert _flow(demand=100.0, allocated=25.0).satisfaction_ratio == 0.25


def test_flow_with_no_demand_is_fully_satisfied() -> None:
    assert _flow(demand=0.0, allocated=0.0).satisfaction_ratio == 1.0


def test_satisfaction_ratio_never_exceeds_one() -> None:
    assert _flow(demand=10.0, allocated=50.0).satisfaction_ratio == 1.0


def test_totals_aggregate_across_flows() -> None:
    state = NetworkState(
        timestep=3,
        links=(_link(),),
        flows=(_flow(demand=100.0, allocated=40.0, dropped=10.0), _flow(demand=50.0, allocated=50.0)),
    )

    assert state.total_demand_mbps == 150.0
    assert state.total_allocated_mbps == 90.0
    assert state.total_dropped_mbps == 10.0


def test_link_lookup_raises_on_an_unknown_id() -> None:
    state = NetworkState(timestep=0, links=(_link("n1->n2"),), flows=())

    assert state.link_by_id("n1->n2").link_id == "n1->n2"
    with pytest.raises(KeyError):
        state.link_by_id("nope->nope")
