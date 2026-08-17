"""Per-slice SLA attainment, and the aggregates it must not collapse into.

The point of slicing is differentiated promises. Most of these tests pin down the ways a per-slice measure
can quietly become an aggregate again — which would restore exactly the blindness v1 had.
"""

from __future__ import annotations

import pytest

from nexus_errors.config_error import ConfigurationError
from nexus_sim.simulator import NetworkSimulator
from nexus_sim.slicing import all_attainments, attainment_summary, slice_attainment
from nexus_types.network_state import FlowState, LinkState, NetworkState
from nexus_types.slice import (
    DEFAULT_SLICES,
    NetworkSlice,
    SliceSet,
    SliceSla,
    SliceTier,
)
from nexus_types.traffic_class import TrafficClass


def _premium_slice() -> NetworkSlice:
    """The premium slice, narrowed. ``by_id`` returns an optional and a module-level assert does not
    narrow across function boundaries."""
    network_slice = DEFAULT_SLICES.by_id("premium")
    assert network_slice is not None
    return network_slice


_PREMIUM = _premium_slice()


def _flow(
    flow_id: str,
    traffic_class: TrafficClass,
    latency_ms: float = 5.0,
    demand: float = 100.0,
    allocated: float = 100.0,
    dropped: float = 0.0,
) -> FlowState:
    return FlowState(
        flow_id=flow_id,
        source="n0",
        target="n1",
        traffic_class=traffic_class,
        demand_mbps=demand,
        allocated_mbps=allocated,
        dropped_mbps=dropped,
        path=("n0->n1",),
        candidate_paths=(("n0->n1",),),
        latency_ms=latency_ms,
    )


def _slices_in(summary: dict[str, object]) -> list[dict[str, object]]:
    """Narrow the summary's slice list. Tool-shaped payloads are ``object`` on purpose; narrowing here
    keeps the assertions honest without a ``cast`` that would hide a real contract change."""
    rows = summary["slices"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
    return [dict(row) for row in rows]


def _state(*flows: FlowState) -> NetworkState:
    return NetworkState(
        timestep=1,
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
        flows=flows,
    )


# --- The slice contract -----------------------------------------------------------------------------


def test_two_slices_cannot_claim_one_traffic_class() -> None:
    # Membership is derived from the class, so this would make "which slice" ambiguous and every
    # per-slice number silently dependent on iteration order.
    with pytest.raises(ConfigurationError, match="ambiguous"):
        SliceSet(
            slices=(
                NetworkSlice(
                    slice_id="a",
                    tier=SliceTier.PREMIUM,
                    traffic_class=TrafficClass.BULK,
                    sla=SliceSla(max_latency_ms=10.0, min_satisfaction=0.9),
                ),
                NetworkSlice(
                    slice_id="b",
                    tier=SliceTier.STANDARD,
                    traffic_class=TrafficClass.BULK,
                    sla=SliceSla(max_latency_ms=50.0, min_satisfaction=0.5),
                ),
            )
        )


def test_priority_is_derived_from_the_tier_not_configured_separately() -> None:
    # Two fields that must agree are a field that will eventually disagree.
    by_tier = {s.tier: s.priority for s in DEFAULT_SLICES.slices}

    assert by_tier[SliceTier.PREMIUM] > by_tier[SliceTier.STANDARD] > by_tier[SliceTier.BEST_EFFORT]


def test_slices_are_ordered_by_descending_priority() -> None:
    assert [s.slice_id for s in DEFAULT_SLICES.ordered] == ["premium", "standard", "best-effort"]


# --- Attainment -------------------------------------------------------------------------------------


def test_a_slice_meeting_every_term_is_met() -> None:
    attainment = slice_attainment(
        _state(_flow("f0", TrafficClass.LATENCY_SENSITIVE, latency_ms=5.0)), _PREMIUM
    )

    assert attainment.is_met
    assert attainment.headroom > 0.0


def test_an_sla_is_a_conjunction_not_an_average() -> None:
    # Excellent latency must not compensate for a breached satisfaction floor.
    attainment = slice_attainment(
        _state(_flow("f0", TrafficClass.LATENCY_SENSITIVE, latency_ms=1.0, demand=100.0, allocated=10.0)),
        _PREMIUM,
    )

    assert attainment.latency_met
    assert not attainment.satisfaction_met
    assert not attainment.is_met


def test_headroom_gives_a_gradient_rather_than_a_step() -> None:
    # A met/unmet boolean tells the policy nothing about whether it is getting warmer, which is how it
    # settles just past a limit it could cheaply have cleared.
    comfortable = slice_attainment(
        _state(_flow("f0", TrafficClass.LATENCY_SENSITIVE, latency_ms=2.0)), _PREMIUM
    )
    marginal = slice_attainment(
        _state(_flow("f0", TrafficClass.LATENCY_SENSITIVE, latency_ms=24.0)), _PREMIUM
    )

    assert comfortable.headroom > marginal.headroom
    assert comfortable.is_met
    assert marginal.is_met


def test_headroom_is_zero_once_a_term_is_breached() -> None:
    breached = slice_attainment(
        _state(_flow("f0", TrafficClass.LATENCY_SENSITIVE, latency_ms=500.0)), _PREMIUM
    )

    assert breached.headroom == 0.0


def test_an_empty_slice_is_compliant_rather_than_breached() -> None:
    # Otherwise the policy is penalised for traffic that does not exist, and rewarded for creating it.
    attainment = slice_attainment(_state(_flow("f0", TrafficClass.BULK)), _PREMIUM)

    assert attainment.flows == 0
    assert attainment.is_met
    assert attainment.headroom == 1.0


def test_the_attainment_carries_the_sla_it_was_measured_against() -> None:
    # A stored attainment read against a different promise would report the wrong verdict.
    attainment = slice_attainment(_state(_flow("f0", TrafficClass.LATENCY_SENSITIVE)), _PREMIUM)

    assert attainment.sla == _PREMIUM.sla


# --- The aggregate must not hide a breach -----------------------------------------------------------


def test_a_healthy_best_effort_slice_cannot_mask_a_breached_premium_one() -> None:
    # This is the exact blindness per-slice measurement exists to prevent: aggregate latency here looks
    # fine, and the premium promise is broken.
    state = _state(
        _flow("f0", TrafficClass.LATENCY_SENSITIVE, latency_ms=400.0),
        _flow("f1", TrafficClass.BACKGROUND, latency_ms=1.0),
        _flow("f2", TrafficClass.BACKGROUND, latency_ms=1.0),
    )

    summary = attainment_summary(state)
    premium = next(s for s in _slices_in(summary) if s["slice_id"] == "premium")
    breached = premium["breached"]

    assert premium["sla_met"] is False
    assert isinstance(breached, list)
    assert "latency" in breached
    assert summary["slices_met"] == 2


def test_the_summary_names_which_terms_breached() -> None:
    # "SLA not met" is not actionable; "latency and satisfaction" is.
    state = _state(
        _flow("f0", TrafficClass.LATENCY_SENSITIVE, latency_ms=400.0, demand=100.0, allocated=10.0)
    )

    premium = next(s for s in _slices_in(attainment_summary(state)) if s["slice_id"] == "premium")
    breached = premium["breached"]

    assert isinstance(breached, list)
    assert set(breached) == {"latency", "satisfaction"}


def test_no_mean_across_slices_is_reported() -> None:
    # A mean would let a satisfied best-effort slice mask a breached premium one — the failure the whole
    # module exists to avoid.
    summary = attainment_summary(_state(_flow("f0", TrafficClass.LATENCY_SENSITIVE)))

    assert "mean_attainment" not in summary
    assert summary["slices_total"] == 3


# --- Under real contention (Use Case 6) -------------------------------------------------------------


def test_contention_breaches_slices_in_a_measurable_way() -> None:
    # The unmanaged network is the problem statement: under a flash crowd it fails the premium slice,
    # which is what a slice-aware policy has to fix.
    simulator = NetworkSimulator.from_config("mesh-6", "flash_crowd", seed=0, flow_count=9)
    for _ in range(30):
        simulator.step(None)

    attainments = all_attainments(simulator.state)

    assert any(not a.is_met for a in attainments)
    assert all(a.flows > 0 for a in attainments)


def test_membership_needs_no_configuration() -> None:
    # The state already carries the traffic class; nothing has to be told which flows are in which slice.
    simulator = NetworkSimulator.from_config("mesh-6", "diurnal", seed=1, flow_count=9)

    total = sum(a.flows for a in all_attainments(simulator.state))

    assert total == len(simulator.state.flows)
