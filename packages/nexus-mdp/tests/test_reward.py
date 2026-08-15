"""Reward tests — including the sign tests every component must have.

A reward term without a sign test is not finished: the whole point is that an obviously-good action scores
above an obviously-bad one, and nothing else in the pipeline checks that.

The degenerate-optimum tests here are the highest-value ones in the file. RL finds whichever exploit is left
open, reliably, and the resulting policy looks successful while doing something an operator would object to.
"""

from __future__ import annotations

import pytest

from nexus_mdp.reward import RewardWeights, compute_reward, total_reward
from nexus_types.network_state import FlowState, LinkState, NetworkState
from nexus_types.traffic_class import TrafficClass


def _flow(
    flow_id: str = "f1",
    demand: float = 100.0,
    allocated: float = 100.0,
    dropped: float = 0.0,
    latency: float = 1.0,
    traffic_class: TrafficClass = TrafficClass.BULK,
) -> FlowState:
    return FlowState(
        flow_id=flow_id,
        source="n1",
        target="n2",
        traffic_class=traffic_class,
        demand_mbps=demand,
        allocated_mbps=allocated,
        dropped_mbps=dropped,
        path=("n1->n2",),
        latency_ms=latency,
    )


def _state(*flows: FlowState) -> NetworkState:
    link = LinkState(
        link_id="n1->n2",
        source="n1",
        target="n2",
        capacity_mbps=100.0,
        effective_capacity_mbps=100.0,
        allocated_mbps=50.0,
        queue_mb=0.0,
        utilization=0.5,
        latency_ms=1.0,
    )
    return NetworkState(timestep=1, links=(link,), flows=flows)


def test_every_component_is_present_and_named() -> None:
    components = compute_reward(_state(_flow()))

    assert set(components) == {
        "throughput_reward",
        "latency_penalty",
        "drop_penalty",
        "fairness_bonus",
    }


def test_serving_all_demand_beats_serving_none() -> None:
    served = total_reward(compute_reward(_state(_flow(allocated=100.0))))
    starved = total_reward(compute_reward(_state(_flow(allocated=0.0))))

    assert served > starved


def test_low_latency_beats_high_latency() -> None:
    fast = total_reward(compute_reward(_state(_flow(latency=1.0))))
    slow = total_reward(compute_reward(_state(_flow(latency=400.0))))

    assert fast > slow


def test_dropping_traffic_is_penalised() -> None:
    clean = total_reward(compute_reward(_state(_flow(allocated=50.0, dropped=0.0))))
    dropping = total_reward(compute_reward(_state(_flow(allocated=50.0, dropped=50.0))))

    assert dropping < clean


def test_fair_allocation_beats_skewed_allocation_at_equal_throughput() -> None:
    fair = total_reward(compute_reward(_state(_flow("f1", allocated=50.0), _flow("f2", allocated=50.0))))
    skewed = total_reward(compute_reward(_state(_flow("f1", allocated=100.0), _flow("f2", allocated=0.0))))

    assert fair > skewed


def test_latency_sensitive_delay_hurts_more_than_background_delay() -> None:
    interactive = total_reward(
        compute_reward(_state(_flow(latency=300.0, traffic_class=TrafficClass.LATENCY_SENSITIVE)))
    )
    background = total_reward(
        compute_reward(_state(_flow(latency=300.0, traffic_class=TrafficClass.BACKGROUND)))
    )

    assert interactive < background


def test_starving_everyone_equally_does_not_earn_fairness_credit() -> None:
    # The classic degenerate optimum: Jain's index scores 1.0 when every flow gets zero.
    starved = compute_reward(_state(_flow("f1", allocated=0.0), _flow("f2", allocated=0.0)))

    assert starved["fairness_bonus"] == pytest.approx(0.0)


def test_dropping_traffic_to_cut_latency_is_not_profitable() -> None:
    # Shedding the congested flow makes the survivor look fast. The drop penalty must exceed that gain.
    congested = _state(
        _flow("f1", allocated=100.0, latency=5.0),
        _flow("f2", demand=100.0, allocated=60.0, latency=350.0),
    )
    shed = _state(
        _flow("f1", allocated=100.0, latency=5.0),
        _flow("f2", demand=100.0, allocated=0.0, dropped=100.0, latency=1.0),
    )

    assert total_reward(compute_reward(shed)) < total_reward(compute_reward(congested))


def test_an_idle_network_scores_neutral_rather_than_positive() -> None:
    # Scoring an empty network well would teach the policy to suppress demand.
    components = compute_reward(_state(_flow(demand=0.0, allocated=0.0)))

    assert total_reward(components) == pytest.approx(0.0)


def test_components_stay_within_comparable_magnitudes() -> None:
    # No term may dominate by orders of magnitude — the others become invisible to the policy.
    components = compute_reward(
        _state(
            _flow("f1", demand=100.0, allocated=10.0, dropped=90.0, latency=500.0),
            _flow("f2", demand=100.0, allocated=100.0, latency=1.0),
        )
    )

    magnitudes = [abs(value) for value in components.values()]
    assert max(magnitudes) <= 4.0


def test_weights_are_respected() -> None:
    state = _state(_flow(latency=400.0))

    light = compute_reward(state, RewardWeights(latency=0.1))
    heavy = compute_reward(state, RewardWeights(latency=5.0))

    assert abs(heavy["latency_penalty"]) > abs(light["latency_penalty"])


def test_total_is_the_sum_of_the_components() -> None:
    components = compute_reward(_state(_flow(allocated=40.0, dropped=10.0, latency=50.0)))

    assert total_reward(components) == pytest.approx(sum(components.values()))
