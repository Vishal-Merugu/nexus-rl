from __future__ import annotations

import numpy as np
import pytest

from nexus_mdp.action_space import (
    DISCRETE_ACTION_COUNT,
    AllocationMode,
    FlowRoutingOptions,
    decode_continuous_action,
    decode_discrete_action,
)
from nexus_sim.simulator import NetworkSimulator
from nexus_types.traffic_class import TrafficClass


def _simulator() -> NetworkSimulator:
    return NetworkSimulator.from_config("mesh-6", "steady", seed=4)


def _routing(simulator: NetworkSimulator) -> tuple[FlowRoutingOptions, ...]:
    return tuple(
        FlowRoutingOptions(flow_id=flow.flow_id, candidate_paths=flow.candidate_paths)
        for flow in simulator.flows
    )


@pytest.mark.parametrize("action_index", range(DISCRETE_ACTION_COUNT))
def test_every_mode_decodes_to_a_share_for_every_flow(action_index: int) -> None:
    simulator = _simulator()

    action = decode_discrete_action(action_index, simulator.state, _routing(simulator))

    assert set(action.share_by_flow_id) == {flow.flow_id for flow in simulator.flows}
    assert all(0.0 <= share <= 1.0 for share in action.share_by_flow_id.values())


def test_open_mode_grants_full_demand_to_everything() -> None:
    simulator = _simulator()

    action = decode_discrete_action(AllocationMode.OPEN, simulator.state, _routing(simulator))

    assert all(share == 1.0 for share in action.share_by_flow_id.values())


def test_aggressive_throttle_is_tighter_than_moderate() -> None:
    simulator = _simulator()
    routing = _routing(simulator)

    moderate = decode_discrete_action(AllocationMode.THROTTLE_MODERATE, simulator.state, routing)
    aggressive = decode_discrete_action(AllocationMode.THROTTLE_AGGRESSIVE, simulator.state, routing)

    for flow_id, share in aggressive.share_by_flow_id.items():
        assert share < moderate.share_by_flow_id[flow_id]


def test_prioritising_latency_favours_latency_sensitive_flows() -> None:
    simulator = _simulator()

    action = decode_discrete_action(
        AllocationMode.PRIORITISE_LATENCY_SENSITIVE, simulator.state, _routing(simulator)
    )

    for flow in simulator.state.flows:
        share = action.share_by_flow_id[flow.flow_id]
        if flow.traffic_class is TrafficClass.LATENCY_SENSITIVE:
            assert share == 1.0
        else:
            assert share < 1.0


def test_shedding_background_leaves_other_classes_untouched() -> None:
    simulator = _simulator()

    action = decode_discrete_action(AllocationMode.SHED_BACKGROUND, simulator.state, _routing(simulator))

    for flow in simulator.state.flows:
        share = action.share_by_flow_id[flow.flow_id]
        if flow.traffic_class is TrafficClass.BACKGROUND:
            assert share < 0.5
        else:
            assert share == 1.0


def test_rerouting_modes_choose_a_path_for_every_flow() -> None:
    simulator = _simulator()

    action = decode_discrete_action(AllocationMode.REROUTE_LEAST_LOADED, simulator.state, _routing(simulator))

    assert set(action.path_by_flow_id) == {flow.flow_id for flow in simulator.flows}


def test_non_rerouting_modes_leave_paths_alone() -> None:
    simulator = _simulator()

    action = decode_discrete_action(AllocationMode.OPEN, simulator.state, _routing(simulator))

    assert action.path_by_flow_id == {}


def test_chosen_paths_are_always_real_candidates() -> None:
    simulator = _simulator()
    candidates = {flow.flow_id: flow.candidate_paths for flow in simulator.flows}

    action = decode_discrete_action(AllocationMode.REROUTE_LEAST_LOADED, simulator.state, _routing(simulator))

    for flow_id, path in action.path_by_flow_id.items():
        assert path in candidates[flow_id]


def test_forbidden_links_are_avoided_when_an_alternative_exists() -> None:
    simulator = _simulator()
    flow = simulator.flows[0]
    forbidden = frozenset(flow.candidate_paths[0])

    action = decode_discrete_action(
        AllocationMode.OPEN, simulator.state, _routing(simulator), forbidden_link_ids=forbidden
    )

    chosen = action.path_by_flow_id[flow.flow_id]
    if len(flow.candidate_paths) > 1:
        assert not set(chosen) & forbidden


def test_a_flow_is_never_left_without_a_path() -> None:
    # A constraint must not be able to black-hole traffic — that is exactly what constrained authority
    # exists to prevent.
    simulator = _simulator()
    flow = simulator.flows[0]
    forbidden = frozenset(link for path in flow.candidate_paths for link in path)

    action = decode_discrete_action(
        AllocationMode.OPEN, simulator.state, _routing(simulator), forbidden_link_ids=forbidden
    )

    assert action.path_by_flow_id[flow.flow_id] in flow.candidate_paths


def test_out_of_range_indices_clamp_into_the_mode_set() -> None:
    simulator = _simulator()
    routing = _routing(simulator)

    high = decode_discrete_action(999, simulator.state, routing)
    low = decode_discrete_action(-5, simulator.state, routing)

    assert high.share_by_flow_id
    assert low.share_by_flow_id


def test_continuous_shares_map_positionally_onto_flows() -> None:
    simulator = _simulator()
    shares = np.linspace(0.0, 1.0, len(simulator.flows), dtype=np.float32)

    action = decode_continuous_action(shares, simulator.state)

    for index, flow in enumerate(simulator.state.flows):
        assert action.share_by_flow_id[flow.flow_id] == pytest.approx(float(shares[index]))


def test_continuous_shares_outside_the_box_are_clipped() -> None:
    simulator = _simulator()
    shares = np.array([-3.0] + [7.0] * (len(simulator.flows) - 1), dtype=np.float32)

    action = decode_continuous_action(shares, simulator.state)

    assert all(0.0 <= share <= 1.0 for share in action.share_by_flow_id.values())


def test_a_short_continuous_vector_defaults_the_remaining_flows_to_full_demand() -> None:
    simulator = _simulator()

    action = decode_continuous_action(np.array([0.5], dtype=np.float32), simulator.state)

    assert len(action.share_by_flow_id) == len(simulator.flows)
    assert action.share_by_flow_id[simulator.flows[-1].flow_id] == 1.0
