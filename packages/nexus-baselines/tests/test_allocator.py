from __future__ import annotations

import pytest

from nexus_baselines.allocator import (
    GreedyThroughputAllocator,
    RoundRobinAllocator,
    StaticAllocator,
    available_baselines,
    build_baseline,
)
from nexus_errors.config_error import ConfigurationError
from nexus_sim.simulator import NetworkSimulator
from nexus_sim.topology import build_topology


def _simulator(pattern: str = "steady") -> NetworkSimulator:
    return NetworkSimulator.from_config("mesh-6", pattern, seed=2)


@pytest.mark.parametrize("name", available_baselines())
def test_every_baseline_produces_shares_within_bounds(name: str) -> None:
    simulator = _simulator()
    allocator = build_baseline(name, simulator.topology, len(simulator.flows))

    for _ in range(30):
        action = allocator.allocate(simulator.state)
        assert all(0.0 <= share <= 1.0 for share in action.share_by_flow_id.values())
        simulator.step(action)


@pytest.mark.parametrize("name", available_baselines())
def test_every_baseline_records_itself_in_provenance(name: str) -> None:
    simulator = _simulator()
    allocator = build_baseline(name, simulator.topology, len(simulator.flows))

    assert allocator.allocate(simulator.state).issued_by == f"baseline:{name}"


@pytest.mark.parametrize("name", available_baselines())
def test_every_baseline_keeps_the_network_within_its_invariants(name: str) -> None:
    simulator = _simulator("bursty")
    allocator = build_baseline(name, simulator.topology, len(simulator.flows))

    for _ in range(200):
        state = simulator.step(allocator.allocate(simulator.state)).state
        for link in state.links:
            assert 0.0 <= link.utilization <= 1.0
            assert link.queue_mb >= 0.0


def test_unknown_baseline_is_rejected() -> None:
    topology = build_topology("mesh-6")

    with pytest.raises(ConfigurationError, match="Unknown baseline"):
        build_baseline("psychic", topology, 8)


def test_static_grants_the_same_absolute_rate_regardless_of_demand() -> None:
    simulator = _simulator()
    allocator = StaticAllocator(equal_rate_mbps=20.0)

    action = allocator.allocate(simulator.state)

    for flow in simulator.state.flows:
        expected_rate = min(flow.demand_mbps, 20.0)
        granted_rate = action.share_by_flow_id[flow.flow_id] * flow.demand_mbps
        assert granted_rate == pytest.approx(expected_rate, rel=1e-4)


def test_static_rate_derived_from_a_topology_is_a_usable_fraction_of_capacity() -> None:
    # Guards against a strawman baseline: the derived rate must be a reasonable operating point, not a
    # setting chosen to make the policy look good.
    topology = build_topology("mesh-6")

    allocator = StaticAllocator.for_topology(topology, flow_count=8)
    action = allocator.allocate(_simulator().state)

    assert any(share > 0.2 for share in action.share_by_flow_id.values())


def test_round_robin_prioritises_a_different_group_over_time() -> None:
    simulator = _simulator()
    allocator = RoundRobinAllocator(group_size=2, rotation_period=1)

    first = allocator.allocate(simulator.state)
    simulator.step(first)
    second = allocator.allocate(simulator.state)

    assert first.share_by_flow_id != second.share_by_flow_id


def test_round_robin_eventually_prioritises_every_flow() -> None:
    simulator = _simulator()
    allocator = RoundRobinAllocator(group_size=1, rotation_period=1)

    prioritised: set[str] = set()
    for _ in range(len(simulator.flows) * 2):
        action = allocator.allocate(simulator.state)
        prioritised.update(flow_id for flow_id, share in action.share_by_flow_id.items() if share == 1.0)
        simulator.step(action)

    assert prioritised == {flow.flow_id for flow in simulator.flows}


def test_round_robin_handles_a_network_with_no_flows() -> None:
    simulator = NetworkSimulator.from_config("mesh-6", "steady", seed=1, flow_count=0)

    action = RoundRobinAllocator().allocate(simulator.state)

    assert action.share_by_flow_id == {}


def test_greedy_throttles_flows_on_congested_paths() -> None:
    simulator = _simulator("flash_crowd")
    allocator = GreedyThroughputAllocator(congestion_threshold=0.5)
    for _ in range(250):  # drive the network into congestion
        simulator.step()

    action = allocator.allocate(simulator.state)

    assert any(share < 1.0 for share in action.share_by_flow_id.values())


def test_greedy_leaves_an_uncongested_network_alone() -> None:
    simulator = NetworkSimulator.from_config("mesh-6", "steady", seed=2, flow_count=2)
    allocator = GreedyThroughputAllocator(congestion_threshold=0.99)

    action = allocator.allocate(simulator.state)

    assert all(share == 1.0 for share in action.share_by_flow_id.values())
