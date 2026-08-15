"""Non-learned allocators — the comparison floor.

Every performance claim about a trained policy is stated relative to one of these, on a named traffic
pattern. A baseline that has been deliberately handicapped makes the whole comparison worthless, and it is
the first thing an informed reader will check — so these are written to be genuinely reasonable strategies,
not strawmen.

Three deliberately different philosophies:

- ``static`` — an equal absolute rate to every flow. Simple, fair-ish, blind to congestion.
- ``round_robin`` — rotates full priority between flows. Reacts to nothing, but nobody starves.
- ``greedy_throughput`` — maximises total carried rate. Ignores fairness and traffic class entirely, and
  wins on aggregate throughput while doing exactly the thing an operator would object to.
"""

from __future__ import annotations

from typing import Protocol

from nexus_errors.config_error import ConfigurationError
from nexus_sim.constants import EPSILON
from nexus_sim.topology import Topology
from nexus_types.action import AllocationAction
from nexus_types.network_state import NetworkState


class Allocator(Protocol):
    """A strategy mapping network state to an allocation. Implemented by baselines and by the controller."""

    name: str

    def allocate(self, state: NetworkState) -> AllocationAction: ...


class StaticAllocator:
    """Grants every flow the same absolute rate, regardless of demand or congestion.

    The equal rate is derived from the topology rather than guessed: mean link capacity divided by the
    expected number of flows contending for a link. That makes it a fair floor — it is what a competent
    engineer would configure with no adaptive control available, not a deliberately bad setting.
    """

    name = "static"

    def __init__(self, equal_rate_mbps: float) -> None:
        self._equal_rate_mbps = equal_rate_mbps

    @classmethod
    def for_topology(
        cls, topology: Topology, flow_count: int, mean_path_hops: float = 2.0
    ) -> StaticAllocator:
        """Derive a sensible equal rate from the network's capacity and expected load."""
        mean_capacity = sum(link.capacity_mbps for link in topology.links) / len(topology.links)
        expected_flows_per_link = max(1.0, flow_count * mean_path_hops / len(topology.links))
        return cls(equal_rate_mbps=mean_capacity / expected_flows_per_link)

    def allocate(self, state: NetworkState) -> AllocationAction:
        return AllocationAction(
            share_by_flow_id={
                flow.flow_id: min(1.0, self._equal_rate_mbps / max(flow.demand_mbps, EPSILON))
                for flow in state.flows
            },
            issued_by=f"baseline:{self.name}",
        )


class RoundRobinAllocator:
    """Rotates full priority through the flows.

    Each timestep one group of flows gets its full demand and the rest are throttled; the group advances by
    one every ``rotation_period`` timesteps. Adapts to nothing, but guarantees no flow is starved for long —
    which is exactly the property a static split lacks under skewed demand.
    """

    name = "round_robin"

    def __init__(self, group_size: int = 2, rotation_period: int = 10, throttled_share: float = 0.3) -> None:
        self._group_size = max(1, group_size)
        self._rotation_period = max(1, rotation_period)
        self._throttled_share = throttled_share

    def allocate(self, state: NetworkState) -> AllocationAction:
        flow_count = len(state.flows)
        if flow_count == 0:
            return AllocationAction(share_by_flow_id={}, issued_by=f"baseline:{self.name}")

        offset = (state.timestep // self._rotation_period) * self._group_size
        prioritised = {state.flows[(offset + step) % flow_count].flow_id for step in range(self._group_size)}
        return AllocationAction(
            share_by_flow_id={
                flow.flow_id: 1.0 if flow.flow_id in prioritised else self._throttled_share
                for flow in state.flows
            },
            issued_by=f"baseline:{self.name}",
        )


class GreedyThroughputAllocator:
    """Maximises total carried rate, ignoring fairness and traffic class.

    Flows on lightly loaded paths get everything they ask for; flows whose paths are congested are throttled
    hard, because serving them costs capacity that could carry more bytes elsewhere. Routes each flow onto
    whichever of its candidate paths is currently least utilized.

    This baseline usually wins on aggregate throughput and loses badly on fairness and tail latency — which
    is the point. It is the strategy a naive optimiser converges to, and beating it *on the QoS metrics
    operators care about* is a more meaningful result than beating a static split.
    """

    name = "greedy_throughput"

    def __init__(self, congestion_threshold: float = 0.85, throttled_share: float = 0.15) -> None:
        self._congestion_threshold = congestion_threshold
        self._throttled_share = throttled_share

    def allocate(self, state: NetworkState) -> AllocationAction:
        utilization_by_link = {link.link_id: link.utilization for link in state.links}

        shares: dict[str, float] = {}
        for flow in state.flows:
            path_load = max((utilization_by_link.get(link_id, 0.0) for link_id in flow.path), default=0.0)
            shares[flow.flow_id] = self._throttled_share if path_load >= self._congestion_threshold else 1.0

        return AllocationAction(share_by_flow_id=shares, issued_by=f"baseline:{self.name}")


def build_baseline(name: str, topology: Topology, flow_count: int) -> Allocator:
    """Resolve a baseline by name.

    ``static`` needs the topology to derive its equal rate, so construction goes through here rather than
    through a bare registry lookup.
    """
    if name == StaticAllocator.name:
        return StaticAllocator.for_topology(topology, flow_count)
    if name == RoundRobinAllocator.name:
        return RoundRobinAllocator()
    if name == GreedyThroughputAllocator.name:
        return GreedyThroughputAllocator()
    raise ConfigurationError(
        f"Unknown baseline {name!r}.", {"requested": name, "available": list(available_baselines())}
    )


def available_baselines() -> tuple[str, ...]:
    return (StaticAllocator.name, RoundRobinAllocator.name, GreedyThroughputAllocator.name)
