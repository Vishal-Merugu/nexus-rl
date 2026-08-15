"""Flows — the unit of allocation.

A flow is a demand between two nodes with a traffic class and a set of candidate paths. The controller
decides how much capacity each flow gets and which of its candidate paths it takes; it does not move
individual packets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nexus_errors.simulation_error import TopologyError
from nexus_sim.constants import DEFAULT_CANDIDATE_PATHS, DEFAULT_FLOW_COUNT
from nexus_sim.topology import Topology
from nexus_types.traffic_class import TrafficClass

_CLASS_CYCLE = (
    TrafficClass.LATENCY_SENSITIVE,
    TrafficClass.BULK,
    TrafficClass.BACKGROUND,
    TrafficClass.BULK,
)
"""Class assignment cycle. Weighted toward BULK because that is the realistic mix, and because a network
of purely latency-sensitive flows makes the fairness term meaningless."""


@dataclass(frozen=True, slots=True)
class FlowSpec:
    """A flow's fixed properties. Demand varies per timestep; everything here does not."""

    flow_id: str
    source: str
    target: str
    traffic_class: TrafficClass
    base_demand_mbps: float
    candidate_paths: tuple[tuple[str, ...], ...]
    """Ordered link-id paths. The action space indexes into this, so the order is part of the contract."""

    @property
    def default_path(self) -> tuple[str, ...]:
        return self.candidate_paths[0]


def build_flows(
    topology: Topology,
    rng: np.random.Generator,
    flow_count: int = DEFAULT_FLOW_COUNT,
    candidate_paths: int = DEFAULT_CANDIDATE_PATHS,
) -> tuple[FlowSpec, ...]:
    """Construct a flow set over a topology.

    Deterministic given ``rng``: the same seed produces the same flows, which is what makes two runs
    comparable at all. Source/target pairs are drawn without replacement where possible so flows do not
    collapse onto one route.
    """
    if flow_count < 0:
        raise TopologyError("flow_count cannot be negative.", {"flow_count": flow_count})

    nodes = topology.nodes
    pairs: list[tuple[str, str]] = [
        (source, target) for source in nodes for target in nodes if source != target
    ]
    if flow_count > len(pairs):
        # Silently truncating here produced a flow set smaller than requested, so the observation came out
        # shorter than the space the environment had already declared — surfacing as a shape error deep
        # inside the training loop, far from the cause.
        raise TopologyError(
            f"Topology {topology.name!r} has only {len(pairs)} distinct source/target pairs, so it cannot "
            f"host {flow_count} flows. Use a larger topology or fewer flows.",
            {"topology": topology.name, "available_pairs": len(pairs), "requested": flow_count},
        )
    chosen_indices = rng.permutation(len(pairs))[:flow_count]

    flows: list[FlowSpec] = []
    for position, pair_index in enumerate(chosen_indices):
        source, target = pairs[int(pair_index)]
        traffic_class = _CLASS_CYCLE[position % len(_CLASS_CYCLE)]
        # Demand spread across flows so the network is meaningfully, not uniformly, loaded.
        base_demand = float(rng.uniform(20.0, 60.0))
        flows.append(
            FlowSpec(
                flow_id=f"f{position}",
                source=source,
                target=target,
                traffic_class=traffic_class,
                base_demand_mbps=base_demand,
                candidate_paths=topology.k_shortest_paths(source, target, candidate_paths),
            )
        )
    return tuple(flows)


def base_demands_mbps(flows: tuple[FlowSpec, ...]) -> NDArray[np.float32]:
    """Base demand per flow, in flow order."""
    return np.array([flow.base_demand_mbps for flow in flows], dtype=np.float32)


def path_incidence(
    flows: tuple[FlowSpec, ...],
    topology: Topology,
    path_choice: NDArray[np.int_] | None = None,
) -> NDArray[np.float32]:
    """Build the flow x link incidence matrix for the currently chosen paths.

    ``incidence[f, l] == 1`` when flow ``f``'s selected path traverses link ``l``. Everything downstream —
    per-link load, bottleneck resolution, end-to-end latency — is a matrix product against this, which is
    what keeps ``step()`` inside its time budget.

    Args:
        flows: Flow specs in array order.
        topology: The graph the flows run over.
        path_choice: Per-flow index into ``candidate_paths``. ``None`` selects every flow's default path.
            Out-of-range indices clamp to the last candidate rather than raising — action decoding must be
            total, and a policy emitting an index for a flow with fewer paths is expected, not exceptional.
    """
    index_of = topology.link_index()
    incidence = np.zeros((len(flows), len(topology.links)), dtype=np.float32)
    for flow_position, flow in enumerate(flows):
        if path_choice is None:
            path = flow.default_path
        else:
            candidate = int(path_choice[flow_position])
            path = flow.candidate_paths[min(candidate, len(flow.candidate_paths) - 1)]
        for link_id in path:
            incidence[flow_position, index_of[link_id]] = 1.0
    return incidence


def selected_paths(
    flows: tuple[FlowSpec, ...],
    path_choice: NDArray[np.int_] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """The path each flow is currently taking, in flow order."""
    if path_choice is None:
        return tuple(flow.default_path for flow in flows)
    return tuple(
        flow.candidate_paths[min(int(path_choice[position]), len(flow.candidate_paths) - 1)]
        for position, flow in enumerate(flows)
    )
