"""The simulator — the single writer of simulation state.

Pure and seeded: the same seed plus the same action sequence produces an identical trajectory, always. That
property is what makes every other kind of debugging in this project possible, so it is defended in tests
before anything else.

The simulator object holds mutable internal state (timestep, queues) because it *is* the authoritative
writer. Everything it emits is frozen: ``step()`` returns a new ``NetworkState`` rather than mutating the
previous one, so snapshots already handed to the dashboard or the agent layer stay valid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nexus_sim.allocation import allocate
from nexus_sim.constants import DEFAULT_FLOW_COUNT
from nexus_sim.flow import FlowSpec, base_demands_mbps, build_flows, path_incidence, selected_paths
from nexus_sim.incident import effective_capacities_mbps
from nexus_sim.latency import flow_latency_ms, link_latency_ms
from nexus_sim.metrics import StepMetrics, compute_metrics
from nexus_sim.topology import Topology, build_topology
from nexus_sim.traffic import TrafficPattern, get_traffic_pattern
from nexus_types.action import AllocationAction
from nexus_types.incident import Incident
from nexus_types.network_state import FlowState, LinkState, NetworkState


@dataclass(frozen=True, slots=True)
class StepResult:
    """One timestep's output: the new state and its QoS metrics."""

    state: NetworkState
    metrics: StepMetrics


class NetworkSimulator:
    """A flow-level network simulator.

    Args:
        topology: The graph traffic runs over.
        flows: Flow specs. Fixed for the lifetime of the simulator.
        traffic_pattern: The demand generator.
        rng: Injected generator. Every stochastic draw goes through it — there is no global RNG anywhere in
            this package, which is what makes runs reproducible.
        incidents: Scheduled degradations. Part of the environment, not the action space.
    """

    def __init__(
        self,
        topology: Topology,
        flows: tuple[FlowSpec, ...],
        traffic_pattern: TrafficPattern,
        rng: np.random.Generator,
        incidents: tuple[Incident, ...] = (),
    ) -> None:
        self._topology = topology
        self._flows = flows
        self._pattern = traffic_pattern
        self._rng = rng
        self._incidents = incidents

        self._nominal_capacity = topology.capacities_mbps()
        self._base_demand = base_demands_mbps(flows)
        self._traffic_classes = tuple(flow.traffic_class for flow in flows)
        self._link_index = topology.link_index()

        self._timestep = 0
        self._queue_mb = np.zeros(len(topology.links), dtype=np.float32)
        self._state, self._metrics = self._simulate(action=None)

    @classmethod
    def from_config(
        cls,
        topology_name: str,
        traffic_pattern_name: str,
        seed: int,
        flow_count: int = DEFAULT_FLOW_COUNT,
        incidents: tuple[Incident, ...] = (),
    ) -> NetworkSimulator:
        """Build a simulator from named components and a seed.

        The flow set is derived from the same seed, so a run config of (topology, pattern, seed, flow_count)
        fully determines the simulation.
        """
        topology = build_topology(topology_name)
        rng = np.random.default_rng(seed)
        flows = build_flows(topology, rng, flow_count=flow_count)
        return cls(
            topology=topology,
            flows=flows,
            traffic_pattern=get_traffic_pattern(traffic_pattern_name),
            rng=rng,
            incidents=incidents,
        )

    @property
    def state(self) -> NetworkState:
        """The current state. Frozen — safe to hand out and hold."""
        return self._state

    @property
    def metrics(self) -> StepMetrics:
        return self._metrics

    @property
    def timestep(self) -> int:
        return self._timestep

    @property
    def topology(self) -> Topology:
        return self._topology

    @property
    def flows(self) -> tuple[FlowSpec, ...]:
        return self._flows

    @property
    def incidents(self) -> tuple[Incident, ...]:
        return self._incidents

    def set_incidents(self, incidents: tuple[Incident, ...]) -> None:
        """Replace the scheduled incidents.

        Takes effect from the next timestep. Incidents are environment, not action space — this exists for
        operator and demo injection, and is deliberately not reachable from the policy or the agent layer.
        """
        self._incidents = incidents

    def reset(self, rng: np.random.Generator | None = None) -> NetworkState:
        """Reset to timestep 0 with empty queues.

        Args:
            rng: Replaces the generator when given. Pass a fresh one seeded identically to replay a
                trajectory exactly; omit it to continue drawing from the existing stream.
        """
        if rng is not None:
            self._rng = rng
        self._timestep = 0
        self._queue_mb = np.zeros(len(self._topology.links), dtype=np.float32)
        self._state, self._metrics = self._simulate(action=None)
        return self._state

    def step(self, action: AllocationAction | None = None) -> StepResult:
        """Advance one timestep and apply an action.

        Args:
            action: The controller's allocation. ``None`` means every flow requests its full demand on its
                default path — the "no controller" case, which is what the simulator does before a policy is
                attached and what makes Stage A runnable on its own.
        """
        self._timestep += 1
        self._state, self._metrics = self._simulate(action)
        return StepResult(state=self._state, metrics=self._metrics)

    def _simulate(self, action: AllocationAction | None) -> tuple[NetworkState, StepMetrics]:
        """Compute one timestep. The hot path — millions of calls per training run."""
        demand_mbps = self._pattern.demand_mbps(self._timestep, self._base_demand, self._rng)

        shares = self._decode_shares(action)
        path_choice = self._decode_paths(action)
        desired_mbps = (demand_mbps * shares).astype(np.float32)

        effective_capacity, active_incident_ids = effective_capacities_mbps(
            self._nominal_capacity, self._topology, self._incidents, self._timestep
        )
        incidence = path_incidence(self._flows, self._topology, path_choice)

        result = allocate(
            desired_mbps=desired_mbps,
            incidence=incidence,
            effective_capacity_mbps=effective_capacity,
            queue_mb=self._queue_mb,
        )
        self._queue_mb = result.queue_mb

        per_link_latency = link_latency_ms(result.utilization, result.queue_mb, effective_capacity)
        per_flow_latency = flow_latency_ms(incidence, per_link_latency)
        paths = selected_paths(self._flows, path_choice)

        state = NetworkState(
            timestep=self._timestep,
            links=tuple(
                LinkState(
                    link_id=link.link_id,
                    source=link.source,
                    target=link.target,
                    capacity_mbps=float(self._nominal_capacity[index]),
                    effective_capacity_mbps=float(effective_capacity[index]),
                    allocated_mbps=float(result.link_allocated_mbps[index]),
                    drained_mbps=float(result.link_drained_mbps[index]),
                    queue_mb=float(result.queue_mb[index]),
                    utilization=float(result.utilization[index]),
                    latency_ms=float(per_link_latency[index]),
                )
                for index, link in enumerate(self._topology.links)
            ),
            flows=tuple(
                FlowState(
                    flow_id=flow.flow_id,
                    source=flow.source,
                    target=flow.target,
                    traffic_class=flow.traffic_class,
                    demand_mbps=float(demand_mbps[index]),
                    allocated_mbps=float(result.allocated_mbps[index]),
                    dropped_mbps=float(result.dropped_mbps[index]),
                    path=paths[index],
                    candidate_paths=flow.candidate_paths,
                    latency_ms=float(per_flow_latency[index]),
                )
                for index, flow in enumerate(self._flows)
            ),
            active_incident_ids=active_incident_ids,
        )

        metrics = compute_metrics(
            timestep=self._timestep,
            demand_mbps=demand_mbps,
            allocated_mbps=result.allocated_mbps,
            dropped_mbps=result.dropped_mbps,
            drained_mbps=result.link_drained_mbps,
            flow_latency_ms=per_flow_latency,
            utilization=result.utilization,
            traffic_classes=self._traffic_classes,
        )
        return state, metrics

    def _decode_shares(self, action: AllocationAction | None) -> NDArray[np.float32]:
        """Per-flow share of demand to request. Missing flows default to their full demand.

        Decoding is total by construction: an action that names flows the simulator does not have, or omits
        ones it does, still produces a valid allocation. A policy emitting a partial action is a bug worth
        finding, but not one worth crashing a training run over.
        """
        if action is None:
            return np.ones(len(self._flows), dtype=np.float32)
        return np.array(
            [action.share_by_flow_id.get(flow.flow_id, 1.0) for flow in self._flows],
            dtype=np.float32,
        )

    def _decode_paths(self, action: AllocationAction | None) -> NDArray[np.int_] | None:
        """Per-flow candidate-path index. ``None`` when every flow keeps its default path."""
        if action is None or not action.path_by_flow_id:
            return None
        choices: list[int] = []
        for flow in self._flows:
            requested = action.path_by_flow_id.get(flow.flow_id)
            if requested is None or requested not in flow.candidate_paths:
                choices.append(0)
            else:
                choices.append(flow.candidate_paths.index(requested))
        return np.array(choices, dtype=np.int_)
