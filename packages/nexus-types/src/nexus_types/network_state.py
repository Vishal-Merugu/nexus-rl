"""The network state contract.

State is **frozen**. ``step()`` returns a new ``NetworkState`` rather than mutating the previous one, so that
snapshots already handed to the dashboard or the agent layer stay valid. Mutating a handed-out state produces
"the explanation doesn't match what I saw" bugs that are near-impossible to trace.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat

from nexus_types.traffic_class import TrafficClass


class LinkState(BaseModel):
    """One directed link at one timestep."""

    model_config = ConfigDict(frozen=True)

    link_id: str = Field(description="Stable identifier, formatted 'src->dst'.")
    source: str
    target: str
    capacity_mbps: NonNegativeFloat = Field(description="Fixed nominal capacity.")
    effective_capacity_mbps: NonNegativeFloat = Field(
        description="Capacity after any active incident. Equals capacity_mbps when healthy."
    )
    allocated_mbps: NonNegativeFloat = Field(
        description="Total rate the link carried: newly admitted traffic plus backlog drained."
    )
    drained_mbps: NonNegativeFloat = Field(
        default=0.0,
        description="The portion of allocated_mbps that came from clearing backlog rather than from new "
        "demand. Reported separately because it is *not* attributed to any flow — see StepMetrics.",
    )
    queue_mb: NonNegativeFloat = Field(description="Scalar backlog awaiting transmission.")
    utilization: float = Field(ge=0.0, le=1.0, description="allocated / effective capacity. Always clamped.")
    latency_ms: NonNegativeFloat = Field(description="Derived: propagation + queueing delay.")


class FlowState(BaseModel):
    """One traffic demand at one timestep."""

    model_config = ConfigDict(frozen=True)

    flow_id: str
    source: str
    target: str
    traffic_class: TrafficClass
    demand_mbps: NonNegativeFloat = Field(description="Rate requested this timestep.")
    allocated_mbps: NonNegativeFloat = Field(description="Rate actually granted.")
    dropped_mbps: NonNegativeFloat = Field(description="Demand shed because queues were full.")
    path: tuple[str, ...] = Field(description="Ordered link_ids the flow traverses.")
    candidate_paths: tuple[tuple[str, ...], ...] = Field(
        default=(),
        description="Every path this flow may take, ordered — the action space indexes into it.\n\n"
        "Carried in the state so the controller can decode actions from what it *received*, rather than "
        "rebuilding the flow set from its own configuration. Two services deriving shared state "
        "independently diverge silently: flow ids are positional, so they always line up even when the "
        "routes behind them do not.",
    )
    latency_ms: NonNegativeFloat = Field(description="Sum of per-link latency along the path.")

    @property
    def satisfaction_ratio(self) -> float:
        """Allocated ÷ requested, in [0, 1]. A flow with no demand is fully satisfied by definition."""
        if self.demand_mbps <= 0.0:
            return 1.0
        return min(1.0, self.allocated_mbps / self.demand_mbps)


class NetworkState(BaseModel):
    """A complete snapshot of the simulated network at one timestep.

    This is the contract between the simulator and everything that reads it: the controller, the agent layer's
    read tools, and the dashboard. All three must be able to join on ``timestep`` — showing two timesteps at
    once is worse than lagging.
    """

    model_config = ConfigDict(frozen=True)

    timestep: int = Field(ge=0)
    links: tuple[LinkState, ...]
    flows: tuple[FlowState, ...]
    active_incident_ids: tuple[str, ...] = Field(
        default=(), description="Incidents degrading the network at this timestep."
    )

    @property
    def total_demand_mbps(self) -> float:
        return sum(flow.demand_mbps for flow in self.flows)

    @property
    def total_allocated_mbps(self) -> float:
        return sum(flow.allocated_mbps for flow in self.flows)

    @property
    def total_dropped_mbps(self) -> float:
        return sum(flow.dropped_mbps for flow in self.flows)

    def link_by_id(self, link_id: str) -> LinkState:
        """Look up a link. Raises ``KeyError`` if absent — callers hold ids that came from this state."""
        for link in self.links:
            if link.link_id == link_id:
                return link
        raise KeyError(link_id)
