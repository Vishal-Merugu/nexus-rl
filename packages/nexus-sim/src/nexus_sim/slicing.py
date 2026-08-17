"""Per-slice SLA attainment — is each promise actually being kept?

The aggregate metrics v1 reported (mean latency, total throughput, fairness) cannot answer the question
slicing exists to pose. A network can hit an excellent mean latency while the premium slice is in
breach, because the best-effort slice is large and fast. Averaging across slices destroys precisely the
distinction the slices were created to make.

So attainment is computed **per slice**, and the aggregate is reported alongside rather than instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nexus_sim.constants import EPSILON
from nexus_types.network_state import FlowState, NetworkState
from nexus_types.slice import DEFAULT_SLICES, NetworkSlice, SliceSet, SliceSla


@dataclass(frozen=True, slots=True)
class SliceAttainment:
    """One slice's observed QoS against its promise, for one timestep."""

    slice_id: str
    sla: SliceSla
    """Carried with the measurement so headroom is computable and so a stored attainment cannot be read
    against the wrong promise."""

    flows: int
    p95_latency_ms: float
    satisfaction: float
    """Fraction of offered demand actually carried."""
    drop_rate: float

    latency_met: bool
    satisfaction_met: bool
    drops_met: bool

    @property
    def is_met(self) -> bool:
        """Whether every term of the SLA holds.

        All three, not a score: an SLA that is 'mostly met' is broken. Partial credit belongs in the
        reward signal, not in the statement of whether a promise was kept.
        """
        return self.latency_met and self.satisfaction_met and self.drops_met

    @property
    def headroom(self) -> float:
        """How far the tightest SLA term is from breaching, in [0, 1]. 0 means at or past the limit.

        The reward uses this rather than the boolean so the policy gets a gradient. A step-function
        "met / not met" signal tells it nothing about whether it is getting warmer, which is how a policy
        ends up sitting just past a limit it could cheaply have cleared.
        """
        if self.flows == 0:
            return 1.0
        return min(
            _fraction_below(self.p95_latency_ms, self.sla.max_latency_ms),
            _fraction_above(self.satisfaction, self.sla.min_satisfaction),
            _fraction_below(self.drop_rate, self.sla.max_drop_rate),
        )


def _fraction_below(value: float, limit: float) -> float:
    """1.0 when the value is at zero, 0.0 when it reaches the ceiling."""
    if limit <= EPSILON:
        return 0.0
    return max(0.0, min(1.0, 1.0 - value / limit))


def _fraction_above(value: float, floor: float) -> float:
    """1.0 when the value comfortably clears the floor, 0.0 when it is at or below it."""
    if floor >= 1.0 - EPSILON:
        return 1.0 if value >= floor else 0.0
    return max(0.0, min(1.0, (value - floor) / (1.0 - floor)))


def flows_in_slice(state: NetworkState, network_slice: NetworkSlice) -> tuple[FlowState, ...]:
    """Membership is derived from the flow's own traffic class.

    Nothing has to be told which flows belong to which slice — the state already carries it. Two services
    agreeing about membership by configuration is the failure `tasks/lessons.md` records under "two
    services must never derive shared state independently".
    """
    return tuple(flow for flow in state.flows if flow.traffic_class is network_slice.traffic_class)


def slice_attainment(state: NetworkState, network_slice: NetworkSlice) -> SliceAttainment:
    """Measure one slice against its SLA for the current timestep."""
    flows = flows_in_slice(state, network_slice)
    if not flows:
        # An empty slice is trivially in compliance. Scoring it as a breach would penalise the policy for
        # traffic that does not exist, and reward it for creating traffic that does.
        return SliceAttainment(
            slice_id=network_slice.slice_id,
            sla=network_slice.sla,
            flows=0,
            p95_latency_ms=0.0,
            satisfaction=1.0,
            drop_rate=0.0,
            latency_met=True,
            satisfaction_met=True,
            drops_met=True,
        )

    latencies = np.array([flow.latency_ms for flow in flows], dtype=np.float64)
    p95 = float(np.percentile(latencies, 95)) if latencies.size > 1 else float(latencies[0])

    demand = sum(flow.demand_mbps for flow in flows)
    allocated = sum(flow.allocated_mbps for flow in flows)
    dropped = sum(flow.dropped_mbps for flow in flows)

    satisfaction = allocated / demand if demand > EPSILON else 1.0
    drop_rate = dropped / demand if demand > EPSILON else 0.0

    return SliceAttainment(
        slice_id=network_slice.slice_id,
        sla=network_slice.sla,
        flows=len(flows),
        p95_latency_ms=p95,
        satisfaction=satisfaction,
        drop_rate=drop_rate,
        latency_met=p95 <= network_slice.sla.max_latency_ms,
        satisfaction_met=satisfaction >= network_slice.sla.min_satisfaction,
        drops_met=drop_rate <= network_slice.sla.max_drop_rate,
    )


def all_attainments(state: NetworkState, slices: SliceSet = DEFAULT_SLICES) -> tuple[SliceAttainment, ...]:
    """Every slice's attainment, in the slice set's stable priority order."""
    return tuple(slice_attainment(state, network_slice) for network_slice in slices.ordered)


def attainment_summary(state: NetworkState, slices: SliceSet = DEFAULT_SLICES) -> dict[str, object]:
    """Per-slice attainment in a shape the dashboard and the agent layer can read.

    Reports each slice separately *and* the count met. It does not report a mean across slices — that
    number would let a satisfied best-effort slice mask a breached premium one, which is the exact
    failure per-slice measurement exists to prevent.
    """
    attainments = all_attainments(state, slices)
    return {
        "timestep": state.timestep,
        "slices": [
            {
                "slice_id": a.slice_id,
                "flows": a.flows,
                "p95_latency_ms": a.p95_latency_ms,
                "satisfaction": a.satisfaction,
                "drop_rate": a.drop_rate,
                "sla_met": a.is_met,
                "breached": [
                    name
                    for name, met in (
                        ("latency", a.latency_met),
                        ("satisfaction", a.satisfaction_met),
                        ("drops", a.drops_met),
                    )
                    if not met
                ],
            }
            for a in attainments
        ],
        "slices_met": sum(1 for a in attainments if a.is_met),
        "slices_total": len(attainments),
    }
