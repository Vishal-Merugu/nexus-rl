"""The allocation step — resolving desired rates against link capacity.

This is the heart of the flow-level model. Given what each flow wants and which links it traverses, it works
out what each flow actually gets, how much backlog builds, and what is dropped.

Every function here is pure: same inputs, same outputs, no state, no RNG. Invariants are asserted rather
than assumed — RL will find and exploit any invariant left unenforced, and the resulting policy looks
successful while doing something physically impossible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nexus_errors.simulation_error import InvariantViolationError
from nexus_sim.constants import EPSILON, MAX_QUEUE_MB, QUEUE_DRAIN_PRIORITY, TIMESTEP_SECONDS


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """What one timestep of allocation produced. All arrays are in flow or link order."""

    allocated_mbps: NDArray[np.float32]
    """Per-flow end-to-end rate granted."""

    dropped_mbps: NDArray[np.float32]
    """Per-flow rate shed because a queue on its path overflowed."""

    link_allocated_mbps: NDArray[np.float32]
    """Per-link rate carried, including backlog drain."""

    link_drained_mbps: NDArray[np.float32]
    """The backlog-clearing portion of ``link_allocated_mbps``.

    Kept separate because it is **not** credited to any flow: this is a flow-level model with a scalar,
    unattributed queue, so once bytes enter a backlog the simulator no longer knows whose they were.
    Per-flow throughput therefore counts only newly admitted traffic, and the two views differ by exactly
    this quantity. Surfaced in metrics so the gap is measurable rather than mysterious; attributing it
    properly would need per-flow queues, which is a fidelity increase on the scale of ADR-0001.
    """

    queue_mb: NDArray[np.float32]
    """Per-link backlog after this timestep."""

    utilization: NDArray[np.float32]
    """Per-link carried rate ÷ effective capacity, clamped to [0, 1]."""


def allocate(
    desired_mbps: NDArray[np.float32],
    incidence: NDArray[np.float32],
    effective_capacity_mbps: NDArray[np.float32],
    queue_mb: NDArray[np.float32],
) -> AllocationResult:
    """Resolve desired flow rates against link capacity for one timestep.

    The model, in order:

    1. Each link reserves a slice of capacity to drain its existing backlog, so a congested link does not
       starve its own queue while new arrivals consume everything.
    2. Remaining capacity is divided proportionally among the flows contending for the link.
    3. A flow is limited by its **bottleneck** — the tightest scaling factor along its path. Sending more
       into a path than its narrowest link can carry would just move the backlog.
    4. Unserved demand becomes queue. Queue above ``MAX_QUEUE_MB`` is dropped, attributed back to flows in
       proportion to what each offered.

    Args:
        desired_mbps: Per-flow rate requested this timestep (demand x the controller's share).
        incidence: Flow x link matrix; 1 where a flow's selected path uses a link.
        effective_capacity_mbps: Per-link capacity after any active incident.
        queue_mb: Per-link backlog carried in from the previous timestep.
    """
    _assert_finite(desired_mbps, "desired_mbps")
    _assert_non_negative(queue_mb, "queue_mb")

    # 1. Capacity reserved for draining existing backlog.
    backlog_rate_mbps = queue_mb / TIMESTEP_SECONDS
    drain_rate_mbps = np.minimum(backlog_rate_mbps, effective_capacity_mbps * QUEUE_DRAIN_PRIORITY)
    capacity_for_new_traffic = np.maximum(effective_capacity_mbps - drain_rate_mbps, 0.0)

    # 2. Proportional share of what remains, per link.
    link_demand_mbps = incidence.T @ desired_mbps
    link_scale = np.where(
        link_demand_mbps > EPSILON,
        np.minimum(1.0, capacity_for_new_traffic / np.maximum(link_demand_mbps, EPSILON)),
        1.0,
    ).astype(np.float32)

    # 3. Each flow is limited by the tightest link on its path. Links it does not use contribute 1.0.
    flow_scale = np.where(incidence > 0.0, link_scale[np.newaxis, :], 1.0).min(axis=1).astype(np.float32)
    allocated_mbps = (desired_mbps * flow_scale).astype(np.float32)

    # 4. What the links could not carry becomes backlog; backlog past the limit is dropped.
    link_served_mbps = incidence.T @ allocated_mbps
    unserved_mbps = np.maximum(link_demand_mbps - link_served_mbps, 0.0)

    drained_mb = drain_rate_mbps * TIMESTEP_SECONDS
    queue_before_limit = queue_mb - drained_mb + unserved_mbps * TIMESTEP_SECONDS
    queue_before_limit = np.maximum(queue_before_limit, 0.0)
    overflow_mb = np.maximum(queue_before_limit - MAX_QUEUE_MB, 0.0)
    new_queue_mb = np.minimum(queue_before_limit, MAX_QUEUE_MB).astype(np.float32)

    dropped_mbps = _attribute_drops(
        overflow_mb=overflow_mb,
        incidence=incidence,
        desired_mbps=desired_mbps,
        link_demand_mbps=link_demand_mbps,
    )

    link_allocated_mbps = (link_served_mbps + drain_rate_mbps).astype(np.float32)
    utilization = np.where(
        effective_capacity_mbps > EPSILON,
        link_allocated_mbps / np.maximum(effective_capacity_mbps, EPSILON),
        0.0,
    )
    utilization = np.clip(utilization, 0.0, 1.0).astype(np.float32)

    result = AllocationResult(
        allocated_mbps=allocated_mbps,
        dropped_mbps=dropped_mbps,
        link_allocated_mbps=link_allocated_mbps,
        link_drained_mbps=drain_rate_mbps.astype(np.float32),
        queue_mb=new_queue_mb,
        utilization=utilization,
    )
    _assert_invariants(result, desired_mbps, effective_capacity_mbps)
    return result


def _attribute_drops(
    overflow_mb: NDArray[np.float32],
    incidence: NDArray[np.float32],
    desired_mbps: NDArray[np.float32],
    link_demand_mbps: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Attribute per-link overflow back to the flows that caused it.

    A flow's share of a link's drops is its share of the demand offered to that link. Attribution is capped
    at what the flow actually offered — a flow crossing several overflowing links must not be charged more
    than it sent.
    """
    if not np.any(overflow_mb > 0.0):
        return np.zeros_like(desired_mbps, dtype=np.float32)

    contribution = incidence * desired_mbps[:, np.newaxis]
    share = np.where(
        link_demand_mbps[np.newaxis, :] > EPSILON,
        contribution / np.maximum(link_demand_mbps[np.newaxis, :], EPSILON),
        0.0,
    )
    dropped_mb = share @ overflow_mb
    dropped_mbps = dropped_mb / TIMESTEP_SECONDS
    capped: NDArray[np.float32] = np.minimum(dropped_mbps, desired_mbps).astype(np.float32)
    return capped


def _assert_finite(values: NDArray[np.float32], name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise InvariantViolationError(
            f"{name} contains NaN or inf. This corrupts the reward and destroys the run silently.",
            {"array": name},
        )


def _assert_non_negative(values: NDArray[np.float32], name: str) -> None:
    if np.any(values < -EPSILON):
        raise InvariantViolationError(f"{name} went negative.", {"array": name, "min": float(np.min(values))})


def _assert_invariants(
    result: AllocationResult,
    desired_mbps: NDArray[np.float32],
    effective_capacity_mbps: NDArray[np.float32],
) -> None:
    """Enforce the physical invariants of the network.

    A negative queue propagates into the reward as a bonus, and the policy learns to cause it. Utilization
    above 1 means the simulator carried more than the link could. Both are always simulation bugs, never
    states a caller should handle.
    """
    tolerance = 1e-3

    if np.any(result.queue_mb < -tolerance):
        raise InvariantViolationError(
            "Queue went negative.", {"min_queue_mb": float(np.min(result.queue_mb))}
        )
    if np.any(result.utilization < -tolerance) or np.any(result.utilization > 1.0 + tolerance):
        raise InvariantViolationError(
            "Utilization left [0, 1].",
            {"min": float(np.min(result.utilization)), "max": float(np.max(result.utilization))},
        )
    if np.any(result.allocated_mbps > desired_mbps + tolerance):
        raise InvariantViolationError(
            "A flow was allocated more than it requested.",
            {"max_excess": float(np.max(result.allocated_mbps - desired_mbps))},
        )
    if np.any(result.link_allocated_mbps > effective_capacity_mbps + tolerance):
        raise InvariantViolationError(
            "A link carried more than its effective capacity.",
            {"max_excess": float(np.max(result.link_allocated_mbps - effective_capacity_mbps))},
        )
    _assert_finite(result.allocated_mbps, "allocated_mbps")
    _assert_finite(result.queue_mb, "queue_mb")
