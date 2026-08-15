"""Latency model.

Flow-level fidelity: latency is *derived* from utilization and queue backlog, not measured from packet
traces. Two terms per link:

1. **Queueing delay under load** — an M/M/1-flavoured ``1/(1-u)`` factor on the propagation delay. Cheap,
   and it reproduces the property that matters for control: delay rises sharply as a link saturates, so the
   policy has a reason to avoid the last few percent of capacity.
2. **Backlog delay** — ``queue_mb / capacity_mbps``, the time to drain what is already waiting.

Both are clamped. The ``1/(1-u)`` factor diverges at full utilization, and an infinity here propagates into
the reward as a NaN that silently destroys a training run.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nexus_sim.constants import (
    BASE_PROPAGATION_MS,
    EPSILON,
    MAX_LINK_LATENCY_MS,
    MAX_UTILIZATION_FOR_DELAY,
)


def link_latency_ms(
    utilization: NDArray[np.float32],
    queue_mb: NDArray[np.float32],
    effective_capacity_mbps: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Per-link latency in milliseconds.

    A link with zero effective capacity (a failed link) is assigned the maximum latency rather than an
    infinity, so that routing over it is heavily penalised but the arithmetic stays finite.
    """
    clamped_utilization = np.clip(utilization, 0.0, MAX_UTILIZATION_FOR_DELAY)
    congestion_factor = 1.0 / (1.0 - clamped_utilization)

    drain_ms = 1000.0 * queue_mb / np.maximum(effective_capacity_mbps, EPSILON)
    latency = BASE_PROPAGATION_MS * congestion_factor + drain_ms

    failed = effective_capacity_mbps <= EPSILON
    latency = np.where(failed, MAX_LINK_LATENCY_MS, latency)
    return np.clip(latency, 0.0, MAX_LINK_LATENCY_MS).astype(np.float32)


def flow_latency_ms(
    incidence: NDArray[np.float32],
    link_latency: NDArray[np.float32],
) -> NDArray[np.float32]:
    """End-to-end latency per flow: the sum along its selected path.

    ``incidence`` is the flow x link matrix, so this is one matrix-vector product rather than a loop over
    flows — which is what keeps the hot path inside its budget.
    """
    return (incidence @ link_latency).astype(np.float32)


def percentile_latency_ms(flow_latency: NDArray[np.float32], percentile: float = 95.0) -> float:
    """Tail latency across flows.

    Report p95, not the mean. QoS is a tail property, and a mean hides exactly the failures the reward is
    meant to punish.
    """
    if flow_latency.size == 0:
        return 0.0
    return float(np.percentile(flow_latency, percentile))
