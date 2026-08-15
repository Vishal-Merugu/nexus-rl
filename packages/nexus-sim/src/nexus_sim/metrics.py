"""Per-timestep QoS metrics.

These, not reward, are what results are reported in. Reward is the objective the policy was designed to
maximise — that it maximises it is not news. Operators care about latency, throughput, fairness and drops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from nexus_sim.fairness import jains_fairness_index, satisfaction_ratios
from nexus_sim.latency import percentile_latency_ms
from nexus_types.traffic_class import TrafficClass


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """QoS metrics for one timestep."""

    timestep: int
    p95_latency_ms: float
    mean_latency_ms: float
    total_demand_mbps: float
    total_throughput_mbps: float
    """Newly admitted traffic, summed over flows. Excludes backlog drain — see ``total_drained_mbps``."""
    total_dropped_mbps: float
    total_drained_mbps: float
    """Backlog cleared this timestep, summed over links.

    Unattributed to flows by construction (the queue is a scalar, so ownership is not tracked), which is
    why per-flow throughput and per-link carried rate differ. ``total_delivered_mbps`` reconciles them.
    """
    fairness_index: float
    mean_utilization: float
    max_utilization: float
    throughput_by_class_mbps: dict[TrafficClass, float] = field(default_factory=dict)

    @property
    def total_delivered_mbps(self) -> float:
        """Everything the network actually moved: new traffic plus backlog drained.

        Reconciles the per-flow and per-link views, which differ by exactly the drained amount.
        """
        return self.total_throughput_mbps + self.total_drained_mbps

    def as_dict(self) -> dict[str, float]:
        """Flatten for a metrics.jsonl record."""
        flat: dict[str, float] = {
            "p95_latency_ms": self.p95_latency_ms,
            "mean_latency_ms": self.mean_latency_ms,
            "total_demand_mbps": self.total_demand_mbps,
            "total_throughput_mbps": self.total_throughput_mbps,
            "total_dropped_mbps": self.total_dropped_mbps,
            "total_drained_mbps": self.total_drained_mbps,
            "total_delivered_mbps": self.total_delivered_mbps,
            "fairness_index": self.fairness_index,
            "mean_utilization": self.mean_utilization,
            "max_utilization": self.max_utilization,
        }
        for traffic_class, throughput in self.throughput_by_class_mbps.items():
            flat[f"throughput_{traffic_class.value.lower()}_mbps"] = throughput
        return flat


def compute_metrics(
    timestep: int,
    demand_mbps: NDArray[np.float32],
    allocated_mbps: NDArray[np.float32],
    dropped_mbps: NDArray[np.float32],
    drained_mbps: NDArray[np.float32],
    flow_latency_ms: NDArray[np.float32],
    utilization: NDArray[np.float32],
    traffic_classes: tuple[TrafficClass, ...],
) -> StepMetrics:
    """Compute QoS metrics for one timestep.

    Per-class throughput is reported alongside the aggregate deliberately: an aggregate win that starves
    ``LATENCY_SENSITIVE`` traffic is a loss, and only the per-class breakdown reveals it.
    """
    ratios = satisfaction_ratios(allocated_mbps, demand_mbps)

    by_class: dict[TrafficClass, float] = {}
    for traffic_class in TrafficClass:
        mask = np.array([flow_class is traffic_class for flow_class in traffic_classes], dtype=bool)
        by_class[traffic_class] = float(np.sum(allocated_mbps[mask])) if mask.any() else 0.0

    return StepMetrics(
        timestep=timestep,
        p95_latency_ms=percentile_latency_ms(flow_latency_ms, 95.0),
        mean_latency_ms=float(np.mean(flow_latency_ms)) if flow_latency_ms.size else 0.0,
        total_demand_mbps=float(np.sum(demand_mbps)),
        total_throughput_mbps=float(np.sum(allocated_mbps)),
        total_dropped_mbps=float(np.sum(dropped_mbps)),
        total_drained_mbps=float(np.sum(drained_mbps)),
        fairness_index=jains_fairness_index(ratios),
        mean_utilization=float(np.mean(utilization)) if utilization.size else 0.0,
        max_utilization=float(np.max(utilization)) if utilization.size else 0.0,
        throughput_by_class_mbps=by_class,
    )
