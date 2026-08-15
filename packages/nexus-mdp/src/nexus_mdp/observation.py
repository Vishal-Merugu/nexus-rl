"""Observation encoding.

Design constraints, all of them load-bearing:

- **Fixed shape.** Length depends only on the topology and flow count, never on runtime state.
- **Stable ordering.** Element 7 means the same thing in every episode and every run, because links and
  flows come out in the simulator's array order. An observation whose ordering drifts silently invalidates
  every checkpoint.
- **Normalised to [0, 1].** Raw Mb/s next to utilization ratios means the network effectively cannot see the
  ratios. Every divisor here is part of the MDP contract — changing one is a version bump.
- **Markov-sufficient for the control problem.** Utilization, backlog, per-flow satisfaction and demand
  together determine the right allocation. Where history matters (a link trending toward saturation), the
  queue term carries it: backlog is an integral of past over-subscription.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nexus_sim.constants import EPSILON, MAX_LINK_LATENCY_MS, MAX_QUEUE_MB
from nexus_types.network_state import NetworkState
from nexus_types.traffic_class import TrafficClass

LINK_FEATURES: int = 3
"""Per link: utilization, normalised backlog, capacity health (effective ÷ nominal)."""

FLOW_FEATURES: int = 4 + len(TrafficClass)
"""Per flow: satisfaction, normalised demand, normalised latency, drop ratio, and a class one-hot."""

GLOBAL_FEATURES: int = 3
"""Mean utilization, network-wide satisfaction, network-wide drop ratio."""

DEMAND_NORMALISER_MBPS: float = 400.0
"""Divisor for per-flow demand. Comfortably above the highest demand a flash crowd produces, so the term
stays inside [0, 1] rather than saturating at the one moment the policy most needs to distinguish."""


def observation_size(link_count: int, flow_count: int) -> int:
    """The fixed observation length for a topology and flow set."""
    return link_count * LINK_FEATURES + flow_count * FLOW_FEATURES + GLOBAL_FEATURES


def encode_observation(state: NetworkState) -> NDArray[np.float32]:
    """Encode a network state as a normalised float32 vector.

    ``float32`` deliberately: Stable-Baselines3 assumes it, and a silent float64 observation costs memory
    and throws dtype errors deep inside its replay buffers.
    """
    link_block = np.empty(len(state.links) * LINK_FEATURES, dtype=np.float32)
    for index, link in enumerate(state.links):
        offset = index * LINK_FEATURES
        link_block[offset] = link.utilization
        link_block[offset + 1] = min(1.0, link.queue_mb / MAX_QUEUE_MB)
        link_block[offset + 2] = (
            link.effective_capacity_mbps / link.capacity_mbps if link.capacity_mbps > EPSILON else 0.0
        )

    classes = tuple(TrafficClass)
    flow_block = np.zeros(len(state.flows) * FLOW_FEATURES, dtype=np.float32)
    for index, flow in enumerate(state.flows):
        offset = index * FLOW_FEATURES
        flow_block[offset] = flow.satisfaction_ratio
        flow_block[offset + 1] = min(1.0, flow.demand_mbps / DEMAND_NORMALISER_MBPS)
        flow_block[offset + 2] = min(1.0, flow.latency_ms / MAX_LINK_LATENCY_MS)
        flow_block[offset + 3] = (
            min(1.0, flow.dropped_mbps / flow.demand_mbps) if flow.demand_mbps > EPSILON else 0.0
        )
        flow_block[offset + 4 + classes.index(flow.traffic_class)] = 1.0

    demand = state.total_demand_mbps
    global_block = np.array(
        [
            float(np.mean([link.utilization for link in state.links])) if state.links else 0.0,
            state.total_allocated_mbps / demand if demand > EPSILON else 1.0,
            state.total_dropped_mbps / demand if demand > EPSILON else 0.0,
        ],
        dtype=np.float32,
    )

    encoded: NDArray[np.float32] = np.clip(
        np.concatenate([link_block, flow_block, global_block]), 0.0, 1.0
    ).astype(np.float32)
    return encoded
