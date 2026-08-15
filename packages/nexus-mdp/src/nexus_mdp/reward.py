"""Reward definition.

Four named components, each logged individually. Never a bare scalar: a flat total with a rising throughput
term and a falling fairness term tells you exactly what happened, where the scalar tells you nothing. This
is the single most useful debugging property the MDP has.

Magnitudes are deliberately comparable — every component is normalised into roughly [0, 1] before weighting,
so no term can dominate by orders of magnitude and render the others invisible to the policy.

**The degenerate optima this design closes:**

- *Perfect fairness by allocating nothing.* Jain's index scores 1.0 when every flow gets zero. Closed by
  weighting fairness by network-wide satisfaction — fairness only pays when traffic is actually flowing.
- *Cutting latency by dropping traffic.* Latency on surviving flows improves when the congested ones are
  shed. Closed by the drop penalty, which is weighted heavily enough to exceed any latency gain drops buy.
- *Ignoring latency-sensitive traffic.* Aggregate throughput can rise while interactive flows starve. Closed
  by per-class latency weighting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nexus_sim.constants import EPSILON, MAX_LINK_LATENCY_MS
from nexus_types.network_state import NetworkState
from nexus_types.traffic_class import TrafficClass

_LATENCY_CLASS_WEIGHT: dict[TrafficClass, float] = {
    TrafficClass.LATENCY_SENSITIVE: 3.0,
    TrafficClass.BULK: 1.0,
    TrafficClass.BACKGROUND: 0.25,
}
"""How much each class's delay counts. Latency-sensitive delay hurts three times as much as bulk delay —
without this, the policy is indifferent to which traffic it delays, which is the opposite of QoS."""

_MAX_LATENCY_CLASS_WEIGHT: float = max(_LATENCY_CLASS_WEIGHT.values())
"""Divisor that keeps the latency term in [0, 1].

Normalising by the *sum* of the participating weights instead would cancel the class weighting entirely —
the weighted mean of a single class is that class's value regardless of its weight. Dividing by the maximum
preserves the relative ordering between classes while still bounding the term.
"""


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Component weights. Part of the MDP — changing one is a version bump.

    Distinct from ``ObjectiveWeights`` in ``nexus-types``, which an operator constraint can adjust at
    runtime. These are fixed at training time and define what the policy learned to want.
    """

    throughput: float = 1.0
    latency: float = 1.0
    drops: float = 2.0
    """Weighted above the others on purpose: dropping traffic to make the remaining flows look fast must
    never be profitable."""
    fairness: float = 0.5


def compute_reward(
    state: NetworkState,
    weights: RewardWeights = RewardWeights(),
) -> dict[str, float]:
    """Compute the named reward components for one timestep.

    Returns the components, not the total. Callers sum them — which keeps it structurally impossible to log
    a total without the parts that produced it.
    """
    demand = state.total_demand_mbps
    if demand <= EPSILON:
        # An idle network is neither good nor bad. Scoring it positively would teach the policy to
        # suppress demand.
        return {
            "throughput_reward": 0.0,
            "latency_penalty": 0.0,
            "drop_penalty": 0.0,
            "fairness_bonus": 0.0,
        }

    satisfaction = state.total_allocated_mbps / demand
    drop_ratio = min(1.0, state.total_dropped_mbps / demand)

    weighted_latency = sum(
        _LATENCY_CLASS_WEIGHT[flow.traffic_class] * min(1.0, flow.latency_ms / MAX_LINK_LATENCY_MS)
        for flow in state.flows
    )
    normalised_latency = (
        weighted_latency / (len(state.flows) * _MAX_LATENCY_CLASS_WEIGHT) if state.flows else 0.0
    )

    fairness = _jains_index(np.array([flow.satisfaction_ratio for flow in state.flows], dtype=np.float32))

    return {
        "throughput_reward": weights.throughput * satisfaction,
        "latency_penalty": -weights.latency * normalised_latency,
        "drop_penalty": -weights.drops * drop_ratio,
        # Scaled by satisfaction so that starving everyone equally — which scores a perfect 1.0 on Jain's
        # index — earns nothing.
        "fairness_bonus": weights.fairness * fairness * satisfaction,
    }


def total_reward(components: dict[str, float]) -> float:
    """Sum the components. The only way a total is ever produced."""
    return float(sum(components.values()))


def _jains_index(ratios: NDArray[np.float32]) -> float:
    if ratios.size == 0:
        return 1.0
    total = float(np.sum(ratios))
    if total <= EPSILON:
        return 0.0  # nothing allocated: no fairness credit, unlike the raw index
    return total**2 / (ratios.size * float(np.sum(np.square(ratios))))
