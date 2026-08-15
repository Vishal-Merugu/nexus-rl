"""Fairness — Jain's index over per-flow satisfaction ratios."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nexus_sim.constants import EPSILON


def satisfaction_ratios(
    allocated_mbps: NDArray[np.float32],
    demand_mbps: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Allocated ÷ requested per flow, in [0, 1].

    A flow with no demand is fully satisfied by definition — otherwise an idle flow would drag the fairness
    index down and the policy would learn to manufacture traffic for it.
    """
    ratios = np.where(
        demand_mbps > EPSILON,
        allocated_mbps / np.maximum(demand_mbps, EPSILON),
        1.0,
    )
    return np.clip(ratios, 0.0, 1.0).astype(np.float32)


def jains_fairness_index(ratios: NDArray[np.float32]) -> float:
    """Jain's fairness index over satisfaction ratios, in (0, 1].

    1.0 means every flow received the same *fraction* of what it asked for. This is a proportional notion:
    an equal-bytes split across unequal demands scores poorly, and that is intended — a large flow and a
    small one both getting 40 Mb/s is not fair to the large one.

    Returns 1.0 for an empty flow set, and for the all-zero case: no allocation at all is trivially equal.
    That is also the reason fairness must never be reported without throughput beside it — allocating
    nothing to anybody scores a perfect 1.0.
    """
    if ratios.size == 0:
        return 1.0
    total = float(np.sum(ratios))
    if total <= EPSILON:
        return 1.0
    sum_of_squares = float(np.sum(np.square(ratios)))
    return total**2 / (ratios.size * sum_of_squares)
