from __future__ import annotations

import numpy as np
import pytest

from nexus_sim.fairness import jains_fairness_index, satisfaction_ratios


def test_equal_satisfaction_scores_one() -> None:
    assert jains_fairness_index(np.array([0.5, 0.5, 0.5], dtype=np.float32)) == pytest.approx(1.0)


def test_unequal_satisfaction_scores_below_one() -> None:
    assert jains_fairness_index(np.array([1.0, 0.1, 0.1], dtype=np.float32)) < 0.7


def test_index_falls_as_allocation_becomes_more_skewed() -> None:
    mild = jains_fairness_index(np.array([0.6, 0.4], dtype=np.float32))
    severe = jains_fairness_index(np.array([1.0, 0.0], dtype=np.float32))

    assert severe < mild


def test_no_flows_scores_one_by_convention() -> None:
    assert jains_fairness_index(np.zeros(0, dtype=np.float32)) == 1.0


def test_allocating_nothing_to_anybody_scores_one() -> None:
    # Documents a real degenerate optimum: perfect fairness by starving everyone. This is exactly why
    # fairness must never be reported without throughput beside it.
    assert jains_fairness_index(np.zeros(4, dtype=np.float32)) == 1.0


def test_satisfaction_is_allocated_over_demand() -> None:
    ratios = satisfaction_ratios(
        np.array([25.0, 50.0], dtype=np.float32), np.array([100.0, 100.0], dtype=np.float32)
    )

    np.testing.assert_allclose(ratios, [0.25, 0.5], rtol=1e-5)


def test_idle_flows_count_as_satisfied_rather_than_dragging_the_index_down() -> None:
    ratios = satisfaction_ratios(
        np.array([0.0, 50.0], dtype=np.float32), np.array([0.0, 50.0], dtype=np.float32)
    )

    np.testing.assert_allclose(ratios, [1.0, 1.0], rtol=1e-5)


def test_over_allocation_does_not_push_satisfaction_above_one() -> None:
    ratios = satisfaction_ratios(np.array([200.0], dtype=np.float32), np.array([100.0], dtype=np.float32))

    assert ratios[0] == pytest.approx(1.0)
