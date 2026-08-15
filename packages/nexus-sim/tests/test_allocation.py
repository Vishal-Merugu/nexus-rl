from __future__ import annotations

import numpy as np
import pytest

from nexus_errors.simulation_error import InvariantViolationError
from nexus_sim.allocation import allocate
from nexus_sim.constants import MAX_QUEUE_MB


def _single_link(
    desired: float, capacity: float, queue: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([desired], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        np.array([capacity], dtype=np.float32),
        np.array([queue], dtype=np.float32),
    )


def test_demand_below_capacity_is_fully_allocated() -> None:
    desired, incidence, capacity, queue = _single_link(desired=40.0, capacity=100.0)

    result = allocate(desired, incidence, capacity, queue)

    assert result.allocated_mbps[0] == pytest.approx(40.0)
    assert result.dropped_mbps[0] == pytest.approx(0.0)
    assert result.queue_mb[0] == pytest.approx(0.0)


def test_demand_above_capacity_is_capped_and_the_excess_queues() -> None:
    desired, incidence, capacity, queue = _single_link(desired=200.0, capacity=100.0)

    result = allocate(desired, incidence, capacity, queue)

    assert result.allocated_mbps[0] == pytest.approx(100.0)
    assert result.queue_mb[0] > 0.0


def test_a_link_never_carries_more_than_its_capacity() -> None:
    desired, incidence, capacity, queue = _single_link(desired=5_000.0, capacity=100.0)

    result = allocate(desired, incidence, capacity, queue)

    assert result.link_allocated_mbps[0] <= 100.0 + 1e-3


def test_utilization_stays_within_bounds_under_extreme_oversubscription() -> None:
    desired, incidence, capacity, queue = _single_link(desired=1e6, capacity=1.0)

    result = allocate(desired, incidence, capacity, queue)

    assert 0.0 <= result.utilization[0] <= 1.0


def test_zero_capacity_link_allocates_nothing_and_does_not_divide_by_zero() -> None:
    desired, incidence, capacity, queue = _single_link(desired=50.0, capacity=0.0)

    result = allocate(desired, incidence, capacity, queue)

    assert result.allocated_mbps[0] == pytest.approx(0.0)
    assert result.utilization[0] == pytest.approx(0.0)
    assert np.isfinite(result.queue_mb[0])


def test_zero_demand_leaves_everything_at_rest() -> None:
    desired, incidence, capacity, queue = _single_link(desired=0.0, capacity=100.0)

    result = allocate(desired, incidence, capacity, queue)

    assert result.allocated_mbps[0] == pytest.approx(0.0)
    assert result.queue_mb[0] == pytest.approx(0.0)
    assert result.utilization[0] == pytest.approx(0.0)


def test_queue_is_capped_and_the_overflow_becomes_drops() -> None:
    desired, incidence, capacity, queue = _single_link(desired=10_000.0, capacity=10.0, queue=MAX_QUEUE_MB)

    result = allocate(desired, incidence, capacity, queue)

    assert result.queue_mb[0] <= MAX_QUEUE_MB + 1e-3
    assert result.dropped_mbps[0] > 0.0


def test_a_flow_is_never_charged_more_drops_than_it_offered() -> None:
    desired, incidence, capacity, queue = _single_link(desired=30.0, capacity=1.0, queue=MAX_QUEUE_MB)

    result = allocate(desired, incidence, capacity, queue)

    assert result.dropped_mbps[0] <= 30.0 + 1e-3


def test_existing_backlog_drains_when_new_demand_is_light() -> None:
    desired, incidence, capacity, queue = _single_link(desired=1.0, capacity=100.0, queue=10.0)

    result = allocate(desired, incidence, capacity, queue)

    assert result.queue_mb[0] < 10.0


def test_a_flow_is_limited_by_the_tightest_link_on_its_path() -> None:
    # One flow crossing two links: a wide one and a narrow one.
    desired = np.array([100.0], dtype=np.float32)
    incidence = np.array([[1.0, 1.0]], dtype=np.float32)
    capacity = np.array([100.0, 20.0], dtype=np.float32)
    queue = np.zeros(2, dtype=np.float32)

    result = allocate(desired, incidence, capacity, queue)

    assert result.allocated_mbps[0] == pytest.approx(20.0, rel=1e-3)


def test_contending_flows_split_a_link_in_proportion_to_demand() -> None:
    desired = np.array([75.0, 25.0], dtype=np.float32)
    incidence = np.array([[1.0], [1.0]], dtype=np.float32)
    capacity = np.array([100.0], dtype=np.float32)
    queue = np.zeros(1, dtype=np.float32)

    result = allocate(desired, incidence, capacity, queue)

    assert result.allocated_mbps[0] == pytest.approx(75.0, rel=1e-3)
    assert result.allocated_mbps[1] == pytest.approx(25.0, rel=1e-3)


def test_flows_on_disjoint_links_do_not_interfere() -> None:
    desired = np.array([50.0, 50.0], dtype=np.float32)
    incidence = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    capacity = np.array([100.0, 10.0], dtype=np.float32)
    queue = np.zeros(2, dtype=np.float32)

    result = allocate(desired, incidence, capacity, queue)

    assert result.allocated_mbps[0] == pytest.approx(50.0)
    assert result.allocated_mbps[1] == pytest.approx(10.0, rel=1e-3)


def test_negative_incoming_queue_is_rejected_as_an_invariant_violation() -> None:
    desired, incidence, capacity, _ = _single_link(desired=10.0, capacity=100.0)

    with pytest.raises(InvariantViolationError):
        allocate(desired, incidence, capacity, np.array([-5.0], dtype=np.float32))


def test_nan_demand_is_rejected_before_it_can_corrupt_the_reward() -> None:
    desired, incidence, capacity, queue = _single_link(desired=float("nan"), capacity=100.0)

    with pytest.raises(InvariantViolationError, match="NaN"):
        allocate(desired, incidence, capacity, queue)


def test_no_flows_is_handled_without_error() -> None:
    result = allocate(
        np.zeros(0, dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
        np.array([100.0, 100.0], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )

    assert result.allocated_mbps.size == 0
    assert result.utilization.shape == (2,)
