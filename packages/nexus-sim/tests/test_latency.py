from __future__ import annotations

import numpy as np
import pytest

from nexus_sim.constants import BASE_PROPAGATION_MS, MAX_LINK_LATENCY_MS
from nexus_sim.latency import flow_latency_ms, link_latency_ms, percentile_latency_ms

_CAPACITY = np.array([100.0], dtype=np.float32)
_NO_QUEUE = np.array([0.0], dtype=np.float32)


def test_idle_link_sits_at_propagation_delay() -> None:
    latency = link_latency_ms(np.array([0.0], dtype=np.float32), _NO_QUEUE, _CAPACITY)

    assert latency[0] == pytest.approx(BASE_PROPAGATION_MS)


def test_latency_rises_with_utilization() -> None:
    light = link_latency_ms(np.array([0.2], dtype=np.float32), _NO_QUEUE, _CAPACITY)
    heavy = link_latency_ms(np.array([0.9], dtype=np.float32), _NO_QUEUE, _CAPACITY)

    assert heavy[0] > light[0]


def test_latency_at_full_utilization_stays_finite() -> None:
    # The 1/(1-u) term diverges at u=1; an inf here becomes a NaN reward and destroys a run silently.
    latency = link_latency_ms(np.array([1.0], dtype=np.float32), _NO_QUEUE, _CAPACITY)

    assert np.isfinite(latency[0])
    assert latency[0] <= MAX_LINK_LATENCY_MS


def test_backlog_adds_drain_time() -> None:
    empty = link_latency_ms(np.array([0.5], dtype=np.float32), _NO_QUEUE, _CAPACITY)
    backed_up = link_latency_ms(
        np.array([0.5], dtype=np.float32), np.array([10.0], dtype=np.float32), _CAPACITY
    )

    assert backed_up[0] > empty[0]


def test_failed_link_reports_maximum_latency_not_infinity() -> None:
    latency = link_latency_ms(np.array([0.0], dtype=np.float32), _NO_QUEUE, np.array([0.0], dtype=np.float32))

    assert latency[0] == pytest.approx(MAX_LINK_LATENCY_MS)


def test_latency_is_always_bounded() -> None:
    latency = link_latency_ms(
        np.array([0.99, 1.0, 0.0], dtype=np.float32),
        np.array([1e6, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    assert np.all(latency <= MAX_LINK_LATENCY_MS)
    assert np.all(latency >= 0.0)


def test_flow_latency_sums_along_the_path() -> None:
    incidence = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    per_link = np.array([2.0, 3.0, 100.0], dtype=np.float32)

    assert flow_latency_ms(incidence, per_link)[0] == pytest.approx(5.0)


def test_p95_reflects_the_tail_not_the_average() -> None:
    # Right-skewed: most flows fast, a couple badly delayed. p95 must surface the tail the mean dilutes.
    latencies = np.array([1.0] * 18 + [200.0, 500.0], dtype=np.float32)

    assert percentile_latency_ms(latencies) > float(np.mean(latencies))


def test_p95_of_no_flows_is_zero() -> None:
    assert percentile_latency_ms(np.zeros(0, dtype=np.float32)) == 0.0
