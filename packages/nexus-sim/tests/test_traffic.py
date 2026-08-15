from __future__ import annotations

import numpy as np
import pytest

from nexus_errors.simulation_error import UnknownTrafficPatternError
from nexus_sim.constants import DIURNAL_PERIOD_TIMESTEPS, FLASH_CROWD_START
from nexus_sim.traffic import available_traffic_patterns, get_traffic_pattern

_BASE = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32)


@pytest.mark.parametrize("name", available_traffic_patterns())
def test_same_seed_produces_the_same_demand_sequence(name: str) -> None:
    pattern = get_traffic_pattern(name)

    first = [pattern.demand_mbps(t, _BASE, np.random.default_rng(4)) for t in range(5)]
    second = [pattern.demand_mbps(t, _BASE, np.random.default_rng(4)) for t in range(5)]

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("name", available_traffic_patterns())
def test_different_seeds_produce_different_demand(name: str) -> None:
    # A pattern that ignores its rng is a real bug that otherwise hides for months.
    pattern = get_traffic_pattern(name)

    first = pattern.demand_mbps(0, _BASE, np.random.default_rng(1))
    second = pattern.demand_mbps(0, _BASE, np.random.default_rng(2))

    assert not np.array_equal(first, second)


@pytest.mark.parametrize("name", available_traffic_patterns())
def test_demand_is_non_negative_and_finite(name: str) -> None:
    pattern = get_traffic_pattern(name)
    rng = np.random.default_rng(0)

    for timestep in range(200):
        demand = pattern.demand_mbps(timestep, _BASE, rng)
        assert np.all(demand >= 0.0)
        assert np.all(np.isfinite(demand))


def test_unknown_pattern_lists_what_is_available() -> None:
    with pytest.raises(UnknownTrafficPatternError) as excinfo:
        get_traffic_pattern("hurricane")

    assert "available" in excinfo.value.context


def test_diurnal_demand_actually_cycles() -> None:
    pattern = get_traffic_pattern("diurnal")
    rng = np.random.default_rng(0)

    totals = [
        float(np.sum(pattern.demand_mbps(t, _BASE, rng)))
        for t in range(0, DIURNAL_PERIOD_TIMESTEPS, DIURNAL_PERIOD_TIMESTEPS // 8)
    ]

    assert max(totals) > 1.5 * min(totals)


def test_bursty_demand_has_a_heavier_tail_than_steady() -> None:
    steady = get_traffic_pattern("steady")
    bursty = get_traffic_pattern("bursty")
    rng_steady = np.random.default_rng(7)
    rng_bursty = np.random.default_rng(7)

    steady_max = max(float(np.max(steady.demand_mbps(t, _BASE, rng_steady))) for t in range(300))
    bursty_max = max(float(np.max(bursty.demand_mbps(t, _BASE, rng_bursty))) for t in range(300))

    assert bursty_max > 2.0 * steady_max


def test_flash_crowd_spikes_only_after_its_trigger_timestep() -> None:
    pattern = get_traffic_pattern("flash_crowd")
    rng = np.random.default_rng(0)

    before = float(np.sum(pattern.demand_mbps(FLASH_CROWD_START - 1, _BASE, rng)))
    after = float(np.sum(pattern.demand_mbps(FLASH_CROWD_START, _BASE, rng)))

    assert after > 2.0 * before


def test_flash_crowd_concentrates_on_a_subset_of_flows() -> None:
    pattern = get_traffic_pattern("flash_crowd")

    demand = pattern.demand_mbps(FLASH_CROWD_START, _BASE, np.random.default_rng(0))

    affected = _BASE.size // 2
    assert float(np.mean(demand[:affected])) > float(np.mean(demand[affected:]))
