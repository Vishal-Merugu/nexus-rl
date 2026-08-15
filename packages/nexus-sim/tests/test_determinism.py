"""Determinism tests.

The most valuable tests in the repo. A simulator whose trajectory depends on hidden state cannot be
debugged: "same seed, same config, different reward" burns hours and looks like an RL problem when it is a
plumbing problem. A failure here is CRITICAL regardless of what else passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_sim.simulator import NetworkSimulator
from nexus_types.action import AllocationAction


def _trajectory(seed: int, steps: int = 25, pattern: str = "bursty") -> list[tuple[float, float, float]]:
    simulator = NetworkSimulator.from_config(topology_name="mesh-6", traffic_pattern_name=pattern, seed=seed)
    trajectory: list[tuple[float, float, float]] = []
    for _ in range(steps):
        result = simulator.step()
        trajectory.append(
            (
                result.metrics.total_throughput_mbps,
                result.metrics.p95_latency_ms,
                result.metrics.fairness_index,
            )
        )
    return trajectory


def test_same_seed_produces_an_identical_trajectory() -> None:
    assert _trajectory(seed=7) == _trajectory(seed=7)


def test_different_seeds_produce_different_trajectories() -> None:
    # A simulator that ignores its seed would silently pass every other determinism test.
    assert _trajectory(seed=7) != _trajectory(seed=8)


def test_reset_with_a_fresh_generator_replays_the_same_trajectory() -> None:
    simulator = NetworkSimulator.from_config(topology_name="mesh-6", traffic_pattern_name="bursty", seed=11)

    def run() -> list[float]:
        simulator.reset(np.random.default_rng(99))
        return [simulator.step().metrics.total_throughput_mbps for _ in range(15)]

    assert run() == run()


def test_identical_action_sequences_produce_identical_trajectories() -> None:
    actions = [
        AllocationAction(share_by_flow_id={f"f{index}": 0.1 * ((index + step) % 10)})
        for step in range(10)
        for index in range(8)
    ][:10]

    def run() -> list[float]:
        simulator = NetworkSimulator.from_config(
            topology_name="mesh-6", traffic_pattern_name="diurnal", seed=3
        )
        return [simulator.step(action).metrics.total_throughput_mbps for action in actions]

    assert run() == run()


def test_flow_construction_is_seeded() -> None:
    first = NetworkSimulator.from_config("mesh-6", "steady", seed=5).flows
    second = NetworkSimulator.from_config("mesh-6", "steady", seed=5).flows
    other = NetworkSimulator.from_config("mesh-6", "steady", seed=6).flows

    assert first == second
    assert first != other


@pytest.mark.parametrize("pattern", ["steady", "diurnal", "bursty", "flash_crowd"])
def test_every_traffic_pattern_is_deterministic(pattern: str) -> None:
    assert _trajectory(seed=42, steps=10, pattern=pattern) == _trajectory(seed=42, steps=10, pattern=pattern)
