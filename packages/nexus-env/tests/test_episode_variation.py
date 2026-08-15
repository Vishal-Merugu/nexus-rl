"""Episode-variation tests.

Regression for the highest-impact RL bug found in audit: a bare ``reset()`` re-seeded with the master seed,
so every training episode was byte-identical. A policy would train on one trajectory forever, look
converged, and generalise to nothing — and every diagnostic would point at reward shaping instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_env.environment import NexusNetworkEnv
from nexus_errors.simulation_error import TopologyError
from nexus_sim.flow import build_flows
from nexus_sim.topology import build_topology


def _env(**overrides: object) -> NexusNetworkEnv:
    kwargs: dict[str, object] = {
        "topology_name": "mesh-6",
        "traffic_pattern_name": "bursty",
        "seed": 5,
        "episode_length": 16,
    }
    kwargs.update(overrides)
    return NexusNetworkEnv(**kwargs)  # type: ignore[arg-type]


def _episode(env: NexusNetworkEnv, steps: int = 8) -> list[float]:
    env.reset()
    return [float(env.step(np.int64(0))[1]) for _ in range(steps)]


def test_consecutive_episodes_differ() -> None:
    env = _env()

    first, second, third = _episode(env), _episode(env), _episode(env)

    assert first != second
    assert second != third


def test_the_sequence_of_episodes_replays_from_the_master_seed() -> None:
    # Variation must not cost reproducibility: the same master seed yields the same run, episode for episode.
    def run() -> list[list[float]]:
        env = _env()
        return [_episode(env) for _ in range(3)]

    assert run() == run()


def test_a_seeded_reset_restarts_the_run() -> None:
    env = _env()
    _episode(env)
    _episode(env)

    env.reset(seed=99)
    after_first = [float(env.step(np.int64(0))[1]) for _ in range(8)]

    fresh = _env()
    fresh.reset(seed=99)
    after_fresh = [float(fresh.step(np.int64(0))[1]) for _ in range(8)]

    assert after_first == after_fresh


def test_the_flow_set_is_stable_across_episodes() -> None:
    # Topology and flows are the *network*; only demand advances. A changing flow set between episodes
    # would silently change the observation's meaning mid-run.
    env = _env()
    before = env.simulator.flows
    _episode(env)
    _episode(env)

    assert env.simulator.flows == before


def test_requesting_more_flows_than_the_topology_supports_is_refused() -> None:
    # Silent truncation produced an observation shorter than the declared space, surfacing as a shape
    # error deep inside the training loop.
    topology = build_topology("mesh-6")

    with pytest.raises(TopologyError, match="distinct source/target pairs"):
        build_flows(topology, np.random.default_rng(0), flow_count=100)


def test_an_oversized_flow_count_fails_at_construction_not_at_training_time() -> None:
    with pytest.raises(TopologyError):
        NexusNetworkEnv(topology_name="mesh-6", flow_count=100)


def test_the_observation_always_matches_the_declared_space() -> None:
    for flow_count in (1, 4, 8, 30):
        env = NexusNetworkEnv(topology_name="mesh-6", flow_count=flow_count, episode_length=4)
        observation, _ = env.reset()
        assert env.observation_space.contains(observation), f"mismatch at flow_count={flow_count}"


def test_a_negative_flow_count_is_refused() -> None:
    with pytest.raises(TopologyError):
        build_flows(build_topology("mesh-6"), np.random.default_rng(0), flow_count=-1)


def test_a_non_positive_episode_length_is_refused() -> None:
    from nexus_errors.config_error import ConfigurationError

    for length in (0, -10):
        with pytest.raises(ConfigurationError, match="episode_length must be positive"):
            NexusNetworkEnv(episode_length=length)


def test_a_non_finite_action_blames_divergence_not_the_mdp() -> None:
    # Clipping leaves NaN as NaN, which tripped the allocation validator with a message about shape
    # mismatches and MDP_VERSION — pointing at entirely the wrong cause.
    from nexus_errors.control_error import ActionDecodingError

    env = NexusNetworkEnv(action_mode="continuous", flow_count=8, episode_length=32)
    env.reset()

    with pytest.raises(ActionDecodingError, match="diverged"):
        env.step(np.full(8, np.nan, dtype=np.float32))
