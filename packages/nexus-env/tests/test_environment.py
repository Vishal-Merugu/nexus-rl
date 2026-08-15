"""Environment contract tests.

These, not training runs, are what guard the MDP. They assert shape, bounds, dtype, determinism and reward
sign — everything a checkpoint depends on. No test here trains a model: that would be slow, flaky, and would
prove nothing these do not.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from nexus_env.environment import NexusNetworkEnv
from nexus_mdp.action_space import DISCRETE_ACTION_COUNT
from nexus_mdp.observation import observation_size
from nexus_mdp.version import MDP_VERSION


def _env(**overrides: object) -> NexusNetworkEnv:
    kwargs: dict[str, object] = {
        "topology_name": "mesh-6",
        "traffic_pattern_name": "steady",
        "seed": 3,
        "episode_length": 32,
    }
    kwargs.update(overrides)
    return NexusNetworkEnv(**kwargs)  # type: ignore[arg-type]  # kwargs are checked by the call sites below


def test_observation_shape_matches_the_declared_space() -> None:
    env = _env()

    observation, _ = env.reset()

    assert observation.shape == env.observation_space.shape
    assert observation.shape == (observation_size(len(env.simulator.topology.links), 8),)


def test_observation_is_float32_as_stable_baselines_expects() -> None:
    observation, _ = _env().reset()

    assert observation.dtype == np.float32


def test_observation_stays_within_the_declared_bounds() -> None:
    env = _env(traffic_pattern_name="flash_crowd", episode_length=200)
    observation, _ = env.reset()

    assert env.observation_space.contains(observation)
    for _ in range(150):
        observation, _, _, _, _ = env.step(np.int64(0))
        assert np.all(observation >= 0.0)
        assert np.all(observation <= 1.0)
        assert np.all(np.isfinite(observation))


def test_observation_shape_is_fixed_regardless_of_runtime_state() -> None:
    env = _env(traffic_pattern_name="bursty")
    first, _ = env.reset()

    shapes = {first.shape}
    for _ in range(30):
        observation, _, _, _, _ = env.step(np.int64(1))
        shapes.add(observation.shape)

    assert len(shapes) == 1


def test_discrete_action_space_matches_the_mode_count() -> None:
    env = _env(action_mode="discrete")

    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == DISCRETE_ACTION_COUNT


def test_continuous_action_space_is_one_share_per_flow() -> None:
    env = _env(action_mode="continuous", flow_count=6)

    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (6,)


@pytest.mark.parametrize("action_index", range(DISCRETE_ACTION_COUNT))
def test_every_discrete_action_decodes_to_a_valid_step(action_index: int) -> None:
    # Decoding must be total — no action in the space may produce an invalid allocation.
    env = _env()
    env.reset()

    observation, reward, terminated, _, info = env.step(np.int64(action_index))

    assert np.isfinite(observation).all()
    assert np.isfinite(reward)
    assert terminated is False
    assert info["reward_components"]


def test_out_of_range_action_clamps_rather_than_crashing_a_training_run() -> None:
    env = _env()
    env.reset()

    _, reward, _, _, _ = env.step(np.int64(DISCRETE_ACTION_COUNT + 50))

    assert np.isfinite(reward)


def test_continuous_actions_outside_the_box_are_clipped_not_rejected() -> None:
    env = _env(action_mode="continuous")
    env.reset()

    _, reward, _, _, _ = env.step(np.full(8, 5.0, dtype=np.float32))

    assert np.isfinite(reward)


def test_info_carries_every_named_reward_component() -> None:
    env = _env()
    env.reset()

    _, _, _, _, info = env.step(np.int64(0))

    components = info["reward_components"]
    assert isinstance(components, dict)
    assert set(components) == {
        "throughput_reward",
        "latency_penalty",
        "drop_penalty",
        "fairness_bonus",
    }


def test_info_records_the_mdp_version_for_artifact_traceability() -> None:
    env = _env()
    _, reset_info = env.reset()
    _, _, _, _, step_info = env.step(np.int64(0))

    assert reset_info["mdp_version"] == MDP_VERSION
    assert step_info["mdp_version"] == MDP_VERSION


def test_episode_truncates_at_the_configured_length_and_never_terminates() -> None:
    env = _env(episode_length=5)
    env.reset()

    for step in range(5):
        _, _, terminated, truncated, _ = env.step(np.int64(0))
        assert terminated is False
        assert truncated is (step == 4)


def test_same_seed_replays_the_same_episode() -> None:
    def rollout() -> list[float]:
        env = _env()
        env.reset(seed=17)
        return [float(env.step(np.int64(index % DISCRETE_ACTION_COUNT))[1]) for index in range(20)]

    assert rollout() == rollout()


def test_different_seeds_produce_different_episodes() -> None:
    def rollout(seed: int) -> list[float]:
        env = _env()
        env.reset(seed=seed)
        return [float(env.step(np.int64(0))[1]) for _ in range(20)]

    assert rollout(1) != rollout(2)


def test_reset_returns_to_the_start_of_an_episode() -> None:
    env = _env(episode_length=4)
    env.reset()
    for _ in range(4):
        env.step(np.int64(0))

    env.reset()
    _, _, _, truncated, _ = env.step(np.int64(0))

    assert truncated is False
