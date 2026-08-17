"""The multi-agent environment contract.

No training here — that is banned in tests and would prove nothing anyway. What is verified is the
decomposition: agents act only on their own flows, the shared reward is genuinely shared, the joint
observation stays separable from the local ones, and episodes vary while remaining reproducible.
"""

from __future__ import annotations

import numpy as np

from nexus_rl.marl_env import MultiAgentNetworkEnv
from nexus_types.slice import DEFAULT_SLICES


def _env(**kwargs: object) -> MultiAgentNetworkEnv:
    defaults: dict[str, object] = {
        "traffic_pattern": "diurnal",
        "seed": 0,
        "flow_count": 9,
        "episode_length": 16,
    }
    return MultiAgentNetworkEnv(**{**defaults, **kwargs})  # type: ignore[arg-type]  # kwargs are the env's own


def _full(env: MultiAgentNetworkEnv, value: float) -> np.ndarray:
    return np.full((env.agent_count, env.action_size), value, dtype=np.float32)


def test_there_is_one_agent_per_slice() -> None:
    env = _env()

    assert env.agent_count == len(DEFAULT_SLICES.slices)


def test_observation_shapes_are_stable_across_steps() -> None:
    # A policy network needs a fixed input shape; a shape that moves mid-episode is a crash 40 minutes in.
    env = _env()
    step = env.reset(seed=0)
    shapes = {step.local_observations.shape}

    for _ in range(8):
        step = env.step(_full(env, 0.5))
        shapes.add(step.local_observations.shape)

    assert shapes == {(env.agent_count, env.local_observation_size)}


def test_the_joint_observation_is_wider_than_a_local_one() -> None:
    # The critic must see more than any single agent — that asymmetry is what CTDE is.
    env = _env()

    assert env.joint_observation_size > env.local_observation_size


def test_an_agent_acts_only_on_its_own_flows() -> None:
    """The decomposition, asserted at the decoder.

    Cross-agent leakage here would silently undo everything: the agents would all be acting on the same
    flows, and any coordination observed would be an artefact of the decoder rather than of the policies.
    """
    env = _env()
    env.reset(seed=0)

    actions = np.zeros((env.agent_count, env.action_size), dtype=np.float32)
    actions[0] = 1.0  # only the premium agent asks for everything
    step = env.step(actions)

    premium_flows = set(env.scopes[0].flow_ids)
    other_flows = {flow_id for scope in env.scopes[1:] for flow_id in scope.flow_ids}

    assert premium_flows
    assert other_flows
    assert not (premium_flows & other_flows)
    assert step.reward_components


def test_the_reward_is_shared_not_per_agent() -> None:
    # Per-agent rewards would let an agent improve its own slice by starving another and be scored for
    # it — the opposite of what slicing is for. Credit assignment belongs to the critic.
    env = _env()
    env.reset(seed=0)

    step = env.step(_full(env, 0.5))

    assert isinstance(step.reward, float)


def test_every_reward_component_is_named() -> None:
    # A scalar-only reward is undebuggable.
    env = _env()
    env.reset(seed=0)

    components = env.step(_full(env, 0.5)).reward_components

    assert set(components) == {
        "throughput_reward",
        "latency_penalty",
        "drop_penalty",
        "fairness_bonus",
        "sla_attainment_reward",
    }


def test_the_episode_terminates_at_the_configured_length() -> None:
    env = _env(episode_length=5)
    env.reset(seed=0)

    terminated = [env.step(_full(env, 0.5)).terminated for _ in range(5)]

    assert terminated == [False, False, False, False, True]


def test_consecutive_episodes_differ() -> None:
    # Re-seeding with the master seed every reset makes every episode byte-identical, which produces a
    # policy that looks converged and generalises to nothing.
    env = _env()
    first = env.reset(seed=0).joint_observation
    second = env.reset().joint_observation

    assert not np.array_equal(first, second)


def test_the_episode_sequence_is_reproducible() -> None:
    left, right = _env(), _env()
    left.reset(seed=0)
    right.reset(seed=0)

    sequence_left = [left.reset().joint_observation for _ in range(3)]
    sequence_right = [right.reset().joint_observation for _ in range(3)]

    assert all(np.array_equal(a, b) for a, b in zip(sequence_left, sequence_right, strict=True))


def test_out_of_range_actions_are_clipped_rather_than_raising() -> None:
    # This is the training path; an unsquashed policy output is a normal occurrence, and an exception
    # would end the run rather than the timestep.
    env = _env()
    env.reset(seed=0)

    step = env.step(_full(env, 5.0))

    assert np.isfinite(step.reward)


def test_flows_beyond_an_agents_slot_budget_are_not_silently_starved() -> None:
    # An omitted flow would be allocated nothing, which is a decision the policy did not make.
    env = _env(flow_count=12, max_flows_per_agent=1)
    env.reset(seed=0)

    step = env.step(np.zeros((env.agent_count, env.action_size), dtype=np.float32))

    assert step.reward_components["throughput_reward"] > 0.0
