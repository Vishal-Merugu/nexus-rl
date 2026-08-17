"""MAPPO — Multi-Agent PPO with a centralized critic.

Hand-rolled rather than taken from a library, for one reason: Stable-Baselines3 is single-agent, and every
way of bending it into a multi-agent shape (flattening agents into one policy, or running N independent
PPOs) destroys the thing being built. Flattening removes the decomposition; independent PPO removes the
centralized critic and with it the answer to non-stationarity. What is actually needed here is small —
PPO's clipped objective, with the value function reading a different input from the policy — so it is
written out.

The two properties that make this MAPPO rather than "PPO run three times":

1. **A centralized critic.** ``V(joint_observation)`` — the critic sees global state. That is what makes
   the advantage estimate meaningful when each agent's environment includes two other agents who are also
   learning. An independent critic on local observations would be estimating the value of a
   non-stationary environment and would chase its own tail.

2. **A shared actor conditioned on the agent's own observation.** Parameter sharing across agents is the
   standard MAPPO choice for homogeneous agents and it is the right one here: the agents differ by *slice
   priority*, which is a feature in their observation, not by role. Sharing means three times the data per
   update instead of a third of it each.

Execution stays decentralized: an agent's action is a function of its own local observation alone. The
critic is used to compute advantages during training and is then discarded — nothing at deployment ever
calls it.

**No result is claimed here.** This trains; whether the trained policy beats the single-agent baseline is
a question for `nexus-eval` over multiple seeds, and the answer is not yet known.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from nexus_errors.config_error import ConfigurationError
from nexus_rl.marl_env import MultiAgentNetworkEnv
from nexus_telemetry.client import get_telemetry

_telemetry = get_telemetry("nexus.mappo")


@dataclass(frozen=True, slots=True)
class MappoConfig:
    """Hyperparameters. From a config file, never inline at a call site."""

    total_timesteps: int = 50_000
    rollout_length: int = 256
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0 or self.rollout_length <= 0:
            raise ConfigurationError(
                "total_timesteps and rollout_length must be positive.",
                {"total_timesteps": self.total_timesteps, "rollout_length": self.rollout_length},
            )
        if self.rollout_length % self.minibatches != 0:
            raise ConfigurationError(
                "rollout_length must divide evenly into minibatches, or the last update sees a "
                "different batch size than the rest and the gradient scale shifts mid-run.",
                {"rollout_length": self.rollout_length, "minibatches": self.minibatches},
            )


@dataclass
class TrainingProgress:
    """What a run recorded. Components individually, never just the total."""

    timesteps: int = 0
    episode_rewards: list[float] = field(default_factory=list)
    component_means: list[dict[str, float]] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)

    @property
    def mean_episode_reward(self) -> float:
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0


def train_mappo(
    env: MultiAgentNetworkEnv,
    config: MappoConfig,
    checkpoint_path: Path | None = None,
) -> TrainingProgress:
    """Train a shared actor with a centralized critic.

    torch is imported here rather than at module scope so that importing ``nexus_rl`` — which the
    evaluation path does, without needing to train — does not pull in torch.
    """
    try:
        import torch
        from torch import nn, optim
    except ImportError as error:  # pragma: no cover - exercised only without the extra installed
        raise ConfigurationError(
            "Training MAPPO needs the optional 'train' extra: make install",
            {"missing": str(error)},
        ) from error

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    agents = env.agent_count
    obs_size = env.local_observation_size
    joint_size = env.joint_observation_size
    action_size = env.action_size

    class Actor(nn.Module):
        """Gaussian policy over per-flow shares, squashed into [0, 1].

        A state-independent log-sigma: the standard PPO choice, and it keeps exploration from collapsing the
        moment the mean is good — which on this problem happens early, because a mediocre allocation
        already scores well.
        """

        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(obs_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
            )
            self.mean = nn.Linear(config.hidden_size, action_size)
            self.log_std = nn.Parameter(torch.zeros(action_size) - 0.5)

        def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
            mean = self.mean(self.body(observation))
            return torch.distributions.Normal(mean, self.log_std.exp())

    class Critic(nn.Module):
        """V(joint state). Centralized — this is the half that makes it MAPPO."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(joint_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, 1),
            )

        def forward(self, joint: torch.Tensor) -> torch.Tensor:
            value: torch.Tensor = self.net(joint)
            return value.squeeze(-1)

    actor, critic = Actor(), Critic()
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=config.learning_rate)

    progress = TrainingProgress()
    step = env.reset(seed=config.seed)
    episode_reward = 0.0
    episode_components: list[dict[str, float]] = []

    while progress.timesteps < config.total_timesteps:
        obs_buffer = np.zeros((config.rollout_length, agents, obs_size), dtype=np.float32)
        joint_buffer = np.zeros((config.rollout_length, joint_size), dtype=np.float32)
        action_buffer = np.zeros((config.rollout_length, agents, action_size), dtype=np.float32)
        logprob_buffer = np.zeros((config.rollout_length, agents), dtype=np.float32)
        reward_buffer = np.zeros(config.rollout_length, dtype=np.float32)
        done_buffer = np.zeros(config.rollout_length, dtype=np.float32)

        for index in range(config.rollout_length):
            obs_buffer[index] = step.local_observations
            joint_buffer[index] = step.joint_observation

            with torch.no_grad():
                distribution = actor.distribution(torch.from_numpy(step.local_observations))
                raw = distribution.sample()  # type: ignore[no-untyped-call]  # untyped torch.distributions
                # torch.distributions is untyped, so mypy cannot see these are safe.
                logprob_buffer[index] = distribution.log_prob(raw).sum(-1).numpy()  # type: ignore[no-untyped-call]  # untyped torch.distributions
            actions = torch.sigmoid(raw).numpy().astype(np.float32)
            action_buffer[index] = raw.numpy()

            step = env.step(actions)
            reward_buffer[index] = step.reward
            done_buffer[index] = float(step.terminated)
            episode_reward += step.reward
            episode_components.append(step.reward_components)
            progress.timesteps += 1

            if step.terminated:
                progress.episode_rewards.append(episode_reward)
                progress.component_means.append(_mean_components(episode_components))
                episode_reward, episode_components = 0.0, []
                step = env.reset()

        # --- Advantages, from the centralized critic ------------------------------------------------
        with torch.no_grad():
            values = critic(torch.from_numpy(joint_buffer)).numpy()
            bootstrap = float(critic(torch.from_numpy(step.joint_observation[None])).item())

        advantages = np.zeros(config.rollout_length, dtype=np.float32)
        running = 0.0
        for index in reversed(range(config.rollout_length)):
            next_value = bootstrap if index == config.rollout_length - 1 else values[index + 1]
            non_terminal = 1.0 - done_buffer[index]
            delta = reward_buffer[index] + config.gamma * next_value * non_terminal - values[index]
            running = delta + config.gamma * config.gae_lambda * non_terminal * running
            advantages[index] = running
        returns = advantages + values

        # Normalised per update. Unnormalised advantages make the effective learning rate depend on the
        # reward scale, so a reward-weight change silently becomes a learning-rate change.
        normalised = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.from_numpy(obs_buffer)
        joint_t = torch.from_numpy(joint_buffer)
        action_t = torch.from_numpy(action_buffer)
        old_logprob_t = torch.from_numpy(logprob_buffer)
        advantage_t = torch.from_numpy(normalised)
        return_t = torch.from_numpy(returns)

        batch = config.rollout_length // config.minibatches
        for _ in range(config.epochs):
            order = rng.permutation(config.rollout_length)
            for start in range(0, config.rollout_length, batch):
                idx = order[start : start + batch]

                distribution = actor.distribution(obs_t[idx])
                logprob = distribution.log_prob(action_t[idx]).sum(-1)  # type: ignore[no-untyped-call]  # untyped torch.distributions
                entropy = distribution.entropy().sum(-1).mean()  # type: ignore[no-untyped-call]  # untyped torch.distributions

                # The advantage is shared across agents — one cooperative reward, one advantage, applied
                # to every agent's ratio. That is the credit-assignment story: the critic decides how good
                # the joint situation was, and each agent is pushed by the same signal.
                ratio = (logprob - old_logprob_t[idx]).exp()
                shared_advantage = advantage_t[idx].unsqueeze(-1)
                unclipped = ratio * shared_advantage
                clipped = (
                    torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * shared_advantage
                )
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = ((critic(joint_t[idx]) - return_t[idx]) ** 2).mean()
                loss = (
                    policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), config.max_grad_norm
                )
                optimizer.step()

                progress.policy_losses.append(float(policy_loss.item()))
                progress.value_losses.append(float(value_loss.item()))
                progress.entropies.append(float(entropy.item()))

        _telemetry.info(
            "MAPPO update",
            {
                "timesteps": progress.timesteps,
                "episodes": len(progress.episode_rewards),
                "mean_episode_reward": progress.mean_episode_reward,
                "policy_loss": progress.policy_losses[-1],
                "value_loss": progress.value_losses[-1],
                "entropy": progress.entropies[-1],
            },
        )

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": actor.state_dict(),
                # The critic is saved for resuming training and for auditing, never for serving. Nothing
                # at execution time reads global state — that is the whole CTDE distinction.
                "critic": critic.state_dict(),
                # asdict, not __dict__: MappoConfig is a slots dataclass and has no __dict__.
                "config": asdict(config),
                "local_observation_size": obs_size,
                "action_size": action_size,
            },
            checkpoint_path,
        )

    return progress


def _mean_components(components: list[dict[str, float]]) -> dict[str, float]:
    """Average each named reward term over an episode.

    Never collapses to a total: a flat total with a rising throughput term and a falling SLA term tells
    you exactly what happened, and a scalar tells you nothing.
    """
    if not components:
        return {}
    keys = components[0].keys()
    return {key: float(np.mean([c.get(key, 0.0) for c in components])) for key in keys}
