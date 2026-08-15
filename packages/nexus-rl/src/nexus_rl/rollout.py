"""Rollouts — running an allocator over the simulator and collecting metrics.

Used identically for baselines and for trained policies, which is what makes the comparison honest: both go
through the same code path, the same simulator, the same seed, and the same metric computation. A comparison
where the policy and the baseline are measured by different code proves nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nexus_baselines.allocator import Allocator
from nexus_errors.config_error import ConfigurationError
from nexus_mdp.observation import encode_observation
from nexus_mdp.reward import RewardWeights, compute_reward, total_reward
from nexus_sim.simulator import NetworkSimulator
from nexus_types.action import AllocationAction
from nexus_types.network_state import NetworkState

PolicyPredictor = Callable[[NDArray[np.float32]], AllocationAction]
"""Maps an encoded observation to an allocation. Wraps a trained policy plus its action decoder."""


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One timestep of a rollout."""

    timestep: int
    reward_components: dict[str, float]
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    """Aggregated results for one rollout.

    Reports p95 latency rather than mean: QoS is a tail property, and a mean hides exactly the failures the
    reward is meant to punish. Fairness is reported alongside throughput, never instead of it — they trade
    off, and showing one alone misleads.
    """

    label: str
    seed: int
    traffic_pattern: str
    steps: int

    mean_reward: float
    mean_p95_latency_ms: float
    mean_throughput_mbps: float
    mean_fairness: float
    total_dropped_mb: float
    mean_reward_components: dict[str, float]
    records: tuple[StepRecord, ...]

    def headline(self) -> dict[str, float]:
        """The four metrics a result is reported in."""
        return {
            "p95_latency_ms": self.mean_p95_latency_ms,
            "throughput_mbps": self.mean_throughput_mbps,
            "fairness": self.mean_fairness,
            "dropped_mb": self.total_dropped_mb,
        }


def rollout_allocator(
    simulator: NetworkSimulator,
    allocator: Allocator,
    steps: int,
    seed: int,
    traffic_pattern: str,
    reward_weights: RewardWeights | None = None,
) -> RolloutSummary:
    """Run a non-learned allocator. Used for every baseline."""
    return _rollout(
        simulator=simulator,
        label=allocator.name,
        act=allocator.allocate,
        steps=steps,
        seed=seed,
        traffic_pattern=traffic_pattern,
        reward_weights=reward_weights,
    )


def rollout_policy(
    simulator: NetworkSimulator,
    predictor: PolicyPredictor,
    steps: int,
    seed: int,
    traffic_pattern: str,
    label: str = "policy",
    reward_weights: RewardWeights | None = None,
) -> RolloutSummary:
    """Run a trained policy through exactly the same path a baseline takes."""
    return _rollout(
        simulator=simulator,
        label=label,
        act=lambda state: predictor(encode_observation(state)),
        steps=steps,
        seed=seed,
        traffic_pattern=traffic_pattern,
        reward_weights=reward_weights,
    )


def _rollout(
    simulator: NetworkSimulator,
    label: str,
    act: Callable[[NetworkState], AllocationAction],
    steps: int,
    seed: int,
    traffic_pattern: str,
    reward_weights: RewardWeights | None,
) -> RolloutSummary:
    if steps <= 0:
        raise ConfigurationError("A rollout needs at least one step.", {"steps": steps})

    weights = reward_weights or RewardWeights()
    records: list[StepRecord] = []

    for _ in range(steps):
        action = act(simulator.state)
        result = simulator.step(action)
        components = compute_reward(result.state, weights)
        records.append(
            StepRecord(
                timestep=result.state.timestep,
                reward_components=components,
                metrics=result.metrics.as_dict(),
            )
        )

    component_names = records[0].reward_components.keys()
    return RolloutSummary(
        label=label,
        seed=seed,
        traffic_pattern=traffic_pattern,
        steps=len(records),
        mean_reward=float(np.mean([total_reward(r.reward_components) for r in records])),
        mean_p95_latency_ms=float(np.mean([r.metrics["p95_latency_ms"] for r in records])),
        mean_throughput_mbps=float(np.mean([r.metrics["total_throughput_mbps"] for r in records])),
        mean_fairness=float(np.mean([r.metrics["fairness_index"] for r in records])),
        total_dropped_mb=float(sum(r.metrics["total_dropped_mbps"] for r in records)),
        mean_reward_components={
            name: float(np.mean([record.reward_components[name] for record in records]))
            for name in component_names
        },
        records=tuple(records),
    )
