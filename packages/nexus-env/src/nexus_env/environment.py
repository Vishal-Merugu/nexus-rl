"""The gymnasium environment — the MDP boundary.

This is the only place the observation encoding, action space, and reward definition come together. Nothing
outside this package encodes them, so there is exactly one thing to review when the MDP changes.
"""

from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from nexus_errors.config_error import ConfigurationError
from nexus_mdp.action_space import (
    DISCRETE_ACTION_COUNT,
    FlowRoutingOptions,
    decode_continuous_action,
    decode_discrete_action,
)
from nexus_mdp.observation import encode_observation, observation_size
from nexus_mdp.reward import RewardWeights, compute_reward, total_reward
from nexus_mdp.version import MDP_VERSION
from nexus_sim.constants import DEFAULT_FLOW_COUNT
from nexus_sim.simulator import NetworkSimulator
from nexus_types.action import AllocationAction
from nexus_types.constants import DEFAULT_EPISODE_LENGTH
from nexus_types.incident import Incident

ActionMode = Literal["discrete", "continuous"]


class NexusNetworkEnv(gym.Env[NDArray[np.float32], np.int64 | NDArray[np.float32]]):
    """Network resource allocation as an MDP.

    Args:
        topology_name: Named topology from ``nexus_sim.topology``.
        traffic_pattern_name: Named demand generator. Hold at least one pattern out of training so
            generalisation can actually be demonstrated.
        seed: Master seed. Determines the flow set and every demand draw.
        action_mode: ``"discrete"`` (v1, DQN) or ``"continuous"`` (Stage G, PPO). See ADR-0004.
        flow_count: Number of flows. Changes the observation shape, so it is part of the MDP contract.
        episode_length: Timesteps before truncation.
        incidents: Scheduled degradations. Part of the environment, not the action space.
        reward_weights: Component weights. Changing these is an MDP version bump.
    """

    # Type comes from gymnasium's own `Env.metadata`; re-declaring it as a ClassVar conflicts with
    # the instance variable the base class declares.
    metadata = {"render_modes": []}  # noqa: RUF012 — ClassVar here conflicts with gymnasium's base declaration

    def __init__(
        self,
        topology_name: str = "mesh-6",
        traffic_pattern_name: str = "diurnal",
        seed: int = 0,
        action_mode: ActionMode = "discrete",
        flow_count: int = DEFAULT_FLOW_COUNT,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        incidents: tuple[Incident, ...] = (),
        reward_weights: RewardWeights | None = None,
    ) -> None:
        super().__init__()
        if episode_length <= 0:
            raise ConfigurationError(
                "episode_length must be positive; a non-positive value truncates after a single step.",
                {"episode_length": episode_length},
            )
        self._topology_name = topology_name
        self._traffic_pattern_name = traffic_pattern_name
        self._seed = seed
        self._action_mode: ActionMode = action_mode
        self._flow_count = flow_count
        self._episode_length = episode_length
        self._incidents = incidents
        self._reward_weights = reward_weights or RewardWeights()

        self._simulator = self._build_simulator(seed)
        self._routing = self._build_routing()
        self._steps_taken = 0
        self._episode_index = 0

        size = observation_size(len(self._simulator.topology.links), flow_count)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)
        self.action_space = (
            spaces.Discrete(DISCRETE_ACTION_COUNT)
            if action_mode == "discrete"
            else spaces.Box(low=0.0, high=1.0, shape=(flow_count,), dtype=np.float32)
        )

    @property
    def mdp_version(self) -> int:
        """The contract version this environment implements. Recorded in every run artifact."""
        return MDP_VERSION

    @property
    def simulator(self) -> NetworkSimulator:
        """The underlying simulator. Read-only in spirit — the env owns stepping it."""
        return self._simulator

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        """Start a new episode.

        **A bare reset advances to the next episode of the run, not a repeat of the first.** Training
        loops call ``reset()`` without a seed between episodes; re-seeding with the master seed each time
        made every episode byte-identical, so a policy would train on one trajectory forever and appear to
        converge while learning nothing transferable.

        Episode ``n`` draws from ``default_rng(seed + n)``: different demand each episode, and the whole
        sequence still replays exactly from the master seed.

        Passing a seed explicitly restarts the run — it rebuilds the simulator including the flow set,
        which is what makes a seeded episode reproducible down to the flow layout.
        """
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._episode_index = 0
            self._simulator = self._build_simulator(seed)
            self._routing = self._build_routing()
        else:
            # The flow set and topology are the *network*, and stay fixed across episodes; only the demand
            # stream advances.
            self._episode_index += 1
            self._simulator.reset(np.random.default_rng(self._seed + self._episode_index))

        self._steps_taken = 0
        state = self._simulator.state
        return encode_observation(state), {"mdp_version": MDP_VERSION, "timestep": state.timestep}

    def step(
        self, action: np.int64 | NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        """Apply an action and advance one timestep.

        ``info["reward_components"]`` carries every named term. Diagnose from those, never from the scalar.
        """
        allocation = self._decode(action)
        result = self._simulator.step(allocation)
        self._steps_taken += 1

        components = compute_reward(result.state, self._reward_weights)
        reward = total_reward(components)

        truncated = self._steps_taken >= self._episode_length
        info: dict[str, object] = {
            "reward_components": components,
            "metrics": result.metrics.as_dict(),
            "timestep": result.state.timestep,
            "mdp_version": MDP_VERSION,
            "active_incident_ids": result.state.active_incident_ids,
        }
        # Never terminates on its own: a network under load has no goal state to reach, and an early
        # termination signal would teach the policy to seek one.
        return encode_observation(result.state), reward, False, truncated, info

    def _decode(self, action: np.int64 | NDArray[np.float32]) -> AllocationAction:
        state = self._simulator.state
        if self._action_mode == "discrete":
            return decode_discrete_action(int(np.asarray(action).item()), state, self._routing)
        return decode_continuous_action(np.asarray(action, dtype=np.float32).ravel(), state)

    def _build_simulator(self, seed: int) -> NetworkSimulator:
        return NetworkSimulator.from_config(
            topology_name=self._topology_name,
            traffic_pattern_name=self._traffic_pattern_name,
            seed=seed,
            flow_count=self._flow_count,
            incidents=self._incidents,
        )

    def _build_routing(self) -> tuple[FlowRoutingOptions, ...]:
        return tuple(
            FlowRoutingOptions(flow_id=flow.flow_id, candidate_paths=flow.candidate_paths)
            for flow in self._simulator.flows
        )
