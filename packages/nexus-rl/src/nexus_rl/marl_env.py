"""The multi-agent environment — one agent per slice, over the shared network.

Not a gymnasium `Env`: gymnasium's contract is single-agent, and bending it into a multi-agent shape means
either flattening the agents (destroying the decomposition) or inventing a dict-space convention that only
this project understands. A small explicit contract is clearer than a bent standard one.

What it exposes, and why each piece exists:

- **Per-agent local observations** — what each policy consumes at execution time. Fixed width, genuinely
  local (see `nexus_mdp.multi_agent`).
- **A joint observation** — what the centralized critic consumes during training, and *only* then. Kept as
  a separate return value rather than a field on the local observations so that handing it to an
  execution-time policy has to be deliberate.
- **A single shared reward** — this is a *cooperative* problem. Per-agent rewards would let an agent
  improve its own slice by starving another and be scored for it, which is the opposite of what slicing
  is for. Credit assignment is the critic's job, not the reward function's.

Each agent controls the allocation shares of its own slice's flows and nothing else. That is the
decomposition: an agent cannot act on traffic it does not own, so the only way to affect another slice is
through the shared links — which is exactly the contention it has to learn to navigate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nexus_mdp.multi_agent import (
    AgentScope,
    build_scopes,
    encode_joint_observation,
    encode_local_observation,
    local_observation_size,
)
from nexus_mdp.observation import observation_size
from nexus_mdp.reward import RewardWeights, compute_reward, total_reward
from nexus_sim.randomization import IDENTITY_DOMAIN, SampledDomain
from nexus_sim.simulator import NetworkSimulator
from nexus_types.action import AllocationAction
from nexus_types.slice import DEFAULT_SLICES, SliceSet


@dataclass(frozen=True, slots=True)
class MarlStep:
    """One environment transition, from every agent's point of view."""

    local_observations: NDArray[np.float32]
    """Shape (agents, local_obs_size). Row i is agent i's view."""

    joint_observation: NDArray[np.float32]
    """Shape (joint_obs_size,). The centralized critic's input. Training only."""

    reward: float
    """Shared. Cooperative problem — see the module docstring."""

    reward_components: dict[str, float]
    """Named individually, never just the total. A flat total with a rising throughput term and a falling
    SLA term tells you exactly what happened; a scalar tells you nothing."""

    terminated: bool


class MultiAgentNetworkEnv:
    """One agent per slice, allocating over a shared network."""

    def __init__(
        self,
        topology: str = "mesh-6",
        traffic_pattern: str = "diurnal",
        seed: int = 0,
        flow_count: int = 9,
        episode_length: int = 256,
        slices: SliceSet = DEFAULT_SLICES,
        reward_weights: RewardWeights | None = None,
        domain: SampledDomain = IDENTITY_DOMAIN,
        max_flows_per_agent: int = 8,
        max_links_per_agent: int = 12,
    ) -> None:
        self._topology = topology
        self._traffic_pattern = traffic_pattern
        self._master_seed = seed
        self._flow_count = flow_count
        self._episode_length = episode_length
        self._slices = slices
        self._weights = reward_weights or RewardWeights()
        self._domain = domain
        self._max_flows = max_flows_per_agent
        self._max_links = max_links_per_agent

        self._episode = 0
        self._step_index = 0
        self._simulator = self._build(seed)
        self._scopes = build_scopes(self._simulator.state, slices)

    @property
    def agent_count(self) -> int:
        return len(self._scopes)

    @property
    def local_observation_size(self) -> int:
        return local_observation_size(self._max_flows, self._max_links)

    @property
    def joint_observation_size(self) -> int:
        return observation_size(len(self._simulator.state.links), len(self._simulator.state.flows))

    @property
    def action_size(self) -> int:
        """Per-agent action width: one share per owned flow slot.

        Fixed rather than per-agent-actual so every agent's policy has the same shape and a shared network
        is possible. Slots beyond an agent's flow count are ignored on decode.
        """
        return self._max_flows

    @property
    def scopes(self) -> tuple[AgentScope, ...]:
        return self._scopes

    def reset(self, seed: int | None = None) -> MarlStep:
        """Start a new episode.

        A bare reset advances the episode counter and draws from ``seed + episode`` — varied across
        episodes, and the whole sequence still replays exactly. Re-seeding with the master seed every time
        makes every episode byte-identical, which produces a policy that looks converged and generalises
        to nothing.
        """
        if seed is None:
            self._episode += 1
            seed = self._master_seed + self._episode
        else:
            self._episode = 0

        self._step_index = 0
        self._simulator = self._build(seed)
        self._scopes = build_scopes(self._simulator.state, self._slices)
        return self._observe(reward=0.0, components={}, terminated=False)

    def step(self, actions: NDArray[np.float32]) -> MarlStep:
        """Apply one joint action.

        ``actions`` has shape (agents, action_size), each entry a share in [0, 1]. Values are clipped
        rather than rejected: this is the training path, where an unsquashed policy output is a normal
        occurrence and an exception would end the run.
        """
        self._simulator.step(self._decode(np.clip(actions, 0.0, 1.0)))
        self._step_index += 1
        components = compute_reward(self._simulator.state, self._weights, self._slices)
        return self._observe(
            reward=total_reward(components),
            components=components,
            terminated=self._step_index >= self._episode_length,
        )

    def _decode(self, actions: NDArray[np.float32]) -> AllocationAction:
        """Assemble the joint action from per-agent shares.

        An agent's row indexes into *its own* flow list. Cross-agent leakage here would silently undo the
        decomposition — the agents would all be acting on the same flows and coordination would be an
        artefact of the decoder rather than of the policies.
        """
        shares: dict[str, float] = {}
        for agent_index, scope in enumerate(self._scopes):
            for slot, flow_id in enumerate(scope.flow_ids[: self._max_flows]):
                shares[flow_id] = float(actions[agent_index, slot])
        # Any flow beyond an agent's slot budget keeps a neutral full share rather than vanishing: an
        # omitted flow would be allocated nothing, which is a decision the policy did not make.
        for scope in self._scopes:
            for flow_id in scope.flow_ids[self._max_flows :]:
                shares.setdefault(flow_id, 1.0)
        return AllocationAction(share_by_flow_id=shares, issued_by="marl")

    def _observe(self, reward: float, components: dict[str, float], terminated: bool) -> MarlStep:
        state = self._simulator.state
        local = np.stack(
            [
                encode_local_observation(state, scope, self._max_flows, self._max_links, self._slices)
                for scope in self._scopes
            ]
        )
        return MarlStep(
            local_observations=local.astype(np.float32),
            joint_observation=encode_joint_observation(state),
            reward=reward,
            reward_components=components,
            terminated=terminated,
        )

    def _build(self, seed: int) -> NetworkSimulator:
        return NetworkSimulator.from_config(
            self._topology,
            self._traffic_pattern,
            seed=seed,
            flow_count=self._flow_count,
            domain=self._domain,
        )
