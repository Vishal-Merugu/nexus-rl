"""Centralized Training with Decentralized Execution — the contract, not the algorithm.

CTDE is the dominant paradigm in cooperative multi-agent RL: during training a centralized critic sees
global state, but the policy each agent *deploys* consumes only its own local observation. That gives
coordinated behaviour without requiring agents to talk to each other at execution time — which is exactly
the shape real distributed network orchestration has.

This module owns the part that has to be right before any algorithm is chosen: **who observes what**.
Getting the partition and the local encoding wrong makes every MAPPO result meaningless, and no amount of
training fixes it.

Three commitments, each closing a way the decomposition can quietly become fake:

- **One agent per slice.** Membership is derived from the flow's traffic class, so an agent's scope is
  determined by the state rather than agreed by configuration between services.
- **A local observation is genuinely local.** It contains the agent's own flows and the links they
  traverse — not the full state with other agents' entries zeroed, which is the easy mistake that leaves
  the observation globally-shaped and the "decentralization" nominal.
- **The joint observation is for the critic only.** It is built here so training can use it, and it is
  structurally separate so nothing can accidentally hand it to an execution-time policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nexus_mdp.observation import encode_observation
from nexus_sim.constants import EPSILON, MAX_LINK_LATENCY_MS
from nexus_sim.slicing import slice_attainment
from nexus_types.network_state import FlowState, NetworkState
from nexus_types.slice import DEFAULT_SLICES, SliceSet
from nexus_types.traffic_class import TrafficClass

LOCAL_FLOW_FEATURES = 4
"""Per owned flow: satisfaction, normalised demand, normalised latency, drop ratio."""

LOCAL_LINK_FEATURES = 2
"""Per traversed link: utilization, normalised queue."""

LOCAL_CONTEXT_FEATURES = 4
"""Slice-level context: SLA headroom, mean contention on owned links, own share of total demand, priority.

The contention term is what makes cooperation learnable at all. Without some signal about pressure the
agent does not itself create, each agent optimises its own slice in isolation and the joint policy is
three selfish policies in a trenchcoat.
"""

DEMAND_NORMALISER_MBPS = 200.0
MAX_QUEUE_MB = 50.0


@dataclass(frozen=True, slots=True)
class AgentScope:
    """Which flows and links one agent is responsible for."""

    agent_id: str
    slice_id: str
    traffic_class: TrafficClass
    priority: int
    flow_ids: tuple[str, ...]
    link_ids: tuple[str, ...]
    """Links the agent's flows traverse. Overlapping with other agents' — that overlap *is* the
    contention they must learn to resolve."""


def build_scopes(state: NetworkState, slices: SliceSet = DEFAULT_SLICES) -> tuple[AgentScope, ...]:
    """Partition the network into one agent per slice, from the state alone.

    Derived, never configured. Flow ids are positional (``f0``..``fn``), so two components computing this
    partition independently would produce ids that line up perfectly while referring to different flows —
    the failure `tasks/lessons.md` records under "two services must never derive shared state
    independently".
    """
    scopes: list[AgentScope] = []
    for index, network_slice in enumerate(slices.ordered):
        owned = [flow for flow in state.flows if flow.traffic_class is network_slice.traffic_class]
        links: list[str] = []
        for flow in owned:
            for link_id in flow.path:
                if link_id not in links:
                    links.append(link_id)
        scopes.append(
            AgentScope(
                agent_id=f"agent-{index}",
                slice_id=network_slice.slice_id,
                traffic_class=network_slice.traffic_class,
                priority=network_slice.priority,
                flow_ids=tuple(flow.flow_id for flow in owned),
                link_ids=tuple(links),
            )
        )
    return tuple(scopes)


def local_observation_size(max_flows: int, max_links: int) -> int:
    """Fixed local observation width.

    Fixed rather than ragged because a policy network needs a stable input shape, and because an agent
    whose observation grows with its flow count would silently change meaning between episodes.
    """
    return max_flows * LOCAL_FLOW_FEATURES + max_links * LOCAL_LINK_FEATURES + LOCAL_CONTEXT_FEATURES


def encode_local_observation(
    state: NetworkState,
    scope: AgentScope,
    max_flows: int,
    max_links: int,
    slices: SliceSet = DEFAULT_SLICES,
) -> NDArray[np.float32]:
    """Encode what one agent can actually see at execution time.

    **Nothing about other agents' flows enters this vector.** Only aggregate pressure on shared links
    does, which is information a real distributed controller would genuinely have — the link is carrying
    somebody's traffic and its utilization is observable. Handing an agent the other slices' per-flow
    detail would make the CTDE claim false while leaving every test passing.
    """
    observation = np.zeros(local_observation_size(max_flows, max_links), dtype=np.float32)

    owned_flows = [flow for flow in state.flows if flow.flow_id in set(scope.flow_ids)][:max_flows]
    for index, flow in enumerate(owned_flows):
        offset = index * LOCAL_FLOW_FEATURES
        observation[offset] = flow.satisfaction_ratio
        observation[offset + 1] = min(1.0, flow.demand_mbps / DEMAND_NORMALISER_MBPS)
        observation[offset + 2] = min(1.0, flow.latency_ms / MAX_LINK_LATENCY_MS)
        observation[offset + 3] = (
            min(1.0, flow.dropped_mbps / flow.demand_mbps) if flow.demand_mbps > EPSILON else 0.0
        )

    link_base = max_flows * LOCAL_FLOW_FEATURES
    owned_link_ids = set(scope.link_ids)
    owned_links = [link for link in state.links if link.link_id in owned_link_ids][:max_links]
    for index, link in enumerate(owned_links):
        offset = link_base + index * LOCAL_LINK_FEATURES
        observation[offset] = link.utilization
        observation[offset + 1] = min(1.0, link.queue_mb / MAX_QUEUE_MB)

    context_base = link_base + max_links * LOCAL_LINK_FEATURES
    network_slice = slices.by_id(scope.slice_id)
    observation[context_base] = _headroom(state, scope, slices)
    observation[context_base + 1] = (
        float(np.mean([link.utilization for link in owned_links])) if owned_links else 0.0
    )
    observation[context_base + 2] = _demand_share(state, owned_flows)
    observation[context_base + 3] = (network_slice.priority / 2.0) if network_slice else 0.0
    return observation


def encode_joint_observation(state: NetworkState) -> NDArray[np.float32]:
    """Global state for the centralized critic. **Training only.**

    Deliberately the existing single-agent encoding rather than a concatenation of local ones: the critic
    is supposed to see more than the sum of the agents' views, and a concatenation sees exactly that and
    no more.

    Nothing in a deployed policy may call this. That is the whole distinction CTDE rests on, and it is
    kept as a separate function precisely so handing it to an execution path is a visible act rather than
    a default argument nobody noticed.
    """
    return encode_observation(state)


def _headroom(state: NetworkState, scope: AgentScope, slices: SliceSet) -> float:
    network_slice = slices.by_id(scope.slice_id)
    if network_slice is None:
        return 0.0
    return slice_attainment(state, network_slice).headroom


def _demand_share(state: NetworkState, owned_flows: list[FlowState]) -> float:
    """The agent's share of total offered demand — how much of the problem is its own.

    Without it an agent cannot distinguish "the network is congested because of me" from "…because of
    everyone else", and those call for opposite behaviour.
    """
    total = state.total_demand_mbps
    if total <= EPSILON:
        return 0.0
    owned = sum(flow.demand_mbps for flow in owned_flows)
    return min(1.0, owned / total)
