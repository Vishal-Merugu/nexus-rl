"""The CTDE contract: who observes what.

This has to be right before any algorithm is chosen. A decomposition that looks multi-agent while every
agent secretly sees global state produces MAPPO numbers that mean nothing, and no amount of training
reveals the problem — every test still passes and the reward still improves.

So the load-bearing test here is the negative one: an agent's local observation must not change when
another slice's flows change.
"""

from __future__ import annotations

import numpy as np

from nexus_mdp.multi_agent import (
    LOCAL_CONTEXT_FEATURES,
    LOCAL_FLOW_FEATURES,
    LOCAL_LINK_FEATURES,
    build_scopes,
    encode_joint_observation,
    encode_local_observation,
    local_observation_size,
)
from nexus_sim.simulator import NetworkSimulator
from nexus_types.network_state import FlowState, LinkState, NetworkState
from nexus_types.slice import DEFAULT_SLICES
from nexus_types.traffic_class import TrafficClass

MAX_FLOWS, MAX_LINKS = 6, 10


def _sim(flow_count: int = 9) -> NetworkSimulator:
    return NetworkSimulator.from_config("mesh-6", "diurnal", seed=0, flow_count=flow_count)


def _state_with(flows: tuple[FlowState, ...]) -> NetworkState:
    return NetworkState(
        timestep=1,
        links=(
            LinkState(
                link_id="n0->n1",
                source="n0",
                target="n1",
                capacity_mbps=100.0,
                effective_capacity_mbps=100.0,
                allocated_mbps=50.0,
                queue_mb=1.0,
                utilization=0.5,
                latency_ms=5.0,
            ),
        ),
        flows=flows,
    )


def _flow(flow_id: str, traffic_class: TrafficClass, demand: float = 100.0) -> FlowState:
    return FlowState(
        flow_id=flow_id,
        source="n0",
        target="n1",
        traffic_class=traffic_class,
        demand_mbps=demand,
        allocated_mbps=demand * 0.8,
        dropped_mbps=0.0,
        path=("n0->n1",),
        candidate_paths=(("n0->n1",),),
        latency_ms=5.0,
    )


# --- The partition ----------------------------------------------------------------------------------


def test_there_is_one_agent_per_slice() -> None:
    scopes = build_scopes(_sim().state)

    assert len(scopes) == len(DEFAULT_SLICES.slices)
    assert [s.slice_id for s in scopes] == [s.slice_id for s in DEFAULT_SLICES.ordered]


def test_the_partition_is_derived_from_state_not_configuration() -> None:
    # Flow ids are positional, so two components computing this independently would line up perfectly
    # while referring to different flows.
    state = _sim().state
    scopes = build_scopes(state)

    for scope in scopes:
        owned = {f.flow_id for f in state.flows if f.traffic_class is scope.traffic_class}
        assert set(scope.flow_ids) == owned


def test_every_flow_belongs_to_exactly_one_agent() -> None:
    state = _sim().state
    scopes = build_scopes(state)

    owned = [flow_id for scope in scopes for flow_id in scope.flow_ids]

    assert sorted(owned) == sorted(f.flow_id for f in state.flows)
    assert len(owned) == len(set(owned))


def test_agents_share_links_because_that_is_the_contention() -> None:
    # Non-overlapping scopes would make coordination unnecessary and the multi-agent framing pointless.
    scopes = build_scopes(_sim(12).state)
    link_sets = [set(scope.link_ids) for scope in scopes if scope.link_ids]

    assert any(a & b for i, a in enumerate(link_sets) for b in link_sets[i + 1 :])


def test_a_slice_with_no_flows_still_gets_an_agent() -> None:
    # Otherwise the agent count changes with traffic and the policy set changes shape mid-episode.
    state = _state_with((_flow("f0", TrafficClass.BULK),))

    scopes = build_scopes(state)

    assert len(scopes) == 3
    assert any(scope.flow_ids == () for scope in scopes)


# --- Local observations are actually local ----------------------------------------------------------


def test_an_agents_observation_is_unchanged_by_another_slices_flows() -> None:
    """The load-bearing CTDE test.

    If this fails, "decentralized execution" is a claim the code does not support, and every multi-agent
    result built on it is really a single-agent result with extra steps.
    """
    premium = _flow("f0", TrafficClass.LATENCY_SENSITIVE)
    before = _state_with((premium, _flow("f1", TrafficClass.BULK, demand=50.0)))
    after = _state_with((premium, _flow("f1", TrafficClass.BULK, demand=190.0)))

    scope = next(s for s in build_scopes(before) if s.traffic_class is TrafficClass.LATENCY_SENSITIVE)
    left = encode_local_observation(before, scope, MAX_FLOWS, MAX_LINKS)
    right = encode_local_observation(
        after,
        next(s for s in build_scopes(after) if s.traffic_class is TrafficClass.LATENCY_SENSITIVE),
        MAX_FLOWS,
        MAX_LINKS,
    )

    # The demand-share context feature legitimately moves — total demand changed. Every per-flow and
    # per-link feature must not.
    flow_and_link_width = MAX_FLOWS * LOCAL_FLOW_FEATURES + MAX_LINKS * LOCAL_LINK_FEATURES
    assert np.array_equal(left[:flow_and_link_width], right[:flow_and_link_width])


def test_the_local_observation_has_a_fixed_width() -> None:
    # A policy network needs a stable input shape; a ragged one changes meaning between episodes.
    state = _sim().state
    scopes = build_scopes(state)

    widths = {encode_local_observation(state, scope, MAX_FLOWS, MAX_LINKS).shape[0] for scope in scopes}

    assert widths == {local_observation_size(MAX_FLOWS, MAX_LINKS)}


def test_the_observation_is_float32_because_sb3_assumes_it() -> None:
    state = _sim().state
    scope = build_scopes(state)[0]

    assert encode_local_observation(state, scope, MAX_FLOWS, MAX_LINKS).dtype == np.float32


def test_every_local_feature_is_normalised() -> None:
    # An unnormalised feature two orders larger than the rest makes the others invisible to the policy.
    simulator = _sim()
    for _ in range(40):
        simulator.step(None)

    for scope in build_scopes(simulator.state):
        observation = encode_local_observation(simulator.state, scope, MAX_FLOWS, MAX_LINKS)
        assert np.all(observation >= 0.0)
        assert np.all(observation <= 1.0)


def test_the_agent_can_see_contention_it_does_not_itself_cause() -> None:
    # Without this an agent cannot tell "congested because of me" from "because of everyone else", and
    # those call for opposite behaviour.
    state = _sim(12).state
    scope = build_scopes(state)[0]

    observation = encode_local_observation(state, scope, MAX_FLOWS, MAX_LINKS)
    context = observation[-LOCAL_CONTEXT_FEATURES:]

    assert context[1] > 0.0  # mean utilization on shared links
    assert 0.0 < context[2] < 1.0  # own share of total demand


def test_priority_is_visible_so_agents_can_differentiate() -> None:
    state = _sim().state
    scopes = build_scopes(state)

    priorities = [encode_local_observation(state, scope, MAX_FLOWS, MAX_LINKS)[-1] for scope in scopes]

    assert priorities == sorted(priorities, reverse=True)
    assert len(set(priorities)) == 3


def test_more_flows_than_the_cap_are_truncated_rather_than_overflowing() -> None:
    state = _sim(12).state
    scope = max(build_scopes(state), key=lambda s: len(s.flow_ids))

    observation = encode_local_observation(state, scope, max_flows=1, max_links=2)

    assert observation.shape[0] == local_observation_size(1, 2)


# --- The joint observation is the critic's alone ----------------------------------------------------


def test_the_joint_observation_is_wider_than_any_local_one() -> None:
    # The critic must see more than any agent does; that asymmetry is what CTDE is.
    state = _sim().state
    joint = encode_joint_observation(state)
    local = encode_local_observation(state, build_scopes(state)[0], MAX_FLOWS, MAX_LINKS)

    assert joint.shape[0] > local.shape[0]


def test_the_joint_observation_is_not_a_concatenation_of_local_ones() -> None:
    # A concatenation sees exactly the sum of the agents' views and no more, which defeats the point of
    # having a centralized critic at all.
    state = _sim().state
    joint = encode_joint_observation(state)
    concatenated = np.concatenate(
        [encode_local_observation(state, scope, MAX_FLOWS, MAX_LINKS) for scope in build_scopes(state)]
    )

    assert joint.shape[0] != concatenated.shape[0]
