"""Domain randomization, and the determinism it must not cost.

The whole value of this project's simulator is "same seed, same config ⇒ identical trajectory". Varying
the environment across episodes is supposed to make the *policy* robust, not make the *runs*
irreproducible. Most of these tests are about that distinction.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_errors.config_error import ConfigurationError
from nexus_sim.randomization import (
    HELD_OUT_DOMAINS,
    IDENTITY_DOMAIN,
    RandomizationRanges,
    SampledDomain,
    apply_domain,
    sample_domain,
)
from nexus_sim.simulator import NetworkSimulator
from nexus_sim.topology import build_topology


def test_the_default_is_the_nominal_network_so_randomization_is_opt_in() -> None:
    # Every recorded result predates this feature. A mode that could silently be on would invalidate them.
    nominal = NetworkSimulator.from_config("mesh-6", "steady", seed=0, flow_count=6)
    explicit = NetworkSimulator.from_config("mesh-6", "steady", seed=0, flow_count=6, domain=IDENTITY_DOMAIN)

    assert nominal.state.model_dump() == explicit.state.model_dump()


def test_sampling_is_reproducible_from_a_generator() -> None:
    first = sample_domain(np.random.default_rng(42), RandomizationRanges())
    second = sample_domain(np.random.default_rng(42), RandomizationRanges())

    assert first == second


def test_consecutive_draws_differ_so_episodes_are_not_identical() -> None:
    rng = np.random.default_rng(0)

    draws = [sample_domain(rng, RandomizationRanges()) for _ in range(5)]

    assert len({draw.capacity_scale for draw in draws}) == 5


def test_a_randomized_run_still_replays_exactly() -> None:
    # The property that must survive: vary the environment, keep the trajectory reproducible.
    domain = sample_domain(np.random.default_rng(7), RandomizationRanges())

    left = NetworkSimulator.from_config("mesh-6", "diurnal", seed=3, flow_count=6, domain=domain)
    right = NetworkSimulator.from_config("mesh-6", "diurnal", seed=3, flow_count=6, domain=domain)
    for _ in range(20):
        left.step(None)
        right.step(None)

    assert left.state.model_dump() == right.state.model_dump()


def test_a_different_domain_produces_a_different_network() -> None:
    # Otherwise randomization is decorative.
    nominal = NetworkSimulator.from_config("mesh-6", "steady", seed=1, flow_count=6)
    scaled = NetworkSimulator.from_config(
        "mesh-6", "steady", seed=1, flow_count=6, domain=SampledDomain(0.5, 1.0, 1.0)
    )

    assert nominal.state.links[0].capacity_mbps != scaled.state.links[0].capacity_mbps


def test_the_flow_set_does_not_move_with_the_capacity_draw() -> None:
    # Randomizing which flows exist would change the problem rather than the conditions, and would make
    # two episodes incomparable for reasons that have nothing to do with robustness.
    nominal = NetworkSimulator.from_config("mesh-6", "steady", seed=1, flow_count=6)
    scaled = NetworkSimulator.from_config(
        "mesh-6", "steady", seed=1, flow_count=6, domain=SampledDomain(0.5, 1.0, 1.0)
    )

    assert [f.flow_id for f in nominal.state.flows] == [f.flow_id for f in scaled.state.flows]
    assert [f.candidate_paths for f in nominal.state.flows] == [f.candidate_paths for f in scaled.state.flows]


def test_applying_a_domain_never_mutates_the_nominal_topology() -> None:
    # Compounding scale on scale turns a bounded range into a random walk.
    topology = build_topology("mesh-6")
    before = [link.capacity_mbps for link in topology.links]

    apply_domain(topology, SampledDomain(0.5, 1.0, 1.0))
    apply_domain(topology, SampledDomain(0.5, 1.0, 1.0))

    assert [link.capacity_mbps for link in topology.links] == before


def test_demand_scaling_reaches_the_flows() -> None:
    low = NetworkSimulator.from_config(
        "mesh-6", "steady", seed=1, flow_count=6, domain=SampledDomain(1.0, 0.5, 1.0)
    )
    high = NetworkSimulator.from_config(
        "mesh-6", "steady", seed=1, flow_count=6, domain=SampledDomain(1.0, 2.0, 1.0)
    )

    assert sum(f.demand_mbps for f in high.state.flows) > sum(f.demand_mbps for f in low.state.flows)


def test_latency_scaling_reaches_the_links() -> None:
    slow = NetworkSimulator.from_config(
        "mesh-6", "steady", seed=1, flow_count=6, domain=SampledDomain(1.0, 1.0, 3.0)
    )
    fast = NetworkSimulator.from_config("mesh-6", "steady", seed=1, flow_count=6)
    for _ in range(5):
        slow.step(None)
        fast.step(None)

    assert max(link.latency_ms for link in slow.state.links) > max(
        link.latency_ms for link in fast.state.links
    )


def test_physical_invariants_survive_randomization() -> None:
    # RL will find every invariant that is not enforced, and a randomized capacity is exactly the sort of
    # change that quietly breaks one.
    for domain in HELD_OUT_DOMAINS:
        simulator = NetworkSimulator.from_config("mesh-6", "flash_crowd", seed=2, flow_count=8, domain=domain)
        for _ in range(30):
            simulator.step(None)
        for link in simulator.state.links:
            assert 0.0 <= link.utilization <= 1.0
            assert link.queue_mb >= 0.0
        for flow in simulator.state.flows:
            assert flow.allocated_mbps >= 0.0
            assert flow.dropped_mbps >= 0.0


def test_held_out_domains_sit_outside_the_default_training_ranges() -> None:
    # A "held-out" set drawn from the training distribution would be a harder-sounding test of the same
    # thing. This is what makes Use Case 9 evidence rather than decoration.
    ranges = RandomizationRanges()

    for domain in HELD_OUT_DOMAINS:
        outside = (
            not ranges.capacity[0] <= domain.capacity_scale <= ranges.capacity[1]
            or not ranges.demand[0] <= domain.demand_scale <= ranges.demand[1]
            or not ranges.latency[0] <= domain.latency_scale <= ranges.latency[1]
        )
        assert outside, f"{domain} is inside the training distribution"


@pytest.mark.parametrize("bad", [(0.0, 1.0), (-1.0, 1.0), (1.5, 0.5)])
def test_a_degenerate_range_is_refused_at_construction(bad: tuple[float, float]) -> None:
    with pytest.raises(ConfigurationError):
        RandomizationRanges(capacity=bad)


def test_identity_ranges_read_as_disabled() -> None:
    assert not RandomizationRanges(capacity=(1.0, 1.0), demand=(1.0, 1.0), latency=(1.0, 1.0)).is_enabled
    assert RandomizationRanges().is_enabled
