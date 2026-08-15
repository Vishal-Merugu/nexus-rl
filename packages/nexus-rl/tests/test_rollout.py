from __future__ import annotations

import numpy as np
import pytest

from nexus_baselines.allocator import StaticAllocator, build_baseline
from nexus_errors.config_error import ConfigurationError
from nexus_mdp.action_space import AllocationMode, FlowRoutingOptions, decode_discrete_action
from nexus_rl.evaluate import MINIMUM_SEEDS, evaluate_baseline, render_comparison
from nexus_rl.rollout import rollout_allocator, rollout_policy
from nexus_sim.simulator import NetworkSimulator
from nexus_types.action import AllocationAction


def _simulator(pattern: str = "steady", seed: int = 1) -> NetworkSimulator:
    return NetworkSimulator.from_config("mesh-6", pattern, seed=seed)


def test_rollout_records_one_entry_per_step() -> None:
    simulator = _simulator()

    summary = rollout_allocator(simulator, StaticAllocator(30.0), steps=20, seed=1, traffic_pattern="steady")

    assert summary.steps == 20
    assert len(summary.records) == 20


def test_rollout_carries_its_seed_and_pattern_into_the_summary() -> None:
    summary = rollout_allocator(
        _simulator("bursty", seed=9),
        StaticAllocator(30.0),
        steps=5,
        seed=9,
        traffic_pattern="bursty",
    )

    assert summary.seed == 9
    assert summary.traffic_pattern == "bursty"


def test_every_named_reward_component_is_averaged() -> None:
    summary = rollout_allocator(
        _simulator(), StaticAllocator(30.0), steps=10, seed=1, traffic_pattern="steady"
    )

    assert set(summary.mean_reward_components) == {
        "throughput_reward",
        "latency_penalty",
        "drop_penalty",
        "fairness_bonus",
    }


def test_headline_reports_the_four_qos_metrics() -> None:
    summary = rollout_allocator(
        _simulator(), StaticAllocator(30.0), steps=10, seed=1, traffic_pattern="steady"
    )

    assert set(summary.headline()) == {"p95_latency_ms", "throughput_mbps", "fairness", "dropped_mb"}


def test_an_empty_rollout_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="at least one step"):
        rollout_allocator(_simulator(), StaticAllocator(30.0), steps=0, seed=1, traffic_pattern="steady")


def test_a_policy_and_a_baseline_run_through_the_same_path() -> None:
    # The comparison is only honest because both go through identical code, simulator and metrics.
    simulator = _simulator()
    routing = tuple(
        FlowRoutingOptions(flow_id=flow.flow_id, candidate_paths=flow.candidate_paths)
        for flow in simulator.flows
    )

    def always_open(observation: np.ndarray) -> AllocationAction:
        del observation
        return decode_discrete_action(AllocationMode.OPEN, simulator.state, routing)

    summary = rollout_policy(
        simulator, always_open, steps=15, seed=1, traffic_pattern="steady", label="fixed-policy"
    )

    assert summary.label == "fixed-policy"
    assert summary.steps == 15


def test_a_throttling_policy_carries_less_traffic_than_an_open_one() -> None:
    open_summary = rollout_allocator(
        _simulator(),
        build_baseline("static", _simulator().topology, 8),
        steps=30,
        seed=1,
        traffic_pattern="steady",
    )

    throttled = _simulator()
    routing = tuple(
        FlowRoutingOptions(flow_id=flow.flow_id, candidate_paths=flow.candidate_paths)
        for flow in throttled.flows
    )

    def aggressive(observation: np.ndarray) -> AllocationAction:
        del observation
        return decode_discrete_action(AllocationMode.THROTTLE_AGGRESSIVE, throttled.state, routing)

    throttled_summary = rollout_policy(throttled, aggressive, steps=30, seed=1, traffic_pattern="steady")

    assert throttled_summary.mean_throughput_mbps < open_summary.mean_throughput_mbps


def test_evaluation_aggregates_across_seeds_with_spread() -> None:
    row = evaluate_baseline("static", "diurnal", seeds=(0, 1, 2), steps=20)

    assert row.seeds == 3
    assert row.p95_latency_ms[1] >= 0.0  # spread reported, not just a mean


def test_comparison_table_names_the_simulation_limits() -> None:
    rows = [evaluate_baseline("static", "steady", seeds=(0, 1, 2), steps=10)]

    report = render_comparison(rows)

    assert "Flow-level simulation" in report
    assert "Not validated against physical hardware" in report


def test_a_single_seed_comparison_is_labelled_as_not_a_result() -> None:
    rows = [evaluate_baseline("static", "steady", seeds=(0,), steps=10)]

    report = render_comparison(rows)

    assert "Not a result" in report
    assert f"--seeds {MINIMUM_SEEDS}" in report


def test_a_sufficiently_seeded_comparison_carries_no_warning() -> None:
    rows = [evaluate_baseline("static", "steady", seeds=(0, 1, 2), steps=10)]

    assert "Not a result" not in render_comparison(rows)
