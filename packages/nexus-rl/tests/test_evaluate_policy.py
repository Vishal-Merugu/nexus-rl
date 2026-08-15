"""Evaluating a trained run actually evaluates the trained policy.

Regression: ``nexus-eval --run-id <id>`` validated the checkpoint's MDP version, logged the run's
metadata, and then compared the three baselines against one another. The policy was never rolled out and
never appeared in the table — so the project's central claim, that the learned controller beats the
non-learned floor, could not be produced by the project's own evaluation tool.

It also ignored the run's recorded ``topology``, ``flow_count`` and ``evaluation_patterns``, scoring a
policy on a network and a demand distribution its config never mentioned.

No model is trained here (see CLAUDE.md); a stub predictor stands in for the checkpoint so the wiring is
tested without the training cost.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from nexus_errors.config_error import ConfigurationError
from nexus_mdp.action_space import AllocationMode, decode_discrete_action
from nexus_rl.evaluate import (
    _read_action_mode,
    _read_patterns,
    _require_int,
    _require_str,
    evaluate_baseline,
    evaluate_policy,
)
from nexus_rl.policy_loader import routing_from
from nexus_types.action import AllocationAction
from nexus_types.network_state import NetworkState

_SEEDS = (0, 1, 2)


def _shed_background(observation: NDArray[np.float32], state: NetworkState) -> AllocationAction:
    del observation
    return decode_discrete_action(AllocationMode.SHED_BACKGROUND, state, routing_from(state))


def _throttle(observation: NDArray[np.float32], state: NetworkState) -> AllocationAction:
    del observation
    return decode_discrete_action(AllocationMode.THROTTLE_AGGRESSIVE, state, routing_from(state))


def test_a_policy_produces_a_comparison_row() -> None:
    row = evaluate_policy(_shed_background, "policy:test", "steady", _SEEDS, steps=48)

    assert row.label == "policy:test"
    assert row.traffic_pattern == "steady"
    assert row.seeds == len(_SEEDS)


def test_policy_and_baseline_rows_are_comparable() -> None:
    # Same simulator construction, same seeds, same steps — the rows differ only in the allocator.
    policy_row = evaluate_policy(_throttle, "policy:test", "steady", _SEEDS, steps=48)
    baseline_row = evaluate_baseline("static", "steady", _SEEDS, steps=48)

    assert policy_row.seeds == baseline_row.seeds
    # Throttling everything to 30% must carry less than the static baseline's full-rate split.
    assert policy_row.throughput_mbps[0] < baseline_row.throughput_mbps[0]


def test_the_policy_is_actually_consulted() -> None:
    # Two different policies must produce different numbers. If the predictor were ignored — the
    # original bug — these would be identical.
    shed = evaluate_policy(_shed_background, "shed", "steady", _SEEDS, steps=48)
    throttled = evaluate_policy(_throttle, "throttle", "steady", _SEEDS, steps=48)

    assert shed.throughput_mbps[0] != throttled.throughput_mbps[0]


def test_run_metadata_is_read_rather_than_defaulted() -> None:
    metadata: dict[str, object] = {
        "topology": "mesh-6",
        "flow_count": 8,
        "algorithm": "dqn",
        "action_mode": "discrete",
        "evaluation_patterns": ["steady", "bursty"],
    }

    assert _require_str(metadata, "topology", "r") == "mesh-6"
    assert _require_int(metadata, "flow_count", "r") == 8
    assert _read_action_mode(metadata, "r") == "discrete"
    assert _read_patterns(metadata) == ("steady", "bursty")


@pytest.mark.parametrize(
    ("metadata", "key"),
    [
        ({}, "topology"),
        ({"topology": 6}, "topology"),
    ],
)
def test_missing_metadata_raises_rather_than_silently_defaulting(
    metadata: dict[str, object], key: str
) -> None:
    # Falling back to "mesh-6" would evaluate the policy on a network it never trained on and report
    # the number as though it were comparable.
    with pytest.raises(ConfigurationError):
        _require_str(metadata, key, "run-x")


def test_a_boolean_flow_count_is_not_an_integer() -> None:
    # bool is a subclass of int; `flow_count: true` must not silently become 1 flow.
    with pytest.raises(ConfigurationError):
        _require_int({"flow_count": True}, "flow_count", "run-x")


def test_an_unknown_action_mode_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown action_mode"):
        _read_action_mode({"action_mode": "hybrid"}, "run-x")


def test_absent_patterns_fall_back_to_the_caller_default() -> None:
    assert _read_patterns({}) is None
    assert _read_patterns({"evaluation_patterns": []}) is None
