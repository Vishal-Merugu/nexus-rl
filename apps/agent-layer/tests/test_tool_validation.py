"""Tool-boundary enforcement tests.

Regression: the ``*Input`` models were only used to *generate* JSON schema, never to validate. Every
declared bound was decorative, and a missing required field escaped as a bare KeyError. Bounds on a tool a
language model can call are an authority question, not hygiene.
"""

from __future__ import annotations

import httpx
import pytest

from nexus_agent.client import OrchestratorClient
from nexus_agent.server import build_tool_specs
from nexus_agent.tools import read_tools, write_tools
from nexus_errors.control_error import ToolExecutionError
from nexus_types.constants import MAX_CONSTRAINT_TTL

_SPECS = {spec.name: spec for spec in build_tool_specs()}


def _client() -> OrchestratorClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"timestep": 10, "links": [], "flows": []})

    return OrchestratorClient(
        "http://sim", "http://ctrl", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


@pytest.mark.parametrize(
    ("tool", "arguments", "why"),
    [
        ("avoid_links", {"link_ids": [f"l{i}" for i in range(500)]}, "declared max_length is 32"),
        ("avoid_links", {"link_ids": []}, "declared min_length is 1"),
        ("get_recent_history", {"limit": -50}, "declared range is 1..200"),
        ("get_recent_history", {"limit": 10_000}, "declared range is 1..200"),
        ("get_recent_decisions", {"limit": 0}, "declared minimum is 1"),
        ("prioritise_traffic_class", {"traffic_class": "NOT_A_CLASS"}, "not a TrafficClass"),
        (
            "prioritise_traffic_class",
            {"traffic_class": "BULK", "ttl_timesteps": MAX_CONSTRAINT_TTL + 1},
            "exceeds MAX_CONSTRAINT_TTL",
        ),
        ("prioritise_traffic_class", {"traffic_class": "BULK", "ttl_timesteps": 0}, "ttl must be > 0"),
        ("set_objective_weights", {"latency": -1.0}, "weights are non-negative"),
        ("set_objective_weights", {"latency": 10_000.0}, "exceeds MAX_OBJECTIVE_WEIGHT"),
        ("revoke_constraint", {"constraint_id": ""}, "declared min_length is 1"),
    ],
)
def test_declared_bounds_are_enforced(tool: str, arguments: dict[str, object], why: str) -> None:
    with pytest.raises(ToolExecutionError, match="Invalid arguments"):
        _SPECS[tool].validate_arguments(arguments)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("get_flow_detail", {}),
        ("avoid_links", {}),
        ("revoke_constraint", {}),
        ("prioritise_traffic_class", {}),
    ],
)
def test_a_missing_required_argument_is_a_tool_error(tool: str, arguments: dict[str, object]) -> None:
    # Previously a bare KeyError, which escaped the tool-error path and surfaced as an unhandled crash.
    with pytest.raises(ToolExecutionError, match="Invalid arguments"):
        _SPECS[tool].validate_arguments(arguments)


def test_validation_fills_declared_defaults() -> None:
    filled = _SPECS["get_recent_history"].validate_arguments({})

    assert isinstance(filled, read_tools.GetRecentHistoryInput)
    assert filled.limit == 20


def test_valid_arguments_pass_through() -> None:
    filled = _SPECS["avoid_links"].validate_arguments({"link_ids": ["n0->n1"], "ttl_timesteps": 50})

    assert isinstance(filled, write_tools.AvoidLinksInput)
    assert filled.link_ids == ["n0->n1"]
    assert filled.ttl_timesteps == 50


def test_the_error_tells_the_model_not_to_fake_success() -> None:
    with pytest.raises(ToolExecutionError, match="do not answer as though it had succeeded"):
        _SPECS["get_flow_detail"].validate_arguments({})
