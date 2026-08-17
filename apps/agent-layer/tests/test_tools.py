"""Agent-layer tests.

Two things carry most of the weight here:

1. **The authority audit** — no tool may produce an allocation. This is the project's central design claim,
   and a test is the only thing that keeps it true as tools get added.
2. **Failure paths** — every tool raises rather than returning empty success. Given nothing, the model
   invents numbers that look exactly like real ones.
"""

from __future__ import annotations

import httpx
import pytest

from nexus_agent.client import OrchestratorClient
from nexus_agent.server import build_tool_specs
from nexus_agent.tools import read_tools, write_tools
from nexus_errors.control_error import ToolExecutionError
from nexus_types.constants import MAX_CONSTRAINT_TTL
from nexus_types.traffic_class import TrafficClass

_STATE = {
    "timestep": 42,
    "links": [
        {
            "link_id": "n0->n1",
            "source": "n0",
            "target": "n1",
            "capacity_mbps": 100.0,
            "effective_capacity_mbps": 100.0,
            "allocated_mbps": 80.0,
            "queue_mb": 2.0,
            "utilization": 0.8,
            "latency_ms": 6.0,
        }
    ],
    "flows": [
        {
            "flow_id": "f0",
            "source": "n0",
            "target": "n1",
            "traffic_class": "BULK",
            "demand_mbps": 100.0,
            "allocated_mbps": 80.0,
            "dropped_mbps": 0.0,
            "path": ["n0->n1"],
            "latency_ms": 6.0,
        }
    ],
    "active_incident_ids": [],
}


def _client(handler: object) -> OrchestratorClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return OrchestratorClient(
        simulator_url="http://sim",
        controller_url="http://ctrl",
        client=httpx.AsyncClient(transport=transport),
    )


def _ok_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/state":
        return httpx.Response(200, json=_STATE)
    if path == "/metrics":
        return httpx.Response(200, json={"p95_latency_ms": 6.0, "fairness_index": 0.9})
    if path == "/history":
        return httpx.Response(200, json=[{"timestep": 41, "metrics": {}}])
    if path == "/incidents":
        return httpx.Response(200, json=[])
    if path == "/decisions":
        return httpx.Response(200, json=[{"timestep": 41, "policy": "baseline:static", "shares": {}}])
    if path == "/health":
        return httpx.Response(200, json={"policy": "baseline:static", "trained_policy_loaded": False})
    if path == "/constraints":
        return httpx.Response(200, json={"timestep": 42, "constraints": []})
    if path == "/a1/policytypes":
        return httpx.Response(
            200, json={"schema_version": 1, "policy_type_ids": ["QOS_OBJECTIVE", "TRAFFIC_STEERING"]}
        )
    if path == "/a1/policies":
        return httpx.Response(200, json=[])
    if path.startswith("/a1/policies/"):
        if request.method == "DELETE":
            return httpx.Response(200, json={"removed": True})
        return httpx.Response(
            200,
            json={
                "policy_id": path.rsplit("/", 1)[-1],
                "status": "ACCEPTED",
                "reason": "Applied. Expires at timestep 342.",
                "enforced_elements": [],
                "rejected_elements": [],
                "achievable_alternative": None,
            },
        )
    return httpx.Response(404, json={"detail": "not found"})


def _failing_handler(request: httpx.Request) -> httpx.Response:
    del request
    return httpx.Response(500, json={"detail": "simulator exploded"})


# --- Authority audit -------------------------------------------------------------------------------


def test_no_write_tool_can_produce_an_allocation() -> None:
    # ADR-0002. If this fails, the project's central design claim is no longer true.
    forbidden_names = {"allocate", "set_bandwidth", "set_allocation", "set_share", "act"}

    for spec in build_tool_specs():
        assert spec.name not in forbidden_names


def test_every_write_tool_schema_demands_a_bounded_ttl() -> None:
    for spec in build_tool_specs():
        if not spec.writes or spec.name == "withdraw_policy":
            continue
        properties = spec.input_schema["properties"]
        assert "ttl_timesteps" in properties  # type: ignore[operator]
        assert properties["ttl_timesteps"]["maximum"] == MAX_CONSTRAINT_TTL  # type: ignore[index]


def test_every_tool_declares_a_description_for_the_model() -> None:
    for spec in build_tool_specs():
        assert spec.description.strip(), f"{spec.name} has no description"


# --- Read tools ------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_network_state_returns_structured_data_not_prose() -> None:
    result = await read_tools.get_network_state(_client(_ok_handler))

    assert result["state"]["timestep"] == 42  # type: ignore[index]
    assert "metrics" in result


@pytest.mark.anyio
async def test_flow_detail_includes_the_links_on_the_flow_path() -> None:
    result = await read_tools.get_flow_detail(_client(_ok_handler), flow_id="f0")

    flow = result["flow"]
    path_links = result["path_links"]
    assert isinstance(flow, dict)
    assert isinstance(path_links, list)
    assert flow["flow_id"] == "f0"
    assert [link["link_id"] for link in path_links] == ["n0->n1"]


@pytest.mark.anyio
async def test_asking_about_an_unknown_flow_raises_and_lists_the_real_ones() -> None:
    with pytest.raises(ToolExecutionError) as excinfo:
        await read_tools.get_flow_detail(_client(_ok_handler), flow_id="f99")

    assert excinfo.value.context["available"] == ["f0"]


# --- Failure paths ---------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_a_failed_read_raises_rather_than_returning_empty_success() -> None:
    with pytest.raises(ToolExecutionError):
        await read_tools.get_network_state(_client(_failing_handler))


@pytest.mark.anyio
async def test_a_tool_error_tells_the_model_not_to_answer_from_memory() -> None:
    with pytest.raises(ToolExecutionError, match="do not answer from memory"):
        await read_tools.get_recent_history(_client(_failing_handler))


@pytest.mark.anyio
async def test_a_failed_write_raises() -> None:
    with pytest.raises(ToolExecutionError):
        await write_tools.prioritise_traffic_class(
            _client(_failing_handler), traffic_class=TrafficClass.LATENCY_SENSITIVE
        )


@pytest.mark.anyio
async def test_an_unreachable_service_raises_rather_than_hanging_silently() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ToolExecutionError):
        await read_tools.get_network_state(_client(refuse))


# --- Write tools -----------------------------------------------------------------------------------
#
# The A1 emission path — provenance, timestep stamping, and how each decision status is relayed — is
# covered in test_a1_emission.py. What stays here is the tool-level contract shared with the read tools.


@pytest.mark.anyio
async def test_an_issued_policy_reports_its_expiry_in_timesteps() -> None:
    # "Never imply a change is permanent" is a property of the returned payload, not just the prompt.
    result = await write_tools.prioritise_traffic_class(
        _client(_ok_handler),
        traffic_class=TrafficClass.LATENCY_SENSITIVE,
        ttl_timesteps=300,
        operator_utterance="prioritise low-latency traffic for five minutes",
    )

    assert result["ttl_timesteps"] == 300
    assert result["expires_at_timestep"] == 42 + 300
