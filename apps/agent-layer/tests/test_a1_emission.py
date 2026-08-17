"""The rApp issues A1 policies, and relays the xApp's decision.

This is the Non-RT side of ADR-0008. Stage H built the A1 contract and the xApp's consumption of it, but
the agent layer kept writing ``Constraint`` documents straight to ``POST /constraints`` — so the
negotiation semantics that justify the whole track (partial acceptance, refusal with an achievable
alternative) were unreachable from an operator.

Two properties carry the weight here:

1. **No write tool touches the constraint endpoint.** A1 is the interface; bypassing it is the regression
   this file exists to catch.
2. **A refusal is relayed, not swallowed and not raised.** REJECTED arrives as a 200 carrying the xApp's
   reason. Raising would make the model treat a considered "no" as a broken call and retry it; returning a
   cheerful payload would let it report a guarantee that is not in force.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

import httpx
import pytest

from nexus_agent.client import OrchestratorClient
from nexus_agent.server import build_tool_specs
from nexus_agent.tools import read_tools, write_tools
from nexus_errors.control_error import ToolExecutionError
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
    "flows": [],
    "active_incident_ids": [],
}

_ACCEPTED = {
    "policy_id": "p-abc",
    "status": "ACCEPTED",
    "reason": "Steering new traffic away from n0->n1. Expires at timestep 92.",
    "enforced_elements": ["n0->n1"],
    "rejected_elements": [],
    "achievable_alternative": None,
}

_PARTIAL = {
    "policy_id": "p-abc",
    "status": "PARTIALLY_ACCEPTED",
    "reason": "Steering applied to n0->n1; unknown links ignored: n9->n8.",
    "enforced_elements": ["n0->n1"],
    "rejected_elements": ["n9->n8"],
    "achievable_alternative": None,
}

_REJECTED = {
    "policy_id": "p-abc",
    "status": "REJECTED",
    "reason": "A floor of 900 Mb/s exceeds the network's total capacity of 100 Mb/s.",
    "enforced_elements": [],
    "rejected_elements": ["BULK"],
    "achievable_alternative": "Reduce those floors below 100 Mb/s.",
}


type Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> OrchestratorClient:
    return OrchestratorClient(
        simulator_url="http://sim",
        controller_url="http://ctrl",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _handler_returning(decision: Mapping[str, object]) -> Handler:
    """A controller that answers every policy offer with one fixed decision."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/state":
            return httpx.Response(200, json=_STATE)
        if path == "/metrics":
            return httpx.Response(200, json={"p95_latency_ms": 6.0})
        if path == "/a1/policytypes":
            return httpx.Response(
                200,
                json={
                    "schema_version": 1,
                    "policy_type_ids": ["CLASS_FLOOR", "QOS_OBJECTIVE", "TRAFFIC_STEERING"],
                },
            )
        if path == "/a1/policies":
            return httpx.Response(200, json=[])
        if path.startswith("/a1/policies/"):
            if request.method == "PUT":
                return httpx.Response(200, json=dict(decision))
            if request.method == "DELETE":
                return httpx.Response(200, json={"removed": True})
        return httpx.Response(404, json={"detail": f"unexpected {request.method} {path}"})

    return handler


# --- The interface is A1, not the constraint endpoint -----------------------------------------------


@pytest.mark.anyio
async def test_no_write_tool_touches_the_constraint_endpoint() -> None:
    # ADR-0008: A1 is the interface the agent layer speaks. Writing a Constraint directly bypasses
    # validation, the decision, and every negotiation semantic that justifies the track.
    seen: list[str] = []

    def watch(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return _handler_returning(_ACCEPTED)(request)

    client = _client(watch)
    await write_tools.avoid_links(client, link_ids=["n0->n1"])
    await write_tools.set_objective_weights(client, latency=4.0)
    await write_tools.prioritise_traffic_class(client, traffic_class=TrafficClass.LATENCY_SENSITIVE)
    await write_tools.set_class_floor(client, traffic_class=TrafficClass.BULK, floor_mbps=10.0)

    assert not [call for call in seen if "/constraints" in call], seen
    assert [call for call in seen if call.startswith("PUT /a1/policies/")]


@pytest.mark.anyio
async def test_the_policy_carries_rapp_provenance_and_the_simulator_timestep() -> None:
    captured: dict[str, object] = {}

    def capture(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            captured.update(json.loads(request.content))
        return _handler_returning(_ACCEPTED)(request)

    await write_tools.avoid_links(
        _client(capture), link_ids=["n0->n1"], ttl_timesteps=50, operator_utterance="drain n0->n1"
    )

    assert captured["issued_at_timestep"] == 42
    assert captured["issued_by"] == "rapp:agent-layer"
    assert captured["operator_utterance"] == "drain n0->n1"
    assert captured["policy_type"] == "TRAFFIC_STEERING"
    assert captured["schema_version"] == 1


@pytest.mark.anyio
async def test_the_document_id_matches_the_path_the_xapp_is_offered() -> None:
    # The xApp 400s a mismatch. Building the URL from anything but the document's own id is how that
    # happens.
    paths: list[str] = []
    bodies: list[dict[str, object]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            paths.append(request.url.path)
            bodies.append(json.loads(request.content))
        return _handler_returning(_ACCEPTED)(request)

    await write_tools.set_objective_weights(_client(capture), fairness=2.0)

    assert paths[0] == f"/a1/policies/{bodies[0]['policy_id']}"


# --- The decision is relayed ------------------------------------------------------------------------


@pytest.mark.anyio
async def test_an_accepted_policy_reports_what_was_enforced_and_when_it_expires() -> None:
    result = await write_tools.avoid_links(
        _client(_handler_returning(_ACCEPTED)), link_ids=["n0->n1"], ttl_timesteps=50
    )

    assert result["applied"] is True
    assert result["status"] == "ACCEPTED"
    assert result["expires_at_timestep"] == 42 + 50
    assert result["enforced_elements"] == ["n0->n1"]


@pytest.mark.anyio
async def test_a_partial_acceptance_is_not_reported_as_success() -> None:
    # "Avoid these two links, one of which does not exist" must never read as a completed instruction.
    result = await write_tools.avoid_links(
        _client(_handler_returning(_PARTIAL)), link_ids=["n0->n1", "n9->n8"]
    )

    assert result["status"] == "PARTIALLY_ACCEPTED"
    assert result["rejected_elements"] == ["n9->n8"]
    assert result["enforced_elements"] == ["n0->n1"]


@pytest.mark.anyio
async def test_a_refusal_is_returned_not_raised_and_says_nothing_was_applied() -> None:
    # A refusal is a 200 and part of the protocol. Raising would have the model retry a considered "no".
    result = await write_tools.set_class_floor(
        _client(_handler_returning(_REJECTED)), traffic_class=TrafficClass.BULK, floor_mbps=900.0
    )

    assert result["applied"] is False
    assert result["status"] == "REJECTED"
    assert result["expires_at_timestep"] is None
    assert "exceeds the network's total capacity" in str(result["reason"])
    assert result["achievable_alternative"] == "Reduce those floors below 100 Mb/s."


@pytest.mark.anyio
async def test_the_reason_is_the_xapps_own_words_verbatim() -> None:
    # The agent relays; it does not paraphrase. A paraphrased refusal is where overclaiming returns.
    result = await write_tools.set_class_floor(
        _client(_handler_returning(_REJECTED)), traffic_class=TrafficClass.BULK, floor_mbps=900.0
    )

    assert result["reason"] == _REJECTED["reason"]


# --- Failure paths ----------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_a_decision_the_rapp_cannot_parse_is_a_failed_call() -> None:
    # Distinct from a refusal: the xApp said something this rApp does not understand. Reporting it as a
    # decision would invent an outcome.
    def garbled(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            return httpx.Response(200, json={"status": "MAYBE"})
        return _handler_returning(_ACCEPTED)(request)

    with pytest.raises(ToolExecutionError, match="decision"):
        await write_tools.avoid_links(_client(garbled), link_ids=["n0->n1"])


@pytest.mark.anyio
async def test_a_policy_cannot_be_stamped_when_the_timestep_is_unreadable() -> None:
    def bad_state(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/state":
            return httpx.Response(200, json={"links": [], "flows": []})
        return _handler_returning(_ACCEPTED)(request)

    with pytest.raises(ToolExecutionError, match="timestep"):
        await write_tools.avoid_links(_client(bad_state), link_ids=["n0->n1"])


@pytest.mark.anyio
async def test_an_unreachable_controller_raises_rather_than_reporting_a_decision() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            return httpx.Response(503, json={"detail": "xApp down"})
        return _handler_returning(_ACCEPTED)(request)

    with pytest.raises(ToolExecutionError):
        await write_tools.avoid_links(_client(down), link_ids=["n0->n1"])


# --- Withdrawal -------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_withdrawing_a_policy_goes_through_a1_and_confirms_removal() -> None:
    seen: list[str] = []

    def watch(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return _handler_returning(_ACCEPTED)(request)

    result = await write_tools.withdraw_policy(_client(watch), policy_id="p-abc")

    assert result["withdrawn"] == "p-abc"
    assert "DELETE /a1/policies/p-abc" in seen


@pytest.mark.anyio
async def test_withdrawing_a_policy_that_is_not_in_force_raises() -> None:
    # Otherwise the model reports having lifted guidance that was never there.
    def gone(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(200, json={"removed": False})
        return _handler_returning(_ACCEPTED)(request)

    with pytest.raises(ToolExecutionError, match="already expired"):
        await write_tools.withdraw_policy(_client(gone), policy_id="p-nope")


# --- Reading the policy surface ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_active_policies_are_readable_with_the_types_the_xapp_advertises() -> None:
    # The rApp should choose an achievable policy rather than discovering the limit by rejection.
    result = await read_tools.get_active_policies(_client(_handler_returning(_ACCEPTED)))

    assert result["timestep"] == 42
    assert result["policies"] == []
    assert result["advertised_policy_types"] == ["CLASS_FLOOR", "QOS_OBJECTIVE", "TRAFFIC_STEERING"]
    assert result["a1_schema_version"] == 1


# --- Authority --------------------------------------------------------------------------------------


def test_the_write_surface_is_exactly_the_five_a1_tools() -> None:
    write_names = {spec.name for spec in build_tool_specs() if spec.writes}

    assert write_names == {
        "prioritise_traffic_class",
        "set_objective_weights",
        "avoid_links",
        "set_class_floor",
        "withdraw_policy",
    }


def test_every_issuing_tool_maps_to_an_advertised_policy_type() -> None:
    # A tool that issues a type the xApp does not advertise is refused on every call — an authority
    # surface that looks larger than it is.
    from nexus_a1.validation import SUPPORTED_POLICY_TYPES

    assert set(write_tools.TOOL_POLICY_TYPES.values()) <= SUPPORTED_POLICY_TYPES
    assert {spec.name for spec in build_tool_specs() if spec.writes} - {"withdraw_policy"} == set(
        write_tools.TOOL_POLICY_TYPES
    )
