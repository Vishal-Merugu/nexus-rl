"""The supervisor in the path, speaking the controller's own contract.

The deployment story for this service is one line of configuration: point the simulator's
`NEXUS_CONTROLLER_URL` at the supervisor instead. That only works if `/act` is genuinely
interchangeable — same request shape, same response shape, same failure behaviour — and a mock of the
controller would agree with whatever it was written to agree with.

So these drive the real `nexus_supervisor` app over ASGI into the real `nexus_controller` app. The import
of the controller is test-only: the two meet over HTTP and nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

import nexus_controller.main as controller_main
import nexus_supervisor.main as supervisor_main
from nexus_sim.simulator import NetworkSimulator
from nexus_supervisor.supervisor import SUPERVISOR_ISSUER, SafetySupervisor
from nexus_types.action import AllocationAction


def _state() -> dict[str, object]:
    simulator = NetworkSimulator.from_config("mesh-6", "steady", seed=0, flow_count=8)
    return simulator.state.model_dump(mode="json")


@pytest.fixture
def wired() -> Iterator[TestClient]:
    """A supervisor whose downstream controller is the real controller app."""
    with TestClient(controller_main.app) as controller:

        def route(request: httpx.Request) -> httpx.Response:
            forwarded = controller.request(
                request.method,
                str(request.url.copy_with(scheme="http", host="testserver", port=None)),
                content=request.content,
                headers=dict(request.headers),
            )
            return httpx.Response(forwarded.status_code, content=forwarded.content)

        with TestClient(supervisor_main.app) as supervisor:
            supervisor_main._client = httpx.AsyncClient(transport=httpx.MockTransport(route))
            supervisor_main._controller_url = "http://ctrl"
            supervisor_main._supervisor = SafetySupervisor()
            yield supervisor

    controller_main._decisions.clear()
    controller_main._last_timestep = 0
    controller_main._last_state = None


@pytest.fixture
def isolated() -> Iterator[TestClient]:
    """A supervisor whose downstream controller is unreachable."""
    with TestClient(supervisor_main.app) as supervisor:

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("controller down", request=request)

        supervisor_main._client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
        supervisor_main._controller_url = "http://ctrl"
        supervisor_main._supervisor = SafetySupervisor()
        yield supervisor


def test_act_is_interchangeable_with_the_controllers(wired: TestClient) -> None:
    # The deployment story in one assertion: the same request, the same response type.
    response = wired.post("/act", json={"state": _state()})

    assert response.status_code == 200
    action = AllocationAction.model_validate(response.json())
    assert action.share_by_flow_id


def test_a_clean_action_passes_through_with_the_controllers_provenance(wired: TestClient) -> None:
    response = wired.post("/act", json={"state": _state()})

    assert AllocationAction.model_validate(response.json()).issued_by != SUPERVISOR_ISSUER


def test_an_unreachable_controller_still_yields_an_action(isolated: TestClient) -> None:
    # There is no path where the simulator receives nothing. That path made the network run unmanaged at
    # full demand, which is what this service exists to replace.
    response = isolated.post("/act", json={"state": _state()})

    assert response.status_code == 200
    action = AllocationAction.model_validate(response.json())
    assert action.issued_by == SUPERVISOR_ISSUER
    assert action.share_by_flow_id


def test_health_leads_with_who_is_driving(isolated: TestClient) -> None:
    isolated.post("/act", json={"state": _state()})

    body = isolated.get("/health").json()

    assert body["status"] == "ok"
    assert body["fallback_engaged"] is True
    assert body["fallback_reason"] == "controller_unreachable"
    assert body["activation_count"] == 1


def test_safety_is_separate_from_health(wired: TestClient) -> None:
    # A liveness probe asks "is this process up". Policy observability asks "is the policy behaving",
    # and the second question is the one nobody thinks to instrument.
    wired.post("/act", json={"state": _state()})

    body = wired.get("/safety").json()

    assert body["supervised_timesteps"] == 1
    assert "violations_by_kind" in body
    assert "drift" in body
    assert "correction_rate" in body


def test_a_diverged_controller_response_is_diagnosed_rather_than_parsed_away() -> None:
    """A bare ``NaN`` on the wire is caught as a policy fault, not as a transport error.

    The design point this protects: ``AllocationAction``'s own validator rejects non-finite shares, so
    parsing the controller's response into one would turn a diverged policy into "malformed response" and
    lose the diagnosis. The supervisor parses leniently precisely so it can name the fault.

    The body is written by hand because httpx refuses to *serialise* NaN — but `json.loads` accepts a bare
    `NaN` literal, and that asymmetry is exactly how such a value reaches a consumer in the wild.
    """
    with TestClient(supervisor_main.app) as supervisor:

        def diverged(request: httpx.Request) -> httpx.Response:
            flows = json.loads(request.content)["state"]["flows"]
            shares = ", ".join(f'"{flow["flow_id"]}": NaN' for flow in flows)
            return httpx.Response(
                200,
                content=f'{{"share_by_flow_id": {{{shares}}}, "issued_by": "controller"}}'.encode(),
                headers={"content-type": "application/json"},
            )

        supervisor_main._client = httpx.AsyncClient(transport=httpx.MockTransport(diverged))
        supervisor_main._controller_url = "http://ctrl"
        supervisor_main._supervisor = SafetySupervisor()

        response = supervisor.post("/act", json={"state": _state()})

        assert response.status_code == 200
        assert AllocationAction.model_validate(response.json()).issued_by == SUPERVISOR_ISSUER

        safety = supervisor.get("/safety").json()
        assert safety["violations_by_kind"]["non_finite_share"] > 0
        assert safety["reason"] == "unsafe_action"


def test_an_unparseable_response_falls_back_as_unreachable_not_as_a_bad_policy() -> None:
    # Distinguishing these matters: one says retrain, the other says fix the deployment.
    with TestClient(supervisor_main.app) as supervisor:

        def garbage(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(200, content=b"not json at all")

        supervisor_main._client = httpx.AsyncClient(transport=httpx.MockTransport(garbage))
        supervisor_main._controller_url = "http://ctrl"
        supervisor_main._supervisor = SafetySupervisor()

        supervisor.post("/act", json={"state": _state()})

        assert supervisor.get("/health").json()["fallback_reason"] == "controller_unreachable"


# --- Shadow mode (Use Case 10) ----------------------------------------------------------------------


def _shadow_client(candidate_share: float) -> httpx.MockTransport:
    """A candidate controller that allocates a fixed, very distinctive share to every flow."""

    def route(request: httpx.Request) -> httpx.Response:
        flows = json.loads(request.content)["state"]["flows"]
        issuer = "shadow-candidate" if request.url.host == "shadow" else "controller"
        share = candidate_share if request.url.host == "shadow" else 0.5
        return httpx.Response(
            200,
            json={
                "share_by_flow_id": {flow["flow_id"]: share for flow in flows},
                "issued_by": issuer,
            },
        )

    return httpx.MockTransport(route)


def test_a_shadow_candidates_action_never_reaches_live_traffic() -> None:
    """The guarantee shadow mode rests on.

    The candidate here allocates a share nothing else would produce, so if it ever leaked into the applied
    action it would be unmistakable. Enforced by the absence of a code path from the shadow proposal to
    the returned action — not by a flag.
    """
    with TestClient(supervisor_main.app) as supervisor:
        supervisor_main._client = httpx.AsyncClient(transport=_shadow_client(0.123456))
        supervisor_main._controller_url = "http://ctrl"
        supervisor_main._shadow_url = "http://shadow"
        supervisor_main._supervisor = SafetySupervisor()

        applied = AllocationAction.model_validate(supervisor.post("/act", json={"state": _state()}).json())

        assert 0.123456 not in set(applied.share_by_flow_id.values())
        assert all(share == 0.5 for share in applied.share_by_flow_id.values())
        assert applied.issued_by == "controller"

        supervisor_main._shadow_url = ""


def test_shadow_divergence_is_recorded_while_the_incumbent_drives() -> None:
    with TestClient(supervisor_main.app) as supervisor:
        supervisor_main._client = httpx.AsyncClient(transport=_shadow_client(0.9))
        supervisor_main._controller_url = "http://ctrl"
        supervisor_main._shadow_url = "http://shadow"
        supervisor_main._supervisor = SafetySupervisor()

        for _ in range(3):
            supervisor.post("/act", json={"state": _state()})
        body = supervisor.get("/shadow").json()

        assert body["shadow_configured"] is True
        assert body["samples"] == 3
        assert body["mean_divergence"] > 0.0
        # Too few timesteps to say anything, and it says so rather than guessing.
        assert body["verdict"] == "insufficient_data"

        supervisor_main._shadow_url = ""


def test_shadow_reports_as_unconfigured_when_it_is_off() -> None:
    # Off is the normal state: shadow mode doubles the calls in the fast loop, so it is opt-in.
    with TestClient(supervisor_main.app) as supervisor:
        supervisor_main._shadow_url = ""

        assert supervisor.get("/shadow").json()["shadow_configured"] is False
