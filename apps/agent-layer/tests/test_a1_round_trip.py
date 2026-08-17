"""The rApp and the xApp, wired together with no mock in between.

Every other test in this directory hands the agent layer a hand-written controller response. That proves
the rApp behaves correctly given a decision — it cannot prove the rApp and the xApp agree on the *wire*:
the URL shape, the field names, the enum spellings, the auth header. A mock agrees with whatever it was
written to agree with.

`tasks/lessons.md` is explicit about this failure mode ("two services must never derive shared state
independently"; "test service sequences, not just components"), and A1 is precisely a contract between two
independently deployed tiers. So these tests drive the real `nexus_agent` write tools over ASGI into the
real `nexus_controller` app, and assert on what the network actually ends up enforcing.

The import of `nexus_controller` here is **test-only and deliberate**. The agent layer does not, and must
not, depend on the controller package at runtime — the two tiers meet over HTTP and nothing else. It is
declared in neither service's dependencies; both resolve from the workspace venv, the same way the
controller's own tests import `nexus_sim`.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

import nexus_controller.main as controller_main
from nexus_agent.client import OrchestratorClient
from nexus_agent.tools import read_tools, write_tools
from nexus_controller.a1_store import A1PolicyStore
from nexus_controller.constraint_store import ConstraintStore
from nexus_sim.simulator import NetworkSimulator
from nexus_types.traffic_class import TrafficClass


@pytest.fixture
def wired() -> Iterator[OrchestratorClient]:
    """An agent-layer client whose controller is the real controller app.

    The simulator half stays a stub: these tests are about the A1 boundary, and the simulator only has to
    supply a timestep and the state the xApp validates against.
    """
    simulator = NetworkSimulator.from_config("mesh-6", "steady", seed=0, flow_count=8)
    state = simulator.state.model_dump(mode="json")

    with TestClient(controller_main.app) as controller:
        # One control step, so the xApp has observed a network to validate policies against. Without it
        # every policy is refused as unvalidatable — which is itself covered in the controller's tests.
        controller.post("/act", json={"state": state})

        def route(request: httpx.Request) -> httpx.Response:
            if request.url.host == "sim":
                if request.url.path == "/state":
                    return httpx.Response(200, json=state)
                if request.url.path == "/metrics":
                    return httpx.Response(200, json={})
                return httpx.Response(404, json={"detail": request.url.path})
            forwarded = controller.request(
                request.method,
                str(request.url.copy_with(scheme="http", host="testserver", port=None)),
                content=request.content,
                headers=dict(request.headers),
            )
            return httpx.Response(forwarded.status_code, content=forwarded.content)

        yield OrchestratorClient(
            simulator_url="http://sim",
            controller_url="http://ctrl",
            client=httpx.AsyncClient(transport=httpx.MockTransport(route)),
        )

    controller_main._constraints = ConstraintStore()
    controller_main._a1 = A1PolicyStore(controller_main._constraints)
    controller_main._decisions.clear()
    controller_main._last_timestep = 0
    controller_main._last_state = None


def _rows(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    """Narrow a tool result's list-of-objects field, or fail the test where the shape broke.

    Tool results are ``dict[str, object]`` on purpose — they cross into a language model, and pretending
    otherwise is how unchecked shapes get in. Narrowing here keeps the assertions honest without a
    ``cast`` that would hide a real contract change behind a passing test.
    """
    value = payload[key]
    assert isinstance(value, list), f"{key} was {type(value).__name__}, not a list"
    for item in value:
        assert isinstance(item, dict), f"{key} contained a {type(item).__name__}, not an object"
    return [dict(item) for item in value]


def _link_id() -> str:
    simulator = NetworkSimulator.from_config("mesh-6", "steady", seed=0, flow_count=8)
    return simulator.state.links[0].link_id


def _total_capacity_mbps() -> float:
    simulator = NetworkSimulator.from_config("mesh-6", "steady", seed=0, flow_count=8)
    return sum(link.effective_capacity_mbps for link in simulator.state.links)


@pytest.mark.anyio
async def test_an_objective_policy_travels_end_to_end_and_becomes_enforcement(
    wired: OrchestratorClient,
) -> None:
    # The whole point of the tier: an operator instruction reaches the thing that allocates bandwidth.
    result = await write_tools.prioritise_traffic_class(
        wired,
        traffic_class=TrafficClass.LATENCY_SENSITIVE,
        ttl_timesteps=100,
        operator_utterance="keep interactive traffic fast",
    )

    assert result["status"] == "ACCEPTED"
    assert result["applied"] is True

    # A1 is the interface; the constraint is the enforcement. Both views must show it.
    policies = _rows(await read_tools.get_active_policies(wired), "policies")
    constraints = _rows(await read_tools.get_active_constraints(wired), "constraints")
    assert [p["policy_id"] for p in policies] == [result["policy_id"]]
    assert constraints


@pytest.mark.anyio
async def test_the_rapp_and_xapp_agree_on_every_advertised_policy_type(
    wired: OrchestratorClient,
) -> None:
    # A tool issuing a type the xApp does not implement would be refused on every call. This catches an
    # enum spelling drift between the two packages that no mock ever could.
    advertised = (await read_tools.get_active_policies(wired))["advertised_policy_types"]

    assert isinstance(advertised, list)
    assert {t.value for t in write_tools.TOOL_POLICY_TYPES.values()} <= set(advertised)


@pytest.mark.anyio
async def test_an_impossible_floor_is_refused_by_the_real_network_not_a_mock(
    wired: OrchestratorClient,
) -> None:
    # Use Case 8. The refusal has to come from the xApp measuring its own topology, otherwise the
    # "negotiation" is just the agent layer agreeing with itself.
    result = await write_tools.set_class_floor(
        wired,
        traffic_class=TrafficClass.BULK,
        floor_mbps=_total_capacity_mbps() * 10,
        operator_utterance="guarantee bulk 10x the network",
    )

    assert result["applied"] is False
    assert result["status"] == "REJECTED"
    assert result["achievable_alternative"] is not None

    # Nothing was enforced, and nothing was recorded as in force.
    policies = await read_tools.get_active_policies(wired)
    assert policies["policies"] == []


@pytest.mark.anyio
async def test_steering_onto_a_fictional_link_is_partially_accepted_with_both_lists(
    wired: OrchestratorClient,
) -> None:
    real = _link_id()

    result = await write_tools.avoid_links(
        wired, link_ids=[real, "n99->n98"], operator_utterance="drain for maintenance"
    )

    assert result["status"] == "PARTIALLY_ACCEPTED"
    assert result["enforced_elements"] == [real]
    assert result["rejected_elements"] == ["n99->n98"]


@pytest.mark.anyio
async def test_only_the_agreed_part_of_a_partial_policy_is_enforced(
    wired: OrchestratorClient,
) -> None:
    # Translating the request rather than the decision would apply guidance the operator was told had been
    # refused — worse than either accepting or rejecting outright.
    real = _link_id()
    await write_tools.avoid_links(wired, link_ids=[real, "n99->n98"])

    forbidden = [
        constraint["forbidden_link_ids"]
        for constraint in _rows(await read_tools.get_active_constraints(wired), "constraints")
    ]

    assert forbidden == [[real]]


@pytest.mark.anyio
async def test_a_withdrawn_policy_leaves_neither_a_policy_nor_a_constraint(
    wired: OrchestratorClient,
) -> None:
    # The two stores are separate. A withdrawal that clears one and not the other leaves the operator's
    # view and the network's behaviour disagreeing — with no error anywhere.
    issued = await write_tools.avoid_links(wired, link_ids=[_link_id()], ttl_timesteps=500)
    policy_id = issued["policy_id"]
    assert isinstance(policy_id, str)

    withdrawn = await write_tools.withdraw_policy(wired, policy_id=policy_id)

    assert withdrawn["withdrawn"] == policy_id
    assert (await read_tools.get_active_policies(wired))["policies"] == []
    assert (await read_tools.get_active_constraints(wired))["constraints"] == []
