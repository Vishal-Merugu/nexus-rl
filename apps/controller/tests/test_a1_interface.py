"""The controller as an A1 policy consumer (Near-RT RIC xApp).

Tests the *sequence*, not the pieces: offer-before-state, offer-then-enforce, offer-then-read-back,
duplicate ids, and expiry across advancing time. Per `tasks/lessons.md`, component tests in isolation are
exactly what missed the read-path-mutation and dying-loop bugs in v1, so the A1 surface gets sequence
coverage from the start.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import nexus_controller.main as controller_main
from nexus_a1.policy import A1PolicyType
from nexus_controller.a1_store import A1PolicyStore
from nexus_controller.constraint_store import ConstraintStore
from nexus_sim.simulator import NetworkSimulator


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(controller_main.app) as test_client:
        yield test_client
    # Module-level singletons persist between tests; reset them so ordering cannot matter.
    controller_main._constraints = ConstraintStore()
    controller_main._a1 = A1PolicyStore(controller_main._constraints)
    controller_main._decisions.clear()
    controller_main._last_timestep = 0
    controller_main._last_state = None


def _state() -> dict[str, object]:
    simulator = NetworkSimulator.from_config("mesh-6", "steady", seed=0, flow_count=8)
    return simulator.state.model_dump(mode="json")


def _advance(client: TestClient) -> dict[str, object]:
    """Drive one control step so the controller has observed the network."""
    state = _state()
    client.post("/act", json={"state": state})
    return state


def _policy(
    policy_id: str = "p-1",
    policy_type: A1PolicyType = A1PolicyType.QOS_OBJECTIVE,
    statement: dict[str, object] | None = None,
    ttl: int = 200,
    issued_at: int = 0,
) -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "policy_type": policy_type.value,
        "schema_version": 1,
        "statement": statement or {"qos_objective": {"latency": 3.0}},
        "issued_by": "rapp:agent-layer",
        "operator_utterance": "keep interactive traffic fast",
        "issued_at_timestep": issued_at,
        "ttl_timesteps": ttl,
    }


def test_the_xapp_advertises_its_policy_types(client: TestClient) -> None:
    response = client.get("/a1/policytypes")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert "QOS_OBJECTIVE" in body["policy_type_ids"]


def test_a_policy_offered_before_any_state_is_refused_not_accepted_blind(client: TestClient) -> None:
    # Accepting an unvalidated policy would report a guarantee whose feasibility was never checked.
    response = client.put("/a1/policies/p-1", json=_policy())

    assert response.status_code == 200
    decision = response.json()
    assert decision["status"] == "REJECTED"
    assert "not yet received any network state" in decision["reason"]


def test_an_accepted_policy_becomes_an_enforced_constraint(client: TestClient) -> None:
    _advance(client)

    response = client.put("/a1/policies/p-1", json=_policy())

    assert response.json()["status"] == "ACCEPTED"
    # A1 is the interface; the constraint is the enforcement. The control path is unchanged.
    constraints = client.get("/constraints").json()
    assert [c["constraint_id"] for c in constraints["constraints"]] == ["p-1"]


def test_the_operator_utterance_survives_into_enforcement(client: TestClient) -> None:
    _advance(client)
    client.put("/a1/policies/p-1", json=_policy())

    constraints = client.get("/constraints").json()

    assert constraints["constraints"][0]["operator_utterance"] == "keep interactive traffic fast"


def test_an_infeasible_floor_is_rejected_and_enforces_nothing(client: TestClient) -> None:
    _advance(client)

    response = client.put(
        "/a1/policies/p-2",
        json=_policy(
            "p-2",
            A1PolicyType.CLASS_FLOOR,
            {"class_floor_mbps": {"LATENCY_SENSITIVE": 9_000_000.0}},
        ),
    )

    decision = response.json()
    assert decision["status"] == "REJECTED"
    assert decision["achievable_alternative"] is not None
    assert client.get("/constraints").json()["constraints"] == []


def test_steering_onto_a_nonexistent_link_is_rejected(client: TestClient) -> None:
    _advance(client)

    response = client.put(
        "/a1/policies/p-3",
        json=_policy("p-3", A1PolicyType.TRAFFIC_STEERING, {"avoid_link_ids": ["n99->n98"]}),
    )

    assert response.json()["status"] == "REJECTED"


def test_a_decision_is_readable_after_the_fact(client: TestClient) -> None:
    _advance(client)
    client.put("/a1/policies/p-1", json=_policy())

    response = client.get("/a1/policies/p-1/status")

    assert response.status_code == 200
    assert response.json()["policy_id"] == "p-1"


def test_a_reused_policy_id_is_refused_rather_than_treated_as_an_update(client: TestClient) -> None:
    # ConstraintStore.remove deletes every match, so a reused id would make a later deletion ambiguous.
    _advance(client)
    client.put("/a1/policies/p-1", json=_policy())

    second = client.put("/a1/policies/p-1", json=_policy())

    assert second.json()["status"] == "REJECTED"
    assert "already in force" in second.json()["reason"]


def test_a_mismatched_path_and_document_id_is_a_client_error(client: TestClient) -> None:
    _advance(client)

    response = client.put("/a1/policies/p-other", json=_policy("p-1"))

    assert response.status_code == 400


def test_a_withdrawn_policy_stops_being_enforced(client: TestClient) -> None:
    _advance(client)
    client.put("/a1/policies/p-1", json=_policy())

    removed = client.delete("/a1/policies/p-1")

    assert removed.json() == {"removed": True}
    assert client.get("/constraints").json()["constraints"] == []
    assert client.get("/a1/policies").json() == []


def test_reading_policies_does_not_delete_them(client: TestClient) -> None:
    # The v1 equivalent of this bug pruned on a read path and a liveness probe wiped the store.
    _advance(client)
    client.put("/a1/policies/p-1", json=_policy())

    client.get("/a1/policies")
    client.get("/a1/policies", params={"timestep": 0})
    client.get("/health")

    assert len(client.get("/a1/policies").json()) == 1


def test_a_policy_expires_on_the_controllers_own_clock(client: TestClient) -> None:
    _advance(client)
    client.put("/a1/policies/p-1", json=_policy(ttl=3, issued_at=0))
    assert len(client.get("/a1/policies").json()) == 1

    # Advance simulation time past the policy's lifetime via the control path.
    state = _state()
    for timestep in range(1, 6):
        state["timestep"] = timestep
        client.post("/act", json={"state": state})

    assert client.get("/a1/policies").json() == []
    assert client.get("/constraints").json()["constraints"] == []


def test_there_is_no_a1_route_that_accepts_an_allocation(client: TestClient) -> None:
    # The central claim of ADR-0002, restated at the A1 boundary: the Non-RT tier issues guidance and
    # never a control action. Enforced by the absence of a route, not by a check inside one.
    paths = {route.path for route in controller_main.app.routes if hasattr(route, "path")}
    a1_paths = {p for p in paths if p.startswith("/a1")}

    assert a1_paths == {
        "/a1/policytypes",
        "/a1/policies",
        "/a1/policies/{policy_id}",
        "/a1/policies/{policy_id}/status",
    }
