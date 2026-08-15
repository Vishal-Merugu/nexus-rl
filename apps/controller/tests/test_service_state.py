"""Service-level state-transition tests.

Every case here is a regression for a bug that the unit suite missed entirely. The gap was structural: the
old tests covered components in isolation and never exercised *sequences* of requests against a live
service, which is where read paths mutating state, defaults resolving to nonsense, and background tasks
dying all hide.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import nexus_controller.main as controller_main
from nexus_controller.constraint_store import ConstraintStore
from nexus_sim.simulator import NetworkSimulator

_LIVE_CONSTRAINT = {
    "constraint_id": "c-live",
    "issued_by": "agent-layer",
    "operator_utterance": "prioritise low-latency traffic",
    "issued_at_timestep": 500,
    "ttl_timesteps": 1000,
    "objective_weights": {"throughput": 1.0, "latency": 3.0, "drops": 1.0, "fairness": 0.5},
}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(controller_main.app) as test_client:
        yield test_client
    # Module-level singletons persist between tests; reset them so ordering cannot matter.
    controller_main._constraints = ConstraintStore()
    controller_main._decisions.clear()
    controller_main._last_timestep = 0


def _advance_to(client: TestClient, timestep: int) -> None:
    """Drive the controller's clock by issuing a decision, as the control loop would."""
    simulator = NetworkSimulator.from_config("mesh-6", "diurnal", seed=1337)
    state = simulator.state.model_dump(mode="json")
    state["timestep"] = timestep
    client.post("/act", json={"state": state})


def test_health_does_not_destroy_constraints(client: TestClient) -> None:
    # A liveness probe used to prune at the last-decision timestep, wiping every constraint issued after
    # it. In-cluster that meant operator instructions survived seconds, not minutes.
    client.post("/constraints", json=_LIVE_CONSTRAINT, params={"current_timestep": 500})

    for _ in range(5):
        client.get("/health")

    active = client.get("/constraints", params={"timestep": 600}).json()["constraints"]
    assert len(active) == 1


def test_health_reports_the_stored_constraint_count(client: TestClient) -> None:
    client.post("/constraints", json=_LIVE_CONSTRAINT, params={"current_timestep": 500})

    assert client.get("/health").json()["stored_constraints"] == 1


def test_a_parameterless_constraint_read_does_not_destroy_them(client: TestClient) -> None:
    # The default used to be timestep 0, which pruned everything issued later.
    client.post("/constraints", json=_LIVE_CONSTRAINT, params={"current_timestep": 500})

    client.get("/constraints")

    assert len(client.get("/constraints", params={"timestep": 600}).json()["constraints"]) == 1


def test_reading_an_earlier_timestep_does_not_destroy_later_constraints(client: TestClient) -> None:
    client.post("/constraints", json=_LIVE_CONSTRAINT, params={"current_timestep": 500})

    client.get("/constraints", params={"timestep": 0})

    assert len(client.get("/constraints", params={"timestep": 600}).json()["constraints"]) == 1


def test_constraints_default_to_the_controller_current_time(client: TestClient) -> None:
    _advance_to(client, 550)
    client.post("/constraints", json=_LIVE_CONSTRAINT)

    assert len(client.get("/constraints").json()["constraints"]) == 1


def test_expired_constraints_are_collected_on_the_control_path(client: TestClient) -> None:
    client.post("/constraints", json=_LIVE_CONSTRAINT, params={"current_timestep": 500})

    _advance_to(client, 5_000)  # well past issue + ttl

    assert client.get("/health").json()["stored_constraints"] == 0


def test_health_exposes_the_controller_clock(client: TestClient) -> None:
    _advance_to(client, 42)

    assert client.get("/health").json()["last_timestep"] == 42


def test_the_fallback_keeps_its_rates_and_provenance_under_a_constraint(client: TestClient) -> None:
    # Adding a routing constraint used to swap the static baseline for a full-open action and relabel it
    # as coming from the controller — silently removing all throttling and misreporting the source.
    simulator = NetworkSimulator.from_config("mesh-6", "diurnal", seed=1337)
    state = simulator.state.model_dump(mode="json")
    before = client.post("/act", json={"state": state}).json()

    client.post(
        "/constraints",
        json={
            "constraint_id": "c-avoid",
            "issued_by": "agent-layer",
            "issued_at_timestep": 0,
            "ttl_timesteps": 500,
            "forbidden_link_ids": [simulator.topology.links[0].link_id],
        },
        params={"current_timestep": 0},
    )
    after = client.post("/act", json={"state": state}).json()

    assert after["issued_by"] == before["issued_by"] == "baseline:static"
    assert after["share_by_flow_id"] == before["share_by_flow_id"]


def test_a_routing_constraint_moves_only_the_flows_it_affects(client: TestClient) -> None:
    simulator = NetworkSimulator.from_config("mesh-6", "diurnal", seed=1337)
    state = simulator.state.model_dump(mode="json")
    forbidden = simulator.topology.links[0].link_id
    affected = {flow.flow_id for flow in simulator.state.flows if forbidden in flow.path}

    client.post(
        "/constraints",
        json={
            "constraint_id": "c-narrow",
            "issued_by": "agent-layer",
            "issued_at_timestep": 0,
            "ttl_timesteps": 500,
            "forbidden_link_ids": [forbidden],
        },
        params={"current_timestep": 0},
    )
    action = client.post("/act", json={"state": state}).json()

    assert set(action["path_by_flow_id"]) == affected


def test_expired_constraints_are_not_served_to_clients(client: TestClient) -> None:
    # Making snapshot() non-mutating exposed a second issue: the serialised payload carried expired
    # constraints, because `.active()` is a method the JSON does not include. The agent layer would then
    # report a lapsed rule as still in force.
    client.post(
        "/constraints",
        json={
            "constraint_id": "c-short",
            "issued_by": "agent-layer",
            "issued_at_timestep": 0,
            "ttl_timesteps": 100,
            "objective_weights": {"throughput": 1.0, "latency": 3.0, "drops": 1.0, "fairness": 0.5},
        },
        params={"current_timestep": 0},
    )

    in_force = client.get("/constraints", params={"timestep": 50}).json()["constraints"]
    lapsed = client.get("/constraints", params={"timestep": 500}).json()["constraints"]

    assert [c["constraint_id"] for c in in_force] == ["c-short"]
    assert lapsed == []


def test_the_controller_holds_no_model_of_the_network(client: TestClient) -> None:
    """A network it was never configured for still decodes correctly.

    The controller used to rebuild the flow set from its own NEXUS_SEED/NEXUS_TOPOLOGY. Flow ids are
    positional (f0..fn), so a config mismatch produced ids that lined up perfectly while the routes behind
    them belonged to different flows — wrong paths, no error anywhere. Decoding from the received state
    makes that divergence structurally impossible.
    """
    # Deliberately nothing like the controller's configured topology/seed.
    foreign = NetworkSimulator.from_config("mesh-10", "bursty", seed=98765, flow_count=12)
    state = foreign.state.model_dump(mode="json")

    action = client.post("/act", json={"state": state}).json()

    assert set(action["share_by_flow_id"]) == {flow.flow_id for flow in foreign.state.flows}


def test_routing_constraints_resolve_against_the_received_topology(client: TestClient) -> None:
    foreign = NetworkSimulator.from_config("mesh-10", "steady", seed=4242)
    forbidden = foreign.topology.links[0].link_id
    affected = {f.flow_id for f in foreign.state.flows if forbidden in f.path}

    client.post(
        "/constraints",
        json={
            "constraint_id": "c-foreign",
            "issued_by": "agent-layer",
            "issued_at_timestep": 0,
            "ttl_timesteps": 500,
            "forbidden_link_ids": [forbidden],
        },
        params={"current_timestep": 0},
    )
    action = client.post("/act", json={"state": foreign.state.model_dump(mode="json")}).json()

    assert set(action["path_by_flow_id"]) == affected
    for path in action["path_by_flow_id"].values():
        assert all(link in {link_.link_id for link_ in foreign.topology.links} for link in path)
