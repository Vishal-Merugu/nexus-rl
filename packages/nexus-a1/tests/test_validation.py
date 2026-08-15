"""A1 policy validation — the negotiation surface.

The property under test throughout: an infeasible policy must be *refused with a reason*, never accepted
and quietly not honoured. An operator who is told a guarantee is in force, when it is not, has been
misled by the system rather than merely unlucky.
"""

from __future__ import annotations

import pytest

from nexus_a1.decision import A1PolicyStatus
from nexus_a1.policy import (
    A1Policy,
    A1PolicyStatement,
    A1PolicyType,
    QosObjective,
)
from nexus_a1.validation import validate_policy
from nexus_errors.control_error import ConstraintRejectedError
from nexus_types.constants import MAX_CONSTRAINT_TTL
from nexus_types.network_state import FlowState, LinkState, NetworkState
from nexus_types.traffic_class import TrafficClass


def _link(link_id: str, capacity: float = 100.0) -> LinkState:
    source, target = link_id.split("->")
    return LinkState(
        link_id=link_id,
        source=source,
        target=target,
        capacity_mbps=capacity,
        effective_capacity_mbps=capacity,
        allocated_mbps=10.0,
        queue_mb=0.0,
        utilization=0.1,
        latency_ms=5.0,
    )


def _flow(flow_id: str, candidate_paths: tuple[tuple[str, ...], ...]) -> FlowState:
    return FlowState(
        flow_id=flow_id,
        source="n0",
        target="n2",
        traffic_class=TrafficClass.BULK,
        demand_mbps=50.0,
        allocated_mbps=40.0,
        dropped_mbps=0.0,
        path=candidate_paths[0],
        candidate_paths=candidate_paths,
        latency_ms=10.0,
    )


def _state() -> NetworkState:
    return NetworkState(
        timestep=10,
        links=(_link("n0->n1"), _link("n1->n2"), _link("n0->n2")),
        flows=(
            _flow("f0", (("n0->n1", "n1->n2"), ("n0->n2",))),
            _flow("f1", (("n0->n1", "n1->n2"),)),
        ),
    )


def _policy(
    policy_type: A1PolicyType,
    statement: A1PolicyStatement,
    *,
    ttl: int = 100,
    issued_at: int = 10,
    schema_version: int = 1,
) -> A1Policy:
    return A1Policy(
        policy_id="p-1",
        policy_type=policy_type,
        schema_version=schema_version,
        statement=statement,
        issued_by="rapp:agent-layer",
        issued_at_timestep=issued_at,
        ttl_timesteps=ttl,
    )


# --- schema version negotiation -------------------------------------------------------------------


def test_a_newer_schema_is_refused_rather_than_guessed_at() -> None:
    policy = _policy(
        A1PolicyType.QOS_OBJECTIVE,
        A1PolicyStatement(qos_objective=QosObjective(latency=3.0)),
        schema_version=99,
    )

    decision = validate_policy(policy, _state(), current_timestep=10)

    assert decision.status is A1PolicyStatus.REJECTED
    assert "schema version 99" in decision.reason


# --- traffic steering -----------------------------------------------------------------------------


def test_steering_onto_real_links_is_accepted() -> None:
    policy = _policy(A1PolicyType.TRAFFIC_STEERING, A1PolicyStatement(avoid_link_ids=("n0->n1",)))

    decision = validate_policy(policy, _state(), current_timestep=10)

    assert decision.status is A1PolicyStatus.PARTIALLY_ACCEPTED
    # f1 has only one candidate path and it uses n0->n1, so the restriction cannot hold for it.
    assert "f1" in decision.rejected_elements
    assert decision.enforced_elements == ("n0->n1",)


def test_steering_onto_links_that_do_not_exist_is_rejected_with_the_real_ones() -> None:
    policy = _policy(A1PolicyType.TRAFFIC_STEERING, A1PolicyStatement(avoid_link_ids=("n9->n9",)))

    decision = validate_policy(policy, _state(), current_timestep=10)

    assert decision.status is A1PolicyStatus.REJECTED
    assert decision.achievable_alternative is not None
    assert "n0->n1" in decision.achievable_alternative


def test_a_partly_unknown_link_set_is_partially_accepted_not_accepted() -> None:
    # Collapsing this into ACCEPTED would let "avoid these two, one of which is fictional" report as
    # complete success.
    policy = _policy(
        A1PolicyType.TRAFFIC_STEERING,
        A1PolicyStatement(avoid_link_ids=("n0->n2", "n9->n9")),
    )

    decision = validate_policy(policy, _state(), current_timestep=10)

    assert decision.status is A1PolicyStatus.PARTIALLY_ACCEPTED
    assert decision.enforced_elements == ("n0->n2",)
    assert "n9->n9" in decision.rejected_elements


def test_a_flow_with_no_alternative_path_is_reported_not_black_holed() -> None:
    policy = _policy(
        A1PolicyType.TRAFFIC_STEERING,
        A1PolicyStatement(avoid_link_ids=("n0->n1", "n1->n2", "n0->n2")),
    )

    decision = validate_policy(policy, _state(), current_timestep=10)

    assert decision.status is A1PolicyStatus.PARTIALLY_ACCEPTED
    assert "f0" in decision.rejected_elements
    assert "f1" in decision.rejected_elements
    assert decision.achievable_alternative is not None
    assert "black-hole" in decision.achievable_alternative


# --- class floors ---------------------------------------------------------------------------------


def test_a_floor_beyond_total_capacity_is_rejected_with_what_is_achievable() -> None:
    # Total effective capacity is 300 Mb/s across three 100 Mb/s links.
    policy = _policy(
        A1PolicyType.CLASS_FLOOR,
        A1PolicyStatement(class_floor_mbps={TrafficClass.LATENCY_SENSITIVE: 5_000.0}),
    )

    decision = validate_policy(policy, _state(), current_timestep=10)

    assert decision.status is A1PolicyStatus.REJECTED
    assert "300" in decision.reason
    assert decision.achievable_alternative is not None


def test_a_feasible_floor_is_accepted() -> None:
    policy = _policy(
        A1PolicyType.CLASS_FLOOR,
        A1PolicyStatement(class_floor_mbps={TrafficClass.LATENCY_SENSITIVE: 50.0}),
    )

    decision = validate_policy(policy, _state(), current_timestep=10)

    assert decision.status is A1PolicyStatus.ACCEPTED
    assert decision.enforced_elements == ("LATENCY_SENSITIVE",)


def test_a_mixed_feasible_and_infeasible_floor_set_is_partially_accepted() -> None:
    policy = _policy(
        A1PolicyType.CLASS_FLOOR,
        A1PolicyStatement(
            class_floor_mbps={
                TrafficClass.LATENCY_SENSITIVE: 50.0,
                TrafficClass.BULK: 9_000.0,
            }
        ),
    )

    decision = validate_policy(policy, _state(), current_timestep=10)

    assert decision.status is A1PolicyStatus.PARTIALLY_ACCEPTED
    assert decision.enforced_elements == ("LATENCY_SENSITIVE",)
    assert decision.rejected_elements == ("BULK",)


# --- timing ---------------------------------------------------------------------------------------


def test_an_unbounded_ttl_is_refused() -> None:
    with pytest.raises(ValueError):  # noqa: PT011 — pydantic bound, message is pydantic's own
        _policy(
            A1PolicyType.QOS_OBJECTIVE,
            A1PolicyStatement(qos_objective=QosObjective()),
            ttl=MAX_CONSTRAINT_TTL + 1,
        )


def test_a_policy_stamped_in_the_future_is_rejected() -> None:
    policy = _policy(
        A1PolicyType.QOS_OBJECTIVE,
        A1PolicyStatement(qos_objective=QosObjective(latency=2.0)),
        issued_at=500,
    )

    decision = validate_policy(policy, _state(), current_timestep=10)

    assert decision.status is A1PolicyStatus.REJECTED
    assert "unpredictable" in decision.reason


def test_one_timestep_of_slack_is_allowed_for_the_read_then_post_race() -> None:
    policy = _policy(
        A1PolicyType.QOS_OBJECTIVE,
        A1PolicyStatement(qos_objective=QosObjective(latency=2.0)),
        issued_at=11,
    )

    decision = validate_policy(policy, _state(), current_timestep=10)

    assert decision.status is A1PolicyStatus.ACCEPTED


# --- objectives -----------------------------------------------------------------------------------


def test_reweighting_is_accepted_and_says_it_guarantees_nothing() -> None:
    policy = _policy(
        A1PolicyType.QOS_OBJECTIVE,
        A1PolicyStatement(qos_objective=QosObjective(latency=3.0, throughput=1.0)),
    )

    decision = validate_policy(policy, _state(), current_timestep=10)

    assert decision.status is A1PolicyStatus.ACCEPTED
    assert "does not guarantee" in decision.reason


def test_an_unsupported_policy_type_is_refused_with_what_is_supported() -> None:
    policy = _policy(
        A1PolicyType.CLASS_FLOOR,
        A1PolicyStatement(class_floor_mbps={TrafficClass.BULK: 10.0}),
    )

    decision = validate_policy(
        policy,
        _state(),
        current_timestep=10,
        supported_types=frozenset({A1PolicyType.QOS_OBJECTIVE}),
    )

    assert decision.status is A1PolicyStatus.REJECTED
    assert "QOS_OBJECTIVE" in decision.reason


# --- no-op guard ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy_type", "statement"),
    [
        (A1PolicyType.CLASS_FLOOR, A1PolicyStatement()),
        (A1PolicyType.TRAFFIC_STEERING, A1PolicyStatement()),
        (A1PolicyType.QOS_OBJECTIVE, A1PolicyStatement()),
    ],
)
def test_a_policy_whose_statement_is_empty_is_refused_at_construction(
    policy_type: A1PolicyType, statement: A1PolicyStatement
) -> None:
    # A well-formed policy that enforces nothing is the empty-success failure one layer up: the operator
    # is told their instruction is in force while the network ignores it.
    with pytest.raises(ConstraintRejectedError):
        _policy(policy_type, statement)
