"""Translating accepted A1 guidance into the constraint the controller enforces.

The property that matters: translation reflects the **decision**, not the request. A partially-accepted
policy must translate to enforcement covering only the part the xApp agreed to — otherwise the operator
is told a link was ignored while the controller quietly acts on it anyway.
"""

from __future__ import annotations

import pytest

from nexus_a1.decision import A1PolicyDecision, A1PolicyStatus
from nexus_a1.policy import A1Policy, A1PolicyStatement, A1PolicyType, QosObjective
from nexus_a1.translate import policy_to_constraint
from nexus_errors.control_error import ConstraintRejectedError
from nexus_types.traffic_class import TrafficClass


def _policy(policy_type: A1PolicyType, statement: A1PolicyStatement) -> A1Policy:
    return A1Policy(
        policy_id="p-42",
        policy_type=policy_type,
        statement=statement,
        issued_by="rapp:agent-layer",
        operator_utterance="keep the trading traffic fast for the next ten minutes",
        issued_at_timestep=5,
        ttl_timesteps=600,
    )


def _decision(
    status: A1PolicyStatus, enforced: tuple[str, ...] = (), rejected: tuple[str, ...] = ()
) -> A1PolicyDecision:
    return A1PolicyDecision(
        policy_id="p-42",
        status=status,
        reason="test",
        enforced_elements=enforced,
        rejected_elements=rejected,
    )


def test_a_rejected_policy_is_never_translated() -> None:
    policy = _policy(A1PolicyType.QOS_OBJECTIVE, A1PolicyStatement(qos_objective=QosObjective(latency=3.0)))

    with pytest.raises(ConstraintRejectedError, match="no enforcement"):
        policy_to_constraint(policy, _decision(A1PolicyStatus.REJECTED))


def test_provenance_and_the_operator_utterance_survive_translation() -> None:
    policy = _policy(A1PolicyType.QOS_OBJECTIVE, A1PolicyStatement(qos_objective=QosObjective(latency=3.0)))

    constraint = policy_to_constraint(policy, _decision(A1PolicyStatus.ACCEPTED))

    assert constraint.constraint_id == "p-42"
    assert constraint.issued_by == "rapp:agent-layer"
    assert constraint.operator_utterance.startswith("keep the trading traffic")
    assert constraint.ttl_timesteps == 600
    assert constraint.expires_at_timestep() == 605


def test_objective_weights_carry_across() -> None:
    policy = _policy(
        A1PolicyType.QOS_OBJECTIVE,
        A1PolicyStatement(qos_objective=QosObjective(throughput=2.0, latency=3.0, drops=1.0, fairness=0.5)),
    )

    constraint = policy_to_constraint(policy, _decision(A1PolicyStatus.ACCEPTED))

    assert constraint.objective_weights is not None
    assert constraint.objective_weights.latency == 3.0
    assert constraint.objective_weights.throughput == 2.0


def test_partial_steering_enforces_only_what_was_accepted() -> None:
    # The operator asked for three links; the xApp accepted one. Enforcing all three would apply
    # guidance the operator was explicitly told had been refused.
    policy = _policy(
        A1PolicyType.TRAFFIC_STEERING,
        A1PolicyStatement(avoid_link_ids=("n0->n1", "n9->n9", "n8->n8")),
    )

    constraint = policy_to_constraint(
        policy,
        _decision(
            A1PolicyStatus.PARTIALLY_ACCEPTED,
            enforced=("n0->n1",),
            rejected=("n9->n9", "n8->n8"),
        ),
    )

    assert constraint.forbidden_link_ids == ("n0->n1",)


def test_partial_floors_enforce_only_the_feasible_classes() -> None:
    policy = _policy(
        A1PolicyType.CLASS_FLOOR,
        A1PolicyStatement(
            class_floor_mbps={TrafficClass.LATENCY_SENSITIVE: 50.0, TrafficClass.BULK: 9_000.0}
        ),
    )

    constraint = policy_to_constraint(
        policy,
        _decision(
            A1PolicyStatus.PARTIALLY_ACCEPTED,
            enforced=("LATENCY_SENSITIVE",),
            rejected=("BULK",),
        ),
    )

    assert constraint.class_floor_mbps == {TrafficClass.LATENCY_SENSITIVE: 50.0}


def test_a_steering_policy_produces_no_objective_reweighting() -> None:
    # A policy must change exactly what it says it changes. Steering that also silently reweighted the
    # objective would be a far larger change than the operator asked for.
    policy = _policy(A1PolicyType.TRAFFIC_STEERING, A1PolicyStatement(avoid_link_ids=("n0->n1",)))

    constraint = policy_to_constraint(policy, _decision(A1PolicyStatus.ACCEPTED, enforced=("n0->n1",)))

    assert constraint.objective_weights is None
    assert constraint.class_floor_mbps is None
