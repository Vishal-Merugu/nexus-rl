"""Turning an accepted A1 policy into the constraint the controller enforces.

The v1 ``Constraint`` is not replaced by A1 — it becomes the xApp's **internal** representation of
accepted guidance. A1 is the interface; the constraint is the enforcement. Keeping both means the
enforcement path that is already tested (expiry, last-writer-wins objectives, accumulating link
restrictions, class floors) carries straight over, and the O-RAN work is genuinely additive rather than a
rewrite of working code.

Translation happens **after** validation and reflects the decision: a partially-accepted steering policy
translates to a constraint covering only the links the xApp agreed to enforce. Translating the original
request instead would apply guidance the operator was told had been refused.
"""

from __future__ import annotations

from nexus_a1.decision import A1PolicyDecision
from nexus_a1.policy import A1Policy, A1PolicyType
from nexus_errors.control_error import ConstraintRejectedError
from nexus_types.constraint import Constraint, ObjectiveWeights
from nexus_types.traffic_class import TrafficClass


def policy_to_constraint(policy: A1Policy, decision: A1PolicyDecision) -> Constraint:
    """Build the constraint that enforces an accepted (or partially accepted) policy.

    Raises if the decision rejected the policy — a rejected policy has no enforcement, and constructing
    one anyway is how a refusal silently becomes an application.
    """
    if not decision.is_enforced:
        raise ConstraintRejectedError(
            "A rejected A1 policy has no enforcement. Do not translate it.",
            {"policy_id": policy.policy_id, "reason": decision.reason},
        )

    objective_weights: ObjectiveWeights | None = None
    forbidden_link_ids: tuple[str, ...] = ()
    class_floor_mbps: dict[TrafficClass, float] | None = None

    if policy.policy_type is A1PolicyType.QOS_OBJECTIVE:
        objective = policy.statement.qos_objective
        assert objective is not None  # guaranteed by A1Policy._statement_matches_type
        objective_weights = ObjectiveWeights(
            throughput=objective.throughput,
            latency=objective.latency,
            drops=objective.drops,
            fairness=objective.fairness,
        )

    elif policy.policy_type is A1PolicyType.TRAFFIC_STEERING:
        # Only what the xApp agreed to enforce. `enforced_elements` is the decision's own record of
        # which links survived validation, so a link the operator was told is unknown cannot slip back
        # into the constraint here.
        forbidden_link_ids = decision.enforced_elements

    elif policy.policy_type is A1PolicyType.CLASS_FLOOR:
        enforced_classes = set(decision.enforced_elements)
        class_floor_mbps = {
            traffic_class: floor
            for traffic_class, floor in (policy.statement.class_floor_mbps or {}).items()
            if traffic_class.value in enforced_classes
        }

    return Constraint(
        constraint_id=policy.policy_id,
        issued_by=policy.issued_by,
        operator_utterance=policy.operator_utterance,
        issued_at_timestep=policy.issued_at_timestep,
        ttl_timesteps=policy.ttl_timesteps,
        objective_weights=objective_weights,
        class_floor_mbps=class_floor_mbps or None,
        forbidden_link_ids=forbidden_link_ids,
    )
