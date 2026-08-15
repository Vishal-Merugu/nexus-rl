from __future__ import annotations

import pytest
from pydantic import ValidationError

from nexus_errors.control_error import ConstraintRejectedError
from nexus_types.constants import MAX_CONSTRAINT_TTL
from nexus_types.constraint import Constraint, ConstraintSet, ObjectiveWeights
from nexus_types.traffic_class import TrafficClass


def _constraint(
    constraint_id: str = "c1",
    issued_at: int = 0,
    ttl: int = 100,
    weights: ObjectiveWeights | None = None,
    forbidden: tuple[str, ...] = (),
) -> Constraint:
    return Constraint(
        constraint_id=constraint_id,
        issued_by="agent-layer",
        issued_at_timestep=issued_at,
        ttl_timesteps=ttl,
        objective_weights=weights if weights is not None or forbidden else ObjectiveWeights(),
        forbidden_link_ids=forbidden,
    )


def test_constraint_expires_at_issue_plus_ttl() -> None:
    constraint = _constraint(issued_at=10, ttl=5)

    assert constraint.is_active_at(10)
    assert constraint.is_active_at(14)
    assert not constraint.is_active_at(15)
    assert not constraint.is_active_at(9)


def test_ttl_above_max_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _constraint(ttl=MAX_CONSTRAINT_TTL + 1)


def test_zero_ttl_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _constraint(ttl=0)


def test_constraint_that_constrains_nothing_is_rejected() -> None:
    with pytest.raises(ConstraintRejectedError):
        Constraint(
            constraint_id="empty",
            issued_by="agent-layer",
            issued_at_timestep=0,
            ttl_timesteps=10,
        )


def test_all_zero_objective_weights_are_rejected() -> None:
    with pytest.raises(ConstraintRejectedError):
        ObjectiveWeights(throughput=0.0, latency=0.0, drops=0.0, fairness=0.0)


def test_class_floor_alone_is_a_valid_constraint() -> None:
    constraint = Constraint(
        constraint_id="floor",
        issued_by="agent-layer",
        issued_at_timestep=0,
        ttl_timesteps=10,
        class_floor_mbps={TrafficClass.LATENCY_SENSITIVE: 50.0},
    )

    assert constraint.class_floor_mbps == {TrafficClass.LATENCY_SENSITIVE: 50.0}


def test_expired_constraints_drop_out_of_the_active_set() -> None:
    constraints = ConstraintSet(
        timestep=50,
        constraints=(_constraint("old", issued_at=0, ttl=10), _constraint("live", issued_at=40, ttl=20)),
    )

    assert [c.constraint_id for c in constraints.active()] == ["live"]


def test_effective_weights_take_the_latest_writer_not_a_blend() -> None:
    early = _constraint("early", issued_at=0, weights=ObjectiveWeights(latency=2.0))
    late = _constraint("late", issued_at=5, weights=ObjectiveWeights(latency=8.0))

    constraints = ConstraintSet(timestep=10, constraints=(early, late))

    assert constraints.effective_weights().latency == 8.0


def test_effective_weights_default_when_nothing_is_active() -> None:
    constraints = ConstraintSet(timestep=999, constraints=(_constraint(issued_at=0, ttl=10),))

    assert constraints.effective_weights() == ObjectiveWeights()


def test_forbidden_links_accumulate_across_active_constraints() -> None:
    constraints = ConstraintSet(
        timestep=1,
        constraints=(
            _constraint("a", forbidden=("n1->n2",)),
            _constraint("b", forbidden=("n2->n3",)),
            _constraint("expired", issued_at=0, ttl=1, forbidden=("n9->n9",)),
        ),
    )

    assert constraints.forbidden_link_ids() == frozenset({"n1->n2", "n2->n3"})


def test_expiry_is_distinct_from_not_yet_active() -> None:
    # Garbage collection must not delete constraints that simply have not started. Pruning on
    # "not active" destroyed future-dated constraints whenever a caller asked about an earlier moment.
    constraint = _constraint(issued_at=100, ttl=50)

    assert constraint.has_expired_by(150) is True
    assert constraint.has_expired_by(149) is False
    assert constraint.has_expired_by(0) is False  # not started, but certainly not expired
    assert constraint.is_active_at(0) is False


def test_weight_ties_resolve_deterministically_by_id() -> None:
    same_step_a = _constraint("a", issued_at=5, weights=ObjectiveWeights(latency=2.0))
    same_step_b = _constraint("b", issued_at=5, weights=ObjectiveWeights(latency=7.0))

    forward = ConstraintSet(timestep=10, constraints=(same_step_a, same_step_b)).effective_weights()
    reversed_order = ConstraintSet(timestep=10, constraints=(same_step_b, same_step_a)).effective_weights()

    assert forward == reversed_order
    assert forward.latency == 7.0
