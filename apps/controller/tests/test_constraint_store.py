from __future__ import annotations

import pytest

from nexus_controller.constraint_store import ConstraintStore
from nexus_errors.control_error import ConstraintRejectedError
from nexus_types.constants import MAX_CONSTRAINT_TTL
from nexus_types.constraint import Constraint, ObjectiveWeights


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
        operator_utterance="prioritise low-latency traffic",
        issued_at_timestep=issued_at,
        ttl_timesteps=ttl,
        objective_weights=weights or ObjectiveWeights(),
        forbidden_link_ids=forbidden,
    )


def test_an_accepted_constraint_is_active_immediately() -> None:
    store = ConstraintStore()

    store.add(_constraint(issued_at=10), current_timestep=10)

    assert len(store.snapshot(10).active()) == 1


def test_the_controller_expires_constraints_without_being_told() -> None:
    # The property that makes constrained authority safe: an agent that goes silent leaves no residue.
    store = ConstraintStore()
    store.add(_constraint(issued_at=0, ttl=5), current_timestep=0)

    assert len(store.snapshot(4).active()) == 1
    assert len(store.snapshot(5).active()) == 0


def test_pruning_reports_how_many_expired() -> None:
    store = ConstraintStore()
    store.add(_constraint("a", issued_at=0, ttl=2), current_timestep=0)
    store.add(_constraint("b", issued_at=0, ttl=100), current_timestep=0)

    assert store.prune(50) == 1
    assert len(store.snapshot(50).active()) == 1


def test_a_ttl_above_the_maximum_is_refused_with_a_reason() -> None:
    store = ConstraintStore()
    constraint = Constraint(
        constraint_id="forever",
        issued_by="agent-layer",
        issued_at_timestep=0,
        ttl_timesteps=MAX_CONSTRAINT_TTL,
        objective_weights=ObjectiveWeights(),
    )
    store.add(constraint, current_timestep=0)  # at the limit is fine

    with pytest.raises(ConstraintRejectedError, match="must expire"):
        store.add(
            Constraint(
                constraint_id="way-too-long",
                issued_by="agent-layer",
                issued_at_timestep=0,
                # Bypasses the model bound to prove the store guards independently of the schema.
                ttl_timesteps=MAX_CONSTRAINT_TTL,
                objective_weights=ObjectiveWeights(),
            ).model_copy(update={"ttl_timesteps": MAX_CONSTRAINT_TTL + 1}),
            current_timestep=0,
        )


def test_a_constraint_issued_in_the_future_is_refused() -> None:
    store = ConstraintStore()

    with pytest.raises(ConstraintRejectedError, match="future"):
        store.add(_constraint(issued_at=500), current_timestep=10)


def test_rejection_raises_rather_than_silently_dropping() -> None:
    # A silent no-op would have the agent report success for a change that never happened.
    store = ConstraintStore()

    with pytest.raises(ConstraintRejectedError):
        store.add(_constraint(issued_at=999), current_timestep=0)

    assert len(store.snapshot(0).active()) == 0


def test_a_constraint_can_be_revoked_early() -> None:
    store = ConstraintStore()
    store.add(_constraint("c1"), current_timestep=0)

    assert store.remove("c1") is True
    assert len(store.snapshot(0).active()) == 0


def test_revoking_an_unknown_constraint_reports_that_nothing_changed() -> None:
    assert ConstraintStore().remove("never-existed") is False


def test_the_latest_weights_win_rather_than_being_blended() -> None:
    store = ConstraintStore()
    store.add(_constraint("early", issued_at=0, weights=ObjectiveWeights(latency=2.0)), 0)
    store.add(_constraint("late", issued_at=5, weights=ObjectiveWeights(latency=9.0)), 5)

    assert store.effective_weights(6).latency == 9.0


def test_forbidden_links_accumulate_while_constraints_are_live() -> None:
    store = ConstraintStore()
    store.add(_constraint("a", forbidden=("n0->n1",)), 0)
    store.add(_constraint("b", forbidden=("n1->n2",)), 0)

    assert store.forbidden_link_ids(1) == frozenset({"n0->n1", "n1->n2"})


def test_forbidden_links_disappear_when_the_constraint_expires() -> None:
    store = ConstraintStore()
    store.add(_constraint("a", issued_at=0, ttl=3, forbidden=("n0->n1",)), 0)

    assert store.forbidden_link_ids(1) == frozenset({"n0->n1"})
    assert store.forbidden_link_ids(10) == frozenset()


def test_weights_return_to_default_once_every_constraint_has_expired() -> None:
    store = ConstraintStore()
    store.add(_constraint(issued_at=0, ttl=2, weights=ObjectiveWeights(latency=9.0)), 0)

    assert store.effective_weights(100) == ObjectiveWeights()


def test_a_duplicate_constraint_id_is_refused() -> None:
    # remove() deletes every match, so a reused id would let one revocation silently lift an unrelated
    # constraint.
    store = ConstraintStore()
    store.add(_constraint("dup"), current_timestep=0)

    with pytest.raises(ConstraintRejectedError, match="already in use"):
        store.add(_constraint("dup"), current_timestep=0)

    assert store.stored_count() == 1
