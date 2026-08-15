"""Active constraints — owned by the controller, written only by the agent layer.

The controller expires constraints **itself**, on its own clock, rather than waiting to be told. That is the
property that makes constrained authority safe in practice: an agent that crashes, hangs, or is disconnected
mid-instruction leaves no residue. Every operator override has an end.
"""

from __future__ import annotations

from nexus_controller.class_floors import merge_floors
from nexus_errors.control_error import ConstraintRejectedError
from nexus_telemetry.client import get_telemetry
from nexus_types.constants import MAX_CONSTRAINT_TTL
from nexus_types.constraint import Constraint, ConstraintSet, ObjectiveWeights
from nexus_types.traffic_class import TrafficClass

_telemetry = get_telemetry("nexus.controller.constraints")


class ConstraintStore:
    """Holds the constraints in force and expires them as time advances."""

    def __init__(self) -> None:
        self._constraints: list[Constraint] = []

    def add(self, constraint: Constraint, current_timestep: int) -> Constraint:
        """Accept a constraint, or reject it with a reason the agent can relay to the operator.

        Rejection must surface as an error, never as a silent no-op: an agent told nothing will report
        success, and the operator will believe a change happened that did not.
        """
        if constraint.ttl_timesteps > MAX_CONSTRAINT_TTL:
            raise ConstraintRejectedError(
                f"TTL {constraint.ttl_timesteps} exceeds the maximum of {MAX_CONSTRAINT_TTL} timesteps. "
                "Every operator override must expire.",
                {"constraint_id": constraint.constraint_id},
            )
        if constraint.issued_at_timestep > current_timestep + 1:
            raise ConstraintRejectedError(
                "Constraint is issued in the future — the controller would apply it at an unpredictable "
                "point. Issue it against the current timestep.",
                {"issued_at": constraint.issued_at_timestep, "now": current_timestep},
            )

        if any(c.constraint_id == constraint.constraint_id for c in self._constraints):
            # remove() deletes every match, so a reused id would let one revocation silently lift an
            # unrelated constraint.
            raise ConstraintRejectedError(
                f"Constraint id {constraint.constraint_id!r} is already in use.",
                {"constraint_id": constraint.constraint_id},
            )

        self._constraints.append(constraint)
        _telemetry.info(
            "Constraint accepted",
            {
                "constraint_id": constraint.constraint_id,
                "issued_by": constraint.issued_by,
                "expires_at": constraint.expires_at_timestep(),
                "utterance": constraint.operator_utterance,
            },
        )
        return constraint

    def remove(self, constraint_id: str) -> bool:
        """Revoke a constraint early. Returns whether one was actually removed."""
        before = len(self._constraints)
        self._constraints = [c for c in self._constraints if c.constraint_id != constraint_id]
        return len(self._constraints) < before

    def prune(self, current_timestep: int) -> int:
        """Drop constraints that have expired *relative to the advancing simulation*.

        **Only ever call this from the control path**, where ``current_timestep`` is the real, monotonically
        advancing simulation time. Calling it from a read path is destructive: a reader asking about an
        earlier timestep would permanently delete every constraint issued after it. That was a live bug —
        `/health` pruned at the last-decision timestep and a liveness probe wiped the store every few
        seconds.

        Returns how many were dropped.
        """
        live = [c for c in self._constraints if not c.has_expired_by(current_timestep)]
        expired = len(self._constraints) - len(live)
        if expired:
            _telemetry.info("Constraints expired", {"count": expired, "timestep": current_timestep})
        self._constraints = live
        return expired

    def snapshot(self, current_timestep: int) -> ConstraintSet:
        """A **non-mutating** view of the store as of a timestep.

        Safe from any read path. ``ConstraintSet.active()`` does the filtering, so nothing is deleted just
        because a caller asked about a different moment.
        """
        return ConstraintSet(timestep=current_timestep, constraints=tuple(self._constraints))

    def active_snapshot(self, current_timestep: int) -> ConstraintSet:
        """A non-mutating view containing **only** the constraints in force.

        What every read endpoint should return. ``snapshot()`` carries the whole store, and serialising
        that over HTTP hands clients expired rules in the ``constraints`` field — the agent layer would
        then report a lapsed constraint as still in force. The filtering has to happen before the wire,
        not via a method the JSON does not carry.
        """
        return ConstraintSet(timestep=current_timestep, constraints=self.snapshot(current_timestep).active())

    def stored_count(self) -> int:
        """Total constraints held, expired or not. For diagnostics — see also ``active_snapshot``."""
        return len(self._constraints)

    def effective_weights(self, current_timestep: int) -> ObjectiveWeights:
        return self.snapshot(current_timestep).effective_weights()

    def forbidden_link_ids(self, current_timestep: int) -> frozenset[str]:
        return self.snapshot(current_timestep).forbidden_link_ids()

    def class_floors(self, current_timestep: int) -> dict[TrafficClass, float]:
        """Per-class rate guarantees in force, strongest wins per class."""
        return merge_floors(
            [c.class_floor_mbps for c in self.snapshot(current_timestep).active() if c.class_floor_mbps]
        )
