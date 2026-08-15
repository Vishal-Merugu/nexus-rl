"""The xApp's A1 policy surface.

The controller is a Near-RT RIC xApp: it consumes policy guidance over A1 and retains control authority,
optimizing *within* that guidance. This module owns the consumption side — validate, decide, and on
acceptance translate into the constraint machinery that already enforces expiry, objective precedence,
accumulating link restrictions and class floors.

A1 is the interface; ``Constraint`` is the enforcement. Nothing about the control path changed to add
O-RAN alignment, which is the point: the two-tier structure was already the one the standard describes.
See ADR-0008.

**Validation runs against the network as last observed on ``/act``.** The controller holds no model of
the topology (ADR-0006), so the only network it can legitimately reason about is the one the simulator
last sent it. Offering a policy before any state has arrived is refused rather than accepted blind —
an accepted-but-unvalidated policy is exactly the false assurance this boundary exists to prevent.
"""

from __future__ import annotations

from nexus_a1.decision import A1PolicyDecision, A1PolicyStatus
from nexus_a1.policy import A1Policy, A1PolicyType
from nexus_a1.translate import policy_to_constraint
from nexus_a1.validation import SUPPORTED_POLICY_TYPES, validate_policy
from nexus_controller.constraint_store import ConstraintStore
from nexus_errors.control_error import ConstraintRejectedError
from nexus_telemetry.client import get_telemetry
from nexus_types.network_state import NetworkState

_telemetry = get_telemetry("nexus.controller.a1")


class A1PolicyStore:
    """Accepted A1 policies and the decisions that admitted them.

    Wraps a :class:`ConstraintStore` rather than replacing it: enforcement, expiry and precedence are
    unchanged, so the O-RAN work is additive over a control path that is already tested.
    """

    def __init__(self, constraints: ConstraintStore) -> None:
        self._constraints = constraints
        self._policies: dict[str, A1Policy] = {}
        self._decisions: dict[str, A1PolicyDecision] = {}

    @property
    def supported_types(self) -> frozenset[A1PolicyType]:
        """What this xApp advertises it can enforce.

        The Non-RT tier may only issue these. Advertising the set explicitly is what lets the rApp choose
        an achievable policy rather than discovering the limit by rejection.
        """
        return SUPPORTED_POLICY_TYPES

    def offer(self, policy: A1Policy, state: NetworkState | None, current_timestep: int) -> A1PolicyDecision:
        """Offer a policy to the xApp and get its decision.

        Accepting a policy id that is already in force is refused rather than treated as an update:
        ``ConstraintStore.remove`` deletes every match, so a reused id would let one deletion silently
        lift unrelated guidance. Delete then re-issue.
        """
        if state is None:
            return A1PolicyDecision(
                policy_id=policy.policy_id,
                status=A1PolicyStatus.REJECTED,
                reason=(
                    "The controller has not yet received any network state, so this policy's "
                    "feasibility cannot be checked. Accepting it unvalidated would report a guarantee "
                    "that may be impossible. Retry once the control loop is running."
                ),
            )

        if policy.policy_id in self._policies:
            return A1PolicyDecision(
                policy_id=policy.policy_id,
                status=A1PolicyStatus.REJECTED,
                reason=(
                    f"Policy id {policy.policy_id!r} is already in force. Delete it before re-issuing; "
                    f"reusing an id would make a later deletion ambiguous."
                ),
            )

        decision = validate_policy(policy, state, current_timestep, self.supported_types)

        if decision.is_enforced:
            constraint = policy_to_constraint(policy, decision)
            try:
                self._constraints.add(constraint, current_timestep)
            except ConstraintRejectedError as error:
                # The enforcement layer refused what validation admitted. Surfacing the real reason
                # beats reporting a success the control path will not honour.
                _telemetry.capture_exception(error, "A1 policy passed validation but failed enforcement")
                return A1PolicyDecision(
                    policy_id=policy.policy_id,
                    status=A1PolicyStatus.REJECTED,
                    reason=f"Policy could not be enforced: {error}",
                )
            self._policies[policy.policy_id] = policy

        self._decisions[policy.policy_id] = decision
        _telemetry.info(
            "A1 policy decision",
            {
                "policy_id": policy.policy_id,
                "policy_type": policy.policy_type.value,
                "status": decision.status.value,
                "issued_by": policy.issued_by,
                "utterance": policy.operator_utterance,
                "enforced": list(decision.enforced_elements),
                "rejected": list(decision.rejected_elements),
            },
        )
        return decision

    def delete(self, policy_id: str) -> bool:
        """Withdraw a policy. Returns whether one was actually in force."""
        removed = self._constraints.remove(policy_id)
        self._policies.pop(policy_id, None)
        self._decisions.pop(policy_id, None)
        return removed

    def active(self, current_timestep: int) -> tuple[A1Policy, ...]:
        """Policies in force at a timestep. **Non-mutating** — safe from any read path."""
        return tuple(p for p in self._policies.values() if p.is_active_at(current_timestep))

    def decision_for(self, policy_id: str) -> A1PolicyDecision | None:
        return self._decisions.get(policy_id)

    def prune(self, current_timestep: int) -> int:
        """Drop policies whose lifetime has elapsed.

        Call only from the control path, where the timestep is real advancing simulation time — the same
        rule as ``ConstraintStore.prune``, and for the same reason.
        """
        expired = [pid for pid, p in self._policies.items() if p.has_expired_by(current_timestep)]
        for policy_id in expired:
            del self._policies[policy_id]
            self._decisions.pop(policy_id, None)
        if expired:
            _telemetry.info("A1 policies expired", {"count": len(expired), "timestep": current_timestep})
        return len(expired)
