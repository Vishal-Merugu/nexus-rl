"""The A1 policy document — what the Non-RT tier issues to the Near-RT tier.

O-RAN separates intelligence across two timescales. The Non-RT RIC (here: the LLM agent layer, an rApp)
issues **policy guidance**; the Near-RT RIC (here: the controller, an xApp) retains control authority and
optimizes *within* that guidance. The Non-RT tier never issues a control action.

That is exactly the constrained-authority rule ADR-0002 arrived at independently, which is why this
package is a re-expression of the v1 constraint model against a published interface rather than a rewrite
of it. See ADR-0008.

The shape follows A1's own decomposition:

- **policy type** — which kind of guidance this is; determines how the statement is interpreted
- **scope** — what the guidance applies to (which traffic classes, which links)
- **statement** — the guidance itself, as objectives and limits

Every policy is time-boxed. A1 in the standard has no mandatory TTL, and this project adds one on purpose:
an operator override that never expires becomes undocumented permanent policy nobody remembers setting.
That deviation is deliberate and is called out in ADR-0008.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus_a1.version import A1_SCHEMA_VERSION
from nexus_errors.control_error import ConstraintRejectedError
from nexus_types.constants import MAX_CONSTRAINT_TTL, MAX_OBJECTIVE_WEIGHT
from nexus_types.traffic_class import TrafficClass


class A1PolicyType(StrEnum):
    """The policy types this xApp advertises.

    An xApp declares which policy types it can enforce; the Non-RT tier may only issue those. A policy of
    an unadvertised type is refused rather than partially honoured — pretending to enforce guidance you
    cannot interpret is the failure this whole boundary exists to prevent.
    """

    QOS_OBJECTIVE = "QOS_OBJECTIVE"
    """Reweight what the controller optimizes for. Does not retrain the policy — it changes the objective
    the policy's actions are scored against."""

    TRAFFIC_STEERING = "TRAFFIC_STEERING"
    """Keep new traffic off named links. A routing restriction, never a routing instruction."""

    CLASS_FLOOR = "CLASS_FLOOR"
    """Guarantee a minimum aggregate rate to a traffic class."""


class A1PolicyScope(BaseModel):
    """What a policy applies to.

    An empty scope means network-wide. Naming classes or links narrows it — which is what makes partial
    acceptance meaningful: the xApp can honour the part of the scope it can reach and say plainly which
    part it cannot.
    """

    model_config = ConfigDict(frozen=True)

    traffic_classes: tuple[TrafficClass, ...] = Field(
        default=(), description="Classes this policy applies to. Empty means all classes."
    )
    link_ids: tuple[str, ...] = Field(
        default=(),
        max_length=32,
        description="Links this policy applies to. Empty means the whole topology.",
    )


class QosObjective(BaseModel):
    """Relative importance of each reward component, as the operator wants it weighted.

    Mirrors ``nexus_types.constraint.ObjectiveWeights``. Restated here because A1 is a wire contract with
    its own version: the two are free to diverge, and the translation in ``nexus_a1.translate`` is the
    single place that has to know about both.
    """

    model_config = ConfigDict(frozen=True)

    throughput: float = Field(default=1.0, ge=0.0, le=MAX_OBJECTIVE_WEIGHT)
    latency: float = Field(default=1.0, ge=0.0, le=MAX_OBJECTIVE_WEIGHT)
    drops: float = Field(default=1.0, ge=0.0, le=MAX_OBJECTIVE_WEIGHT)
    fairness: float = Field(default=1.0, ge=0.0, le=MAX_OBJECTIVE_WEIGHT)

    @model_validator(mode="after")
    def _not_all_zero(self) -> QosObjective:
        if self.throughput == self.latency == self.drops == self.fairness == 0.0:
            raise ConstraintRejectedError(
                "All objective weights are zero, which leaves the controller with nothing to optimize."
            )
        return self


class A1PolicyStatement(BaseModel):
    """The guidance itself.

    Exactly one field is populated, chosen by the policy's type. Keeping them on one model rather than a
    union per type keeps the wire shape stable across schema versions — a consumer that does not know a
    newer policy type still parses the document and can refuse it by *type*, with a clear reason, instead
    of failing to deserialize it at all.
    """

    model_config = ConfigDict(frozen=True)

    qos_objective: QosObjective | None = None
    avoid_link_ids: tuple[str, ...] = Field(default=(), max_length=32)
    class_floor_mbps: dict[TrafficClass, float] | None = None

    @model_validator(mode="after")
    def _floors_are_non_negative(self) -> A1PolicyStatement:
        for traffic_class, floor in (self.class_floor_mbps or {}).items():
            if floor < 0.0:
                raise ConstraintRejectedError(
                    "A class floor cannot be negative.",
                    {"traffic_class": traffic_class.value, "floor_mbps": floor},
                )
        return self


class A1Policy(BaseModel):
    """One A1 policy document.

    Immutable. Updating guidance means issuing a new policy or deleting this one — an A1 policy is
    identified state at the consumer, not a message the producer can retroactively edit.
    """

    model_config = ConfigDict(frozen=True)

    policy_id: str = Field(min_length=1, description="Unique instance id, assigned by the issuer.")
    policy_type: A1PolicyType
    schema_version: int = Field(
        default=A1_SCHEMA_VERSION,
        ge=1,
        description="Schema this document was written against. The consumer refuses versions it cannot "
        "interpret rather than guessing at the difference.",
    )

    scope: A1PolicyScope = Field(default_factory=A1PolicyScope)
    statement: A1PolicyStatement

    issued_by: str = Field(min_length=1, description="Provenance, e.g. 'rapp:agent-layer'. Never blank.")
    operator_utterance: str = Field(
        default="",
        description="The natural-language intent behind this policy. Kept so every automated change "
        "stays attributable to a human instruction after the fact.",
    )
    issued_at_timestep: int = Field(ge=0)
    ttl_timesteps: int = Field(
        gt=0,
        le=MAX_CONSTRAINT_TTL,
        description="Mandatory and bounded — a deliberate deviation from A1, which has no required "
        "lifetime. The consumer expires the policy itself.",
    )

    @model_validator(mode="after")
    def _statement_matches_type(self) -> A1Policy:
        """The statement must actually carry the guidance its type promises.

        A ``CLASS_FLOOR`` policy with no floors is accepted by every field validator individually and does
        nothing at all — the operator is told their instruction is in force while the network ignores it.
        That is the empty-success failure, one layer up from the tool boundary.
        """
        required: dict[A1PolicyType, object] = {
            A1PolicyType.QOS_OBJECTIVE: self.statement.qos_objective,
            A1PolicyType.TRAFFIC_STEERING: self.statement.avoid_link_ids or None,
            A1PolicyType.CLASS_FLOOR: self.statement.class_floor_mbps or None,
        }
        if required[self.policy_type] is None:
            raise ConstraintRejectedError(
                f"A {self.policy_type.value} policy carries no {self.policy_type.value.lower()} "
                f"statement, so it would be enforced as a no-op.",
                {"policy_id": self.policy_id, "policy_type": self.policy_type.value},
            )
        return self

    def expires_at_timestep(self) -> int:
        return self.issued_at_timestep + self.ttl_timesteps

    def is_active_at(self, timestep: int) -> bool:
        return self.issued_at_timestep <= timestep < self.expires_at_timestep()

    def has_expired_by(self, timestep: int) -> bool:
        """Whether this policy is permanently finished as of ``timestep``.

        Distinct from ``not is_active_at(...)``, which is also true *before* it starts. Garbage collection
        must use this one — see the same distinction on ``nexus_types.constraint.Constraint``.
        """
        return timestep >= self.expires_at_timestep()
