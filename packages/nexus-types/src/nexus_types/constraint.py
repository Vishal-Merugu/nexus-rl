"""The constraint contract — the only thing the agent layer may write.

A constraint is a time-boxed modifier on the controller's optimization. It reweights objectives, restricts
actions, or floors a traffic class. It never sets an allocation. Every constraint expires on its own: the
controller drops it when the TTL elapses rather than waiting to be told, so an agent that goes silent leaves
no residue.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus_errors.control_error import ConstraintRejectedError
from nexus_types.constants import MAX_CONSTRAINT_TTL, MAX_OBJECTIVE_WEIGHT
from nexus_types.traffic_class import TrafficClass


class ObjectiveWeights(BaseModel):
    """Relative weights of the reward components the controller optimizes against.

    Changing these does not retrain the policy — it changes the objective the policy's actions are scored
    and masked against. Distinct from the reward weights used *during* training, which are fixed in the
    training config and part of the MDP.
    """

    model_config = ConfigDict(frozen=True)

    throughput: float = Field(default=1.0, ge=0.0, le=MAX_OBJECTIVE_WEIGHT)
    latency: float = Field(default=1.0, ge=0.0, le=MAX_OBJECTIVE_WEIGHT)
    drops: float = Field(default=1.0, ge=0.0, le=MAX_OBJECTIVE_WEIGHT)
    fairness: float = Field(default=1.0, ge=0.0, le=MAX_OBJECTIVE_WEIGHT)

    @model_validator(mode="after")
    def _not_all_zero(self) -> ObjectiveWeights:
        if self.throughput == self.latency == self.drops == self.fairness == 0.0:
            raise ConstraintRejectedError(
                "All objective weights are zero, which leaves the controller with nothing to optimize."
            )
        return self


class Constraint(BaseModel):
    """A time-boxed modifier the agent layer applies to the controller.

    At least one of ``objective_weights``, ``class_floor_mbps`` or ``forbidden_link_ids`` must be set —
    a constraint that constrains nothing is a silent no-op the operator would read as success.
    """

    model_config = ConfigDict(frozen=True)

    constraint_id: str
    issued_by: str = Field(description="Provenance, e.g. 'agent-layer'. Never blank.", min_length=1)
    operator_utterance: str = Field(
        default="",
        description="The natural-language instruction that produced this constraint. Makes every agent "
        "change attributable after the fact, which is what keeps explanations honest.",
    )
    issued_at_timestep: int = Field(ge=0)
    ttl_timesteps: int = Field(
        gt=0,
        le=MAX_CONSTRAINT_TTL,
        description="Mandatory and bounded. The controller expires the constraint itself.",
    )

    objective_weights: ObjectiveWeights | None = None
    class_floor_mbps: dict[TrafficClass, float] | None = Field(
        default=None, description="Minimum aggregate rate guaranteed to a traffic class."
    )
    forbidden_link_ids: tuple[str, ...] = Field(
        default=(), description="Links the controller may not route new traffic onto."
    )

    @model_validator(mode="after")
    def _constrains_something(self) -> Constraint:
        if self.objective_weights is None and not self.class_floor_mbps and not self.forbidden_link_ids:
            raise ConstraintRejectedError(
                "Constraint has no effect. Set objective weights, a class floor, or forbidden links.",
                {"constraint_id": self.constraint_id},
            )
        return self

    def expires_at_timestep(self) -> int:
        return self.issued_at_timestep + self.ttl_timesteps

    def is_active_at(self, timestep: int) -> bool:
        return self.issued_at_timestep <= timestep < self.expires_at_timestep()

    def has_expired_by(self, timestep: int) -> bool:
        """Whether this constraint is permanently finished as of ``timestep``.

        Distinct from ``not is_active_at(...)``, which is also true *before* a constraint starts. Garbage
        collection must use this one — pruning on "not active" would delete constraints that have not begun
        yet whenever a caller asks about an earlier moment.
        """
        return timestep >= self.expires_at_timestep()


class ConstraintSet(BaseModel):
    """The constraints in force at a given timestep."""

    model_config = ConfigDict(frozen=True)

    timestep: int = Field(ge=0)
    constraints: tuple[Constraint, ...] = ()

    def active(self) -> tuple[Constraint, ...]:
        return tuple(c for c in self.constraints if c.is_active_at(self.timestep))

    def effective_weights(self) -> ObjectiveWeights:
        """Resolve active constraints into one set of weights.

        Last writer wins: the most recently issued constraint carrying weights takes effect. Blending them
        would produce an objective no operator asked for and none could predict.
        """
        weighted = [c for c in self.active() if c.objective_weights is not None]
        if not weighted:
            return ObjectiveWeights()
        # Ties broken by id so two constraints issued on the same timestep resolve identically on every
        # replay — insertion order would make the effective objective depend on request arrival order.
        latest = max(weighted, key=lambda c: (c.issued_at_timestep, c.constraint_id))
        assert latest.objective_weights is not None  # narrowed by the filter above
        return latest.objective_weights

    def forbidden_link_ids(self) -> frozenset[str]:
        """Union of every active constraint's forbidden links. Restrictions accumulate; they never cancel."""
        return frozenset(link_id for c in self.active() for link_id in c.forbidden_link_ids)
