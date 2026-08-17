"""What a registered policy version *is*.

Stage I's requirement, in one sentence: you can answer "which policy is deployed, where did it come from,
and what did it score?" instantly. Each of those three is a field group below, and each is mandatory —
a registry entry that records the artifact but not its provenance is a filename with extra steps.

The registry stores **metadata only**. Weights stay in ``experiments/runs/<run-id>/policy.zip``, which is
gitignored; this ledger is small JSON and is committed, so provenance outlives both the checkpoint and the
machine that produced it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus_errors.config_error import ConfigurationError
from nexus_registry.version import REGISTRY_SCHEMA_VERSION


class Stage(StrEnum):
    """Where a version sits in its lifecycle.

    Deliberately small. A stage is a *deployment claim*, and every extra one is another state whose
    meaning has to be agreed on by the controller, the dashboard and whoever is on call.
    """

    NONE = "none"
    """Registered and scored, deployed nowhere."""

    SHADOW = "shadow"
    """Runs alongside the incumbent, decisions logged, never applied. Stage L wires the execution side;
    the stage exists here from the start so promotion has somewhere to promote *from*."""

    PRODUCTION = "production"
    """Serving live traffic. At most one version holds this at a time."""

    ARCHIVED = "archived"
    """Retired. Kept because a rollback target that has been deleted is not a rollback target, and because
    the score history is the evidence for why it was retired."""


class Provenance(BaseModel):
    """Where a policy came from. Every field is required on purpose.

    A version missing its git SHA or its MDP version cannot be reproduced or even safely loaded, and
    "mostly traceable" is the property that lets an untraceable artifact into production.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1, description="Training run that produced the checkpoint.")
    git_sha: str = Field(min_length=1, description="Commit the run was launched from. May be '-dirty'.")
    mdp_version: int = Field(
        ge=1, description="Observation/action/reward contract the policy was trained against."
    )
    algorithm: str = Field(min_length=1)
    action_mode: str = Field(min_length=1)
    topology: str = Field(min_length=1)
    flow_count: int = Field(gt=0)
    traffic_pattern: str = Field(min_length=1, description="Pattern trained on.")
    seed: int
    total_timesteps: int = Field(gt=0)

    @model_validator(mode="after")
    def _reproducibility_is_recorded_honestly(self) -> Provenance:
        """A dirty tree is recorded, never hidden.

        ``current_git_sha`` marks uncommitted work with a ``-dirty`` suffix. That is not an error — you
        will register such runs while iterating — but it must survive into the ledger, because a version
        whose source cannot be checked out is not reproducible and a reader deserves to know.
        """
        if self.git_sha == "unknown":
            raise ConfigurationError(
                "A policy version cannot be registered without a git SHA: nothing could reproduce it. "
                "Register from inside the repository the run was launched from.",
                {"run_id": self.run_id},
            )
        return self


class EvaluationScore(BaseModel):
    """What a version scored on the reproducible evaluation suite.

    Scores are attached to a **named, versioned suite**. A number from an ad-hoc evaluation is not
    comparable to one from another ad-hoc evaluation, and a registry full of incomparable numbers is worse
    than one with none — it invites exactly the comparison it cannot support.
    """

    model_config = ConfigDict(frozen=True)

    suite_name: str = Field(min_length=1)
    suite_version: int = Field(ge=1)
    seeds: int = Field(ge=1)
    scored_at: datetime

    mean_p95_latency_ms: float
    mean_throughput_mbps: float
    mean_fairness: float
    total_dropped_mb: float

    per_pattern: dict[str, float] = Field(
        default_factory=dict,
        description="Mean p95 latency per traffic pattern. The aggregate hides a policy that wins on its "
        "training distribution and collapses off it — which is the failure worth catching.",
    )

    @property
    def is_citable(self) -> bool:
        """Whether this score meets the project's own bar for a stated result.

        Mirrors ``nexus_rl.evaluate.MINIMUM_SEEDS``. Restated as a property so the registry can refuse to
        promote on an anecdote without importing the evaluation harness.
        """
        return self.seeds >= 3


class PolicyVersion(BaseModel):
    """One registered, addressable policy.

    Immutable except for ``stage``, which the registry transitions explicitly. Rewriting a version's
    scores or provenance in place would destroy the audit trail that is the entire reason this exists —
    a new evaluation of the same checkpoint is a new score record, not an edit.
    """

    model_config = ConfigDict(frozen=True)

    version_id: str = Field(
        min_length=1,
        description="Addressable identifier, e.g. 'v3'. Assigned by the registry, never by the caller — "
        "a caller-chosen id is how two machines register different policies under the same name.",
    )
    schema_version: int = Field(default=REGISTRY_SCHEMA_VERSION, ge=1)
    registered_at: datetime
    provenance: Provenance
    scores: tuple[EvaluationScore, ...] = Field(
        default=(),
        description="Every evaluation this version has been through, newest last. A list rather than a "
        "single score: rescoring after a suite bump must not erase what it scored before.",
    )
    stage: Stage = Stage.NONE
    notes: str = Field(default="", description="Why this version exists, in the words of whoever made it.")

    @property
    def latest_score(self) -> EvaluationScore | None:
        return self.scores[-1] if self.scores else None

    def with_score(self, score: EvaluationScore) -> PolicyVersion:
        return self.model_copy(update={"scores": (*self.scores, score)})

    def with_stage(self, stage: Stage) -> PolicyVersion:
        return self.model_copy(update={"stage": stage})

    def describe(self) -> str:
        """One line an operator can read on ``/health`` or in a log.

        Names the version, its origin and its standing. "final_model.zip" answers none of those, which is
        the failure mode this whole package exists to remove.
        """
        score = self.latest_score
        scored = (
            f"p95 {score.mean_p95_latency_ms:.1f}ms over {score.seeds} seeds "
            f"on {score.suite_name}@v{score.suite_version}"
            if score is not None
            else "unscored"
        )
        return (
            f"{self.version_id} ({self.stage.value}) — {self.provenance.algorithm} from run "
            f"{self.provenance.run_id} @ {self.provenance.git_sha}, "
            f"mdp{self.provenance.mdp_version}, {scored}"
        )


def utc_now() -> datetime:
    """Timestamp for a registry record.

    A module-level helper rather than a default factory so tests can register versions at controlled times
    without patching ``datetime`` globally.
    """
    return datetime.now(UTC)
