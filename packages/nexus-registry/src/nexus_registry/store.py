"""The registry itself — a filesystem ledger of policy versions.

One JSON document per version under ``<root>/versions/``. There is no database and no server, which is a
deliberate consequence of the project's existing rule (see ADR-0009): the thing being stored is a few
hundred bytes of provenance per trained policy, and standing up a tracking service to hold it would add an
operational dependency to a system whose whole point is that it runs offline from a clone.

**The stage lives inside each version record, not in a separate alias file.** An alias table would be a
second source of truth about which policy is deployed, and two components deriving that independently is
exactly the failure `tasks/lessons.md` records under "two services must never derive shared state
independently". Resolving ``production`` means scanning for the version that claims it — and the scan
refuses if two do.

Single-writer by design: the training and evaluation CLIs write, the controller only reads.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from nexus_errors.config_error import ConfigurationError, MdpVersionMismatchError
from nexus_registry.record import EvaluationScore, PolicyVersion, Provenance, Stage, utc_now
from nexus_registry.version import MIN_SUPPORTED_SCHEMA_VERSION, REGISTRY_SCHEMA_VERSION

VERSION_PREFIX = "v"


class PolicyRegistry:
    """Versioned policy artifacts with provenance, scores and deployment stage."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._versions_dir = root / "versions"

    @property
    def root(self) -> Path:
        return self._root

    # --- Reads ------------------------------------------------------------------------------------

    def list_versions(self) -> tuple[PolicyVersion, ...]:
        """Every registered version, oldest first."""
        if not self._versions_dir.exists():
            return ()
        records = [self._read(path) for path in sorted(self._versions_dir.glob("*.json"))]
        return tuple(sorted(records, key=lambda record: _version_number(record.version_id)))

    def get(self, version_id: str) -> PolicyVersion:
        path = self._versions_dir / f"{version_id}.json"
        if not path.exists():
            known = [record.version_id for record in self.list_versions()]
            raise ConfigurationError(
                f"No policy version {version_id!r} in the registry at {self._root}.",
                {"version_id": version_id, "known": known},
            )
        return self._read(path)

    def resolve(self, identifier: str) -> PolicyVersion:
        """Look up by version id (``v3``) or by stage (``production``, ``shadow``).

        Resolving by stage is what lets the controller be configured with an intent — "serve whatever is
        in production" — rather than a filename that has to be edited on every promotion.
        """
        stage = _parse_stage(identifier)
        if stage is None:
            return self.get(identifier)

        claiming = [record for record in self.list_versions() if record.stage is stage]
        if not claiming:
            raise ConfigurationError(
                f"No policy version is at stage {stage.value!r}.",
                {"stage": stage.value, "registered": len(self.list_versions())},
            )
        if len(claiming) > 1:
            # Only reachable if the ledger was hand-edited. Refusing beats picking one: serving an
            # arbitrary policy while reporting a definite stage is a lie the operator cannot detect.
            raise ConfigurationError(
                f"{len(claiming)} versions claim stage {stage.value!r}; the registry is inconsistent.",
                {"stage": stage.value, "versions": [record.version_id for record in claiming]},
            )
        return claiming[0]

    def production(self) -> PolicyVersion | None:
        """The version currently serving, or ``None``. Never raises — "nothing deployed" is a valid state."""
        claiming = [record for record in self.list_versions() if record.stage is Stage.PRODUCTION]
        return claiming[0] if len(claiming) == 1 else None

    # --- Writes -----------------------------------------------------------------------------------

    def register(self, provenance: Provenance, notes: str = "") -> PolicyVersion:
        """Add a new version. Ids are assigned here, never by the caller.

        Registering the same run twice is refused. A checkpoint is one artifact; two version ids pointing
        at it would let the same policy be promoted, compared against itself, and reported as two.
        """
        existing = self.list_versions()
        duplicate = next(
            (record for record in existing if record.provenance.run_id == provenance.run_id), None
        )
        if duplicate is not None:
            raise ConfigurationError(
                f"Run {provenance.run_id!r} is already registered as {duplicate.version_id!r}. "
                f"Re-evaluate that version rather than registering the checkpoint twice.",
                {"run_id": provenance.run_id, "version_id": duplicate.version_id},
            )

        next_number = max((_version_number(record.version_id) for record in existing), default=0) + 1
        version = PolicyVersion(
            version_id=f"{VERSION_PREFIX}{next_number}",
            registered_at=utc_now(),
            provenance=provenance,
            notes=notes,
        )
        self._write(version)
        return version

    def record_score(self, version_id: str, score: EvaluationScore) -> PolicyVersion:
        """Append an evaluation result. Never replaces an earlier one."""
        updated = self.get(version_id).with_score(score)
        self._write(updated)
        return updated

    def promote(
        self, version_id: str, stage: Stage, current_mdp_version: int, force: bool = False
    ) -> PolicyVersion:
        """Move a version to a deployment stage, demoting whatever held it.

        Three refusals, each one a thing that would otherwise be discovered in production:

        - **Stale MDP.** A policy trained against a different observation/action/reward contract does not
          merely score badly; its outputs mean something else entirely.
        - **Unscored, or scored on too few seeds.** Promoting on an anecdote is how a policy that lost
          ends up deployed. ``force`` exists for a deliberate operator override and is recorded in the
          notes, never silent.
        - **Nothing to demote to.** Handled by demoting the incumbent to ``ARCHIVED`` rather than
          ``NONE``: a rollback target must remain findable.
        """
        version = self.get(version_id)

        if stage is Stage.PRODUCTION or stage is Stage.SHADOW:
            if version.provenance.mdp_version != current_mdp_version:
                raise MdpVersionMismatchError(
                    expected=current_mdp_version,
                    found=version.provenance.mdp_version,
                    artifact=f"policy version {version_id}",
                )
            score = version.latest_score
            if score is None and not force:
                raise ConfigurationError(
                    f"Policy version {version_id!r} has never been evaluated. Score it against the "
                    f"evaluation suite before deploying it, or pass force with a reason.",
                    {"version_id": version_id, "stage": stage.value},
                )
            if score is not None and not score.is_citable and not force:
                raise ConfigurationError(
                    f"Policy version {version_id!r} was scored on {score.seeds} seed(s). Seed variance in "
                    f"RL routinely exceeds the effect being measured — that is not evidence it is better.",
                    {"version_id": version_id, "seeds": score.seeds},
                )

        # Demote the incumbent first. Doing it after would leave a window in which two versions claim the
        # stage, and `resolve` refuses that state — a crash mid-promotion would strand the controller.
        for incumbent in self.list_versions():
            if incumbent.stage is stage and incumbent.version_id != version_id:
                self._write(incumbent.with_stage(Stage.ARCHIVED))

        promoted = version.with_stage(stage)
        self._write(promoted)
        return promoted

    # --- Persistence ------------------------------------------------------------------------------

    def _write(self, version: PolicyVersion) -> None:
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        path = self._versions_dir / f"{version.version_id}.json"
        # allow_nan=False: a NaN score would serialise as a bare `NaN` literal, which is not JSON and
        # which every other reader — jq, the dashboard, a browser — rejects. Same rule as metrics.jsonl.
        payload = json.dumps(version.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False)
        path.write_text(payload + "\n", encoding="utf-8")

    def _read(self, path: Path) -> PolicyVersion:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ConfigurationError(
                f"Registry record {path.name} could not be read: {error}",
                {"path": str(path)},
            ) from error

        if not isinstance(raw, dict):
            raise ConfigurationError(f"Registry record {path.name} is not an object.", {"path": str(path)})

        _check_schema_version(raw.get("schema_version"), path)

        try:
            return PolicyVersion.model_validate(raw)
        except ValidationError as error:
            raise ConfigurationError(
                f"Registry record {path.name} is malformed: {error.error_count()} problem(s). A version "
                f"whose provenance cannot be read must not be served.",
                {"path": str(path), "errors": error.errors(include_url=False)},
            ) from error


def _check_schema_version(recorded: object, path: Path) -> None:
    if not isinstance(recorded, int) or isinstance(recorded, bool):
        raise ConfigurationError(
            f"Registry record {path.name} declares no schema version.", {"path": str(path)}
        )
    if recorded > REGISTRY_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Registry record {path.name} uses schema version {recorded}; this build understands up to "
            f"{REGISTRY_SCHEMA_VERSION}. Upgrade rather than guessing at fields it does not define.",
            {"path": str(path), "found": recorded, "supported": REGISTRY_SCHEMA_VERSION},
        )
    if recorded < MIN_SUPPORTED_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Registry record {path.name} uses retired schema version {recorded}.",
            {"path": str(path), "found": recorded},
        )


def _parse_stage(identifier: str) -> Stage | None:
    """A stage name, or ``None`` when the identifier is a version id.

    Compared against the enum's own values so adding a stage cannot leave this behind.
    """
    for stage in Stage:
        if identifier == stage.value:
            return stage
    return None


def _version_number(version_id: str) -> int:
    """Sort key for ``v12``-style ids. Lexical order would place ``v10`` before ``v2``."""
    digits = version_id.removeprefix(VERSION_PREFIX)
    return int(digits) if digits.isdigit() else 0
