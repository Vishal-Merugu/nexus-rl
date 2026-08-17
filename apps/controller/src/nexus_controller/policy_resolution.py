"""Resolving *which* policy this xApp serves, from the registry rather than from a filename.

Stage I's deployment rule: the controller loads a policy by **version identifier**, not by a path to
``final_model.zip``. The difference is not cosmetic. A path answers "what file is this"; a version answers
"which policy is running, which run produced it, from which commit, against which MDP contract, and what
did it score" — and those are the questions asked when something is wrong at an awkward hour.

Configuration is an *intent* (``production``) or an exact pin (``v7``). The intent form is what makes
promotion and rollback a registry operation rather than a redeploy with an edited environment variable.

This module refuses more than it loads, on purpose:

- a version whose MDP contract does not match this build is not served, because its action outputs mean
  something different rather than merely being worse
- a missing checkpoint is reported as such rather than silently falling back to a different version

Every refusal degrades to the static baseline and says why on ``/health``. A controller that will not
start because of a bad pin takes the whole system down when running unmanaged would have been fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nexus_errors.base_error import error_to_string
from nexus_errors.config_error import ConfigurationError, MdpVersionMismatchError
from nexus_errors.control_error import PolicyLoadError
from nexus_mdp.version import MDP_VERSION
from nexus_registry.paths import checkpoint_path, default_registry_root
from nexus_registry.record import PolicyVersion
from nexus_registry.store import PolicyRegistry
from nexus_telemetry.client import get_telemetry

_telemetry = get_telemetry("nexus.controller.resolution")


@dataclass(frozen=True, slots=True)
class ResolvedPolicy:
    """A version the controller is willing to load, and everything needed to describe it afterwards."""

    version: PolicyVersion
    checkpoint: Path

    @property
    def label(self) -> str:
        """What ``/health`` reports as the serving allocator.

        Carries the version id rather than a file stem, so the answer to "what is running" is something
        that can be looked up in the ledger.
        """
        provenance = self.version.provenance
        return f"{provenance.algorithm}:{self.version.version_id}@mdp{provenance.mdp_version}"

    @property
    def provenance_summary(self) -> dict[str, object]:
        """Provenance in the shape ``/health`` serves it."""
        score = self.version.latest_score
        return {
            "version_id": self.version.version_id,
            "stage": self.version.stage.value,
            "run_id": self.version.provenance.run_id,
            "git_sha": self.version.provenance.git_sha,
            "mdp_version": self.version.provenance.mdp_version,
            "algorithm": self.version.provenance.algorithm,
            "trained_on": self.version.provenance.traffic_pattern,
            "suite": f"{score.suite_name}@v{score.suite_version}" if score else None,
            "mean_p95_latency_ms": score.mean_p95_latency_ms if score else None,
            "seeds": score.seeds if score else None,
        }


def resolve_policy(
    identifier: str, registry_root: Path | None = None, runs_root: Path | None = None
) -> ResolvedPolicy:
    """Look up a policy version and check it is fit to serve.

    Raises :class:`PolicyLoadError` for anything that should degrade to the baseline. Callers survive it;
    see :func:`try_resolve_policy`.
    """
    registry = PolicyRegistry(registry_root or default_registry_root())

    try:
        version = registry.resolve(identifier)
    except ConfigurationError as error:
        raise PolicyLoadError(
            f"Could not resolve policy {identifier!r} from the registry: {error_to_string(error)}",
            {"identifier": identifier, "registry_root": str(registry.root)},
        ) from error

    if version.provenance.mdp_version != MDP_VERSION:
        # Deliberately not a silent skip to the next-best version. "Serve something" is the wrong
        # instinct here: the operator asked for a specific policy and needs to know it is unusable.
        mismatch = MdpVersionMismatchError(
            expected=MDP_VERSION,
            found=version.provenance.mdp_version,
            artifact=f"policy version {version.version_id}",
        )
        raise PolicyLoadError(str(mismatch), mismatch.context) from mismatch

    path = checkpoint_path(version.provenance.run_id, runs_root)
    if not path.exists():
        raise PolicyLoadError(
            f"Policy version {version.version_id} is registered but its checkpoint is missing at {path}. "
            f"Checkpoints are gitignored — the ledger travels between machines and the weights do not.",
            {"version_id": version.version_id, "checkpoint": str(path)},
        )

    return ResolvedPolicy(version=version, checkpoint=path)


def try_resolve_policy(
    identifier: str | None, registry_root: Path | None = None, runs_root: Path | None = None
) -> ResolvedPolicy | None:
    """Resolve if configured, returning ``None`` where the baseline is the right answer.

    ``None`` covers both "no policy configured" (a supported mode) and "the configured one cannot be
    served" (an incident). The two are distinguished in the logs and on ``/health`` — collapsing them is
    how a controller ends up quietly serving a baseline nobody asked for.
    """
    if not identifier:
        _telemetry.info(
            "No policy version configured — serving the static baseline. This is a supported mode, not "
            "an error. /health reports the active allocator.",
            {},
        )
        return None

    try:
        resolved = resolve_policy(identifier, registry_root, runs_root)
    except PolicyLoadError as error:
        _telemetry.capture_exception(
            error, f"Policy {identifier!r} could not be resolved — continuing on the static baseline"
        )
        return None

    _telemetry.info("Policy version resolved", resolved.provenance_summary)
    return resolved


def describe_resolution_failure(identifier: str | None) -> str:
    """Why ``/health`` is reporting a baseline."""
    if not identifier:
        return "NEXUS_POLICY_VERSION is unset — running the static baseline by configuration."
    return (
        f"Policy {identifier!r} did not resolve or did not load; see logs. The usual causes are a stale "
        f"MDP_VERSION, a checkpoint absent from experiments/runs/, or nothing promoted to that stage."
    )
