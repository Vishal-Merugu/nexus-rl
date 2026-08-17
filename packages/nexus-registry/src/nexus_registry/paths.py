"""Where the ledger lives.

Resolved in **one** place, called by both the training tooling that writes it and the controller that
reads it. Two components computing this independently is the failure `tasks/lessons.md` records under
"two services must never derive shared state independently" — here it would surface as a controller that
reports "no policy deployed" while the registry two directories over says otherwise.

Deliberately a function, not a module-level constant: resolving it touches the filesystem, and packages in
this repo do no I/O at import time.
"""

from __future__ import annotations

from pathlib import Path

REGISTRY_DIRNAME = "registry"
RUNS_DIRNAME = "runs"
CHECKPOINT_FILENAME = "policy.zip"


def project_root(start: Path | None = None) -> Path:
    """Walk up to the directory holding the workspace ``pyproject.toml``.

    The ledger must land in the same place regardless of where a command was invoked. A CWD-relative path
    scatters registries across whatever directory the user happened to be in, and a registry nobody can
    find is a registry that does not exist.
    """
    origin = start or Path.cwd()
    for candidate in [origin, *origin.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "packages").is_dir():
            return candidate
    return origin


def default_registry_root(start: Path | None = None) -> Path:
    """``<project>/experiments/registry`` — committed, unlike ``experiments/runs``.

    The ledger is a few hundred bytes of provenance per policy and is the answer to "what is deployed and
    where did it come from". Gitignoring it alongside the checkpoints would mean that answer dies with the
    machine that produced it. Weights stay gitignored; metadata does not.
    """
    return project_root(start) / "experiments" / REGISTRY_DIRNAME


def default_runs_root(start: Path | None = None) -> Path:
    """``<project>/experiments/runs`` — where checkpoints and metrics live. Gitignored."""
    return project_root(start) / "experiments" / RUNS_DIRNAME


def checkpoint_path(run_id: str, runs_root: Path | None = None) -> Path:
    """Where a registered version's weights are.

    Addressing lives here rather than in the training harness so the **controller can resolve a policy
    version without depending on the training package** — which would drag gymnasium and Stable-Baselines3
    into a service that only needs to load a file. Same reasoning as the nexus-mdp / nexus-env split
    (ADR-0007).
    """
    return (runs_root or default_runs_root()) / run_id / CHECKPOINT_FILENAME
