"""Run artifacts — where a run's config, metrics and checkpoints live.

Layout::

    experiments/runs/<run-id>/
        config.yaml      # settings + seed + git SHA + MDP_VERSION
        metrics.jsonl    # one record per timestep, reward components named
        policy.zip       # checkpoint (gitignored)

Everything under ``experiments/runs/`` is gitignored. Preserve *numbers* in
``docs/evaluation/results.md``, not weights: checkpoints are large, binary, and invalidated by the next
``MDP_VERSION`` bump.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from nexus_errors.config_error import ConfigurationError, MdpVersionMismatchError


def project_root() -> Path:
    """Walk up to the directory holding the workspace ``pyproject.toml``.

    Run artifacts must land in the same place regardless of where the command was invoked. A CWD-relative
    path silently scatters runs across whatever directory the user happened to be in, and a run nobody can
    find is a run that did not happen.
    """
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "packages").is_dir():
            return candidate
    return Path.cwd()


RUNS_ROOT = project_root() / "experiments" / "runs"


def new_run_id(name: str) -> str:
    """A sortable, human-readable run id: ``<name>-<utc timestamp>``."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{name}-{stamp}"


def run_dir(run_id: str, root: Path = RUNS_ROOT) -> Path:
    return root / run_id


def current_git_sha() -> str:
    """The commit a run was launched from.

    Returns ``"unknown"`` outside a git repo, and marks a dirty tree explicitly — a run launched from
    uncommitted changes is not reproducible, and the artifact should say so rather than imply otherwise.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def read_run_metadata(directory: Path) -> dict[str, object]:
    """Read a run's config snapshot."""
    path = directory / "config.yaml"
    if not path.exists():
        raise ConfigurationError(f"No config.yaml in {directory}.", {"run_dir": str(directory)})
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Malformed config.yaml in {directory}.", {"run_dir": str(directory)})
    return loaded


def assert_mdp_version(directory: Path, current_version: int) -> None:
    """Refuse to use an artifact produced under a different MDP version.

    A policy trained against one observation/action/reward contract is meaningless under another. Failing
    loudly here is the difference between a broken command and a silently wrong number in a writeup.
    """
    metadata = read_run_metadata(directory)
    recorded = metadata.get("mdp_version")
    if not isinstance(recorded, int):
        raise ConfigurationError(
            f"Run {directory.name} records no MDP version, so its results cannot be trusted.",
            {"run_dir": str(directory)},
        )
    if recorded != current_version:
        raise MdpVersionMismatchError(expected=current_version, found=recorded, artifact=str(directory))


def find_runs_with_mdp_version(version: int, root: Path = RUNS_ROOT) -> list[str]:
    """Every run recorded under a given MDP version.

    Used when bumping ``MDP_VERSION`` to list exactly which checkpoints the change invalidates — which is
    what makes that bump a deliberate act rather than silent drift.
    """
    if not root.exists():
        return []
    matching: list[str] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        try:
            metadata = read_run_metadata(directory)
        except ConfigurationError:
            continue
        if metadata.get("mdp_version") == version:
            matching.append(directory.name)
    return matching
