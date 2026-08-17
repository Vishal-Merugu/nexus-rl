"""Registering a run reads its artifact, and never guesses.

The registry's own guarantees are tested in `packages/nexus-registry/tests`. What is tested here is the
seam: turning a run directory on disk into provenance, and refusing when the artifact cannot support it.
A registry entry with a defaulted topology or seed describes a policy that does not exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nexus_errors.config_error import ConfigurationError
from nexus_rl.registry_cli import provenance_from_run

_CONFIG: dict[str, object] = {
    "action_mode": "discrete",
    "algorithm": "dqn",
    "episode_length": 512,
    "evaluation_patterns": ["diurnal", "flash_crowd"],
    "flow_count": 8,
    "git_sha": "abc1234",
    "mdp_version": 1,
    "name": "demo",
    "seed": 7,
    "topology": "mesh-6",
    "total_timesteps": 200_000,
    "traffic_pattern": "diurnal",
}


def _run(tmp_path: Path, **overrides: object) -> Path:
    runs_root = tmp_path / "runs"
    directory = runs_root / "demo-run"
    directory.mkdir(parents=True)
    config = {**_CONFIG, **overrides}
    for key, value in list(config.items()):
        if value is None:
            del config[key]
    (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return runs_root


def test_provenance_comes_from_the_run_artifact(tmp_path: Path) -> None:
    provenance = provenance_from_run("demo-run", _run(tmp_path))

    assert provenance.run_id == "demo-run"
    assert provenance.git_sha == "abc1234"
    assert provenance.seed == 7
    assert provenance.topology == "mesh-6"
    assert provenance.total_timesteps == 200_000


@pytest.mark.parametrize("missing", ["git_sha", "topology", "algorithm", "traffic_pattern"])
def test_a_run_missing_a_required_field_cannot_be_registered(tmp_path: Path, missing: str) -> None:
    # Defaulting any of these would put a policy in the ledger that nothing can reproduce or compare.
    with pytest.raises(ConfigurationError, match=f"no usable '{missing}'"):
        provenance_from_run("demo-run", _run(tmp_path, **{missing: None}))


@pytest.mark.parametrize("missing", ["seed", "flow_count", "mdp_version", "total_timesteps"])
def test_a_run_missing_a_required_number_cannot_be_registered(tmp_path: Path, missing: str) -> None:
    with pytest.raises(ConfigurationError, match=f"no usable '{missing}'"):
        provenance_from_run("demo-run", _run(tmp_path, **{missing: None}))


def test_a_run_that_was_never_written_names_the_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=r"No config\.yaml"):
        provenance_from_run("absent-run", tmp_path / "runs")


def test_a_boolean_is_not_accepted_where_a_number_is_required(tmp_path: Path) -> None:
    # bool is an int in Python. `seed: true` would otherwise register as seed 1 and reproduce nothing.
    with pytest.raises(ConfigurationError, match="no usable 'seed'"):
        provenance_from_run("demo-run", _run(tmp_path, seed=True))
