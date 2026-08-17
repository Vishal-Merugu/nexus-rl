"""The evaluation suite is a contract, so its loader refuses rather than defaults.

Every rejection here is a number that would otherwise reach a comparison table looking like a result: too
few seeds, a single traffic pattern, a misspelled pattern name that silently scores nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nexus_errors.config_error import ConfigurationError
from nexus_rl.suite import MINIMUM_SUITE_SEEDS, EvaluationSuite, load_suite

_VALID: dict[str, object] = {
    "name": "standard",
    "suite_version": 1,
    "topology": "mesh-6",
    "flow_count": 8,
    "steps": 64,
    "seeds": 3,
    "patterns": ["diurnal", "flash_crowd"],
    "baselines": ["static"],
}


def _write(tmp_path: Path, **overrides: object) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({**_VALID, **overrides}), encoding="utf-8")
    return path


def test_the_repo_suite_loads_and_is_the_one_scores_are_recorded_against(tmp_path: Path) -> None:
    del tmp_path
    suite = load_suite(Path("experiments/configs/evaluation-suite.yaml"))

    assert suite.name == "standard"
    assert suite.suite_version >= 1
    assert suite.seeds >= MINIMUM_SUITE_SEEDS
    assert "diurnal" in suite.patterns


def test_the_seed_set_is_fixed_not_sampled(tmp_path: Path) -> None:
    # Reproducibility is the point: two candidates must meet the same seeds, not the same *number* of them.
    suite = load_suite(_write(tmp_path, seeds=4))

    assert suite.seed_range == (0, 1, 2, 3)
    assert load_suite(_write(tmp_path, seeds=4)).seed_range == suite.seed_range


def test_a_suite_below_the_seed_floor_is_refused_at_load(tmp_path: Path) -> None:
    # Failing here rather than after minutes of scoring puts the error where the mistake was made.
    with pytest.raises(ConfigurationError, match="at least 3 seeds"):
        load_suite(_write(tmp_path, seeds=2))


def test_a_single_pattern_suite_is_refused(tmp_path: Path) -> None:
    # Scoring only where a policy trained measures memorisation.
    with pytest.raises(ConfigurationError, match="at least 2 traffic patterns"):
        load_suite(_write(tmp_path, patterns=["diurnal"]))


def test_an_unknown_pattern_is_caught_before_any_compute(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Unknown traffic pattern"):
        load_suite(_write(tmp_path, patterns=["diurnal", "not_a_pattern"]))


def test_an_unknown_baseline_is_caught_before_any_compute(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Unknown baseline"):
        load_suite(_write(tmp_path, baselines=["static", "magic"]))


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    # A silently dropped key is how a suite ends up not scoring what its file says it scores.
    with pytest.raises(ConfigurationError, match="Unknown suite keys"):
        load_suite(_write(tmp_path, sneaky_typo=1))


def test_a_missing_suite_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="No evaluation suite"):
        load_suite(tmp_path / "absent.yaml")


def test_degenerate_dimensions_are_refused_at_construction() -> None:
    with pytest.raises(ConfigurationError, match="must be positive"):
        EvaluationSuite(
            name="s",
            suite_version=1,
            topology="mesh-6",
            flow_count=8,
            steps=0,
            seeds=3,
            patterns=("diurnal", "bursty"),
            baselines=("static",),
        )
