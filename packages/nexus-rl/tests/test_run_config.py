from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nexus_errors.config_error import ConfigurationError, MdpVersionMismatchError
from nexus_rl.artifacts import assert_mdp_version, find_runs_with_mdp_version, new_run_id
from nexus_rl.run_config import RunConfig, load_run_config, write_run_config


def _config(**overrides: object) -> RunConfig:
    base: dict[str, object] = {
        "name": "dqn-test",
        "algorithm": "dqn",
        "seed": 0,
        "total_timesteps": 1000,
    }
    base.update(overrides)
    return RunConfig(**base)  # type: ignore[arg-type]


def test_config_round_trips_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump({"name": "dqn-test", "algorithm": "dqn", "seed": 3, "total_timesteps": 5000}),
        encoding="utf-8",
    )

    loaded = load_run_config(path)

    assert loaded.name == "dqn-test"
    assert loaded.seed == 3


def test_unknown_config_keys_are_rejected_rather_than_ignored(tmp_path: Path) -> None:
    # A silently ignored key is how a run ends up not doing what its config says.
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "typo",
                "algorithm": "dqn",
                "seed": 0,
                "total_timesteps": 100,
                "leaning_rate": 0.01,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Unknown config keys"):
        load_run_config(path)


def test_missing_config_file_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_run_config(tmp_path / "absent.yaml")


def test_dqn_with_a_continuous_action_space_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="DQN"):
        _config(action_mode="continuous")


def test_zero_timestep_budget_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        _config(total_timesteps=0)


def test_evaluating_only_on_the_training_pattern_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="at least two patterns"):
        _config(traffic_pattern="diurnal", evaluation_patterns=("diurnal",))


def test_held_out_patterns_exclude_the_training_pattern() -> None:
    config = _config(traffic_pattern="diurnal", evaluation_patterns=("diurnal", "bursty", "flash_crowd"))

    assert config.held_out_patterns == ("bursty", "flash_crowd")


def test_written_config_records_the_sha_and_mdp_version(tmp_path: Path) -> None:
    write_run_config(_config(), tmp_path, git_sha="abc123", mdp_version=4)

    written = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))

    assert written["git_sha"] == "abc123"
    assert written["mdp_version"] == 4


def test_a_stale_checkpoint_is_refused_rather_than_silently_used(tmp_path: Path) -> None:
    write_run_config(_config(), tmp_path, git_sha="abc123", mdp_version=2)

    with pytest.raises(MdpVersionMismatchError):
        assert_mdp_version(tmp_path, current_version=3)


def test_a_matching_mdp_version_passes(tmp_path: Path) -> None:
    write_run_config(_config(), tmp_path, git_sha="abc123", mdp_version=3)

    assert_mdp_version(tmp_path, current_version=3)


def test_a_run_without_a_recorded_mdp_version_is_not_trusted(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"name": "legacy"}), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="records no MDP version"):
        assert_mdp_version(tmp_path, current_version=1)


def test_runs_can_be_listed_by_mdp_version(tmp_path: Path) -> None:
    write_run_config(_config(name="old"), tmp_path / "old-run", git_sha="a", mdp_version=1)
    write_run_config(_config(name="new"), tmp_path / "new-run", git_sha="b", mdp_version=2)

    assert find_runs_with_mdp_version(1, root=tmp_path) == ["old-run"]


def test_listing_runs_ignores_directories_without_a_config(tmp_path: Path) -> None:
    (tmp_path / "junk").mkdir()

    assert find_runs_with_mdp_version(1, root=tmp_path) == []


def test_run_ids_are_prefixed_by_the_config_name() -> None:
    assert new_run_id("dqn-baseline").startswith("dqn-baseline-")


def test_an_empty_evaluation_set_is_refused() -> None:
    # The guard returned early whenever the training pattern was absent from the list, which let a config
    # that evaluates on nothing at all pass validation.
    with pytest.raises(ConfigurationError, match="at least two patterns"):
        _config(traffic_pattern="diurnal", evaluation_patterns=())


def test_a_single_held_out_pattern_is_still_too_few() -> None:
    with pytest.raises(ConfigurationError, match="at least two patterns"):
        _config(traffic_pattern="diurnal", evaluation_patterns=("bursty",))


def test_evaluating_only_on_patterns_equal_to_training_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="generalisation cannot be measured"):
        _config(traffic_pattern="diurnal", evaluation_patterns=("diurnal", "diurnal"))


def test_a_config_with_a_genuine_hold_out_is_accepted() -> None:
    config = _config(traffic_pattern="diurnal", evaluation_patterns=("diurnal", "flash_crowd"))

    assert config.held_out_patterns == ("flash_crowd",)


@pytest.mark.parametrize("length", [0, -10])
def test_a_non_positive_episode_length_is_refused(length: int) -> None:
    with pytest.raises(ConfigurationError, match="episode_length must be positive"):
        _config(episode_length=length)


@pytest.mark.parametrize("count", [0, -3])
def test_a_non_positive_flow_count_is_refused(count: int) -> None:
    with pytest.raises(ConfigurationError, match="flow_count must be positive"):
        _config(flow_count=count)


def test_run_artifacts_resolve_against_the_project_root_not_the_cwd(tmp_path: Path) -> None:
    # A CWD-relative artifacts root scatters runs across whatever directory the command was invoked from,
    # and a run nobody can find is a run that did not happen.
    from nexus_rl.artifacts import RUNS_ROOT

    assert RUNS_ROOT.is_absolute()
    assert RUNS_ROOT.parts[-2:] == ("experiments", "runs")
