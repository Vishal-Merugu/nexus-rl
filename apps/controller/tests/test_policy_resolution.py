"""The xApp loads a policy by version, and refuses the ones it must not serve.

Stage I's deployment rule. The tests that matter are the refusals: each one is a way a controller could
otherwise end up serving something nobody chose, while ``/health`` reported it as intentional.

No checkpoint is loaded here — that needs torch. Resolution is the part with the decisions in it, and it
runs without the optional extra, which is also how it behaves in a slim controller image.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nexus_controller.policy_resolution import (
    describe_resolution_failure,
    resolve_policy,
    try_resolve_policy,
)
from nexus_errors.control_error import PolicyLoadError
from nexus_mdp.version import MDP_VERSION
from nexus_registry.record import EvaluationScore, Provenance, Stage
from nexus_registry.record import utc_now as _utc_now
from nexus_registry.store import PolicyRegistry


def _provenance(run_id: str = "run-a", mdp_version: int | None = None) -> Provenance:
    return Provenance(
        run_id=run_id,
        git_sha="abc1234",
        mdp_version=MDP_VERSION if mdp_version is None else mdp_version,
        algorithm="dqn",
        action_mode="discrete",
        topology="mesh-6",
        flow_count=8,
        traffic_pattern="diurnal",
        seed=0,
        total_timesteps=1000,
    )


def _score() -> EvaluationScore:
    return EvaluationScore(
        suite_name="standard",
        suite_version=1,
        seeds=5,
        scored_at=_utc_now(),
        mean_p95_latency_ms=9.5,
        mean_throughput_mbps=300.0,
        mean_fairness=0.9,
        total_dropped_mb=10.0,
    )


@pytest.fixture
def deployment(tmp_path: Path) -> tuple[Path, Path]:
    """A registry root and a runs root, with the checkpoint file present."""
    runs_root = tmp_path / "runs"
    (runs_root / "run-a").mkdir(parents=True)
    (runs_root / "run-a" / "policy.zip").write_bytes(b"not a real checkpoint")
    return tmp_path / "registry", runs_root


def _promoted(registry_root: Path, mdp_version: int | None = None) -> PolicyRegistry:
    registry = PolicyRegistry(registry_root)
    version = registry.register(_provenance(mdp_version=mdp_version))
    registry.record_score(version.version_id, _score())
    registry.promote(
        version.version_id,
        Stage.PRODUCTION,
        current_mdp_version=MDP_VERSION if mdp_version is None else mdp_version,
    )
    return registry


def test_a_stage_resolves_to_the_deployed_version(deployment: tuple[Path, Path]) -> None:
    registry_root, runs_root = deployment
    _promoted(registry_root)

    resolved = resolve_policy("production", registry_root, runs_root)

    assert resolved.version.version_id == "v1"
    assert resolved.checkpoint.exists()


def test_the_health_label_names_the_version_not_a_filename(deployment: tuple[Path, Path]) -> None:
    # "dqn:v1@mdp1" can be looked up in the ledger. "dqn:policy@mdp1" cannot, which is the whole problem.
    registry_root, runs_root = deployment
    _promoted(registry_root)

    assert resolve_policy("production", registry_root, runs_root).label == f"dqn:v1@mdp{MDP_VERSION}"


def test_provenance_answers_where_it_came_from_and_what_it_scored(
    deployment: tuple[Path, Path],
) -> None:
    registry_root, runs_root = deployment
    _promoted(registry_root)

    summary = resolve_policy("production", registry_root, runs_root).provenance_summary

    assert summary["run_id"] == "run-a"
    assert summary["git_sha"] == "abc1234"
    assert summary["suite"] == "standard@v1"
    assert summary["mean_p95_latency_ms"] == 9.5
    assert summary["seeds"] == 5


def test_an_exact_pin_resolves_as_well_as_a_stage(deployment: tuple[Path, Path]) -> None:
    registry_root, runs_root = deployment
    _promoted(registry_root)

    assert resolve_policy("v1", registry_root, runs_root).version.version_id == "v1"


# --- Refusals ---------------------------------------------------------------------------------------


def test_a_stale_mdp_version_is_refused_rather_than_served(deployment: tuple[Path, Path]) -> None:
    # Its action outputs mean something different under this build's contract, not merely something worse.
    registry_root, runs_root = deployment
    _promoted(registry_root, mdp_version=MDP_VERSION + 1)

    with pytest.raises(PolicyLoadError, match="MDP_VERSION"):
        resolve_policy("production", registry_root, runs_root)


def test_a_registered_version_with_no_checkpoint_says_so(tmp_path: Path) -> None:
    # The ledger is committed and the weights are not, so this is the normal state on a fresh clone. It
    # must read as a missing artifact, not as a corrupt registry.
    registry_root = tmp_path / "registry"
    _promoted(registry_root)

    with pytest.raises(PolicyLoadError, match="checkpoint is missing"):
        resolve_policy("production", registry_root, tmp_path / "runs")


def test_an_unclaimed_stage_does_not_silently_serve_something_else(
    deployment: tuple[Path, Path],
) -> None:
    registry_root, runs_root = deployment
    PolicyRegistry(registry_root).register(_provenance())

    with pytest.raises(PolicyLoadError, match="Could not resolve"):
        resolve_policy("production", registry_root, runs_root)


def test_an_unknown_version_id_is_refused(deployment: tuple[Path, Path]) -> None:
    registry_root, runs_root = deployment
    _promoted(registry_root)

    with pytest.raises(PolicyLoadError, match="Could not resolve"):
        resolve_policy("v99", registry_root, runs_root)


# --- Degrading to the baseline ----------------------------------------------------------------------


def test_no_configured_version_is_a_supported_mode_not_a_failure() -> None:
    assert try_resolve_policy(None) is None
    assert "unset" in describe_resolution_failure(None)


def test_a_broken_pin_degrades_to_the_baseline_instead_of_crashing(
    deployment: tuple[Path, Path],
) -> None:
    # A controller that refuses to start over a bad pin takes the whole system down, when running
    # unmanaged would have been fine.
    registry_root, runs_root = deployment

    assert try_resolve_policy("v99", registry_root, runs_root) is None


def test_the_two_baseline_causes_read_differently(deployment: tuple[Path, Path]) -> None:
    # "nothing configured" needs no action; "configured but unservable" is an incident. Health must not
    # collapse them into one sentence.
    del deployment
    assert describe_resolution_failure(None) != describe_resolution_failure("production")
    assert "did not resolve" in describe_resolution_failure("production")
