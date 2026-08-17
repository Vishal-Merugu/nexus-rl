"""The registry as a ledger, not a filename table.

The Stage I claim is "which policy is deployed, where did it come from, and what did it score?" — so the
tests are about the *guarantees* that make those answers trustworthy: provenance cannot be missing, a
promotion cannot happen on an anecdote or a stale contract, and exactly one version can claim a stage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nexus_errors.config_error import ConfigurationError, MdpVersionMismatchError
from nexus_registry.record import EvaluationScore, Provenance, Stage
from nexus_registry.store import PolicyRegistry

MDP = 3


def _provenance(run_id: str = "dqn-20260817-101500", mdp_version: int = MDP) -> Provenance:
    return Provenance(
        run_id=run_id,
        git_sha="abc1234",
        mdp_version=mdp_version,
        algorithm="dqn",
        action_mode="discrete",
        topology="mesh-6",
        flow_count=8,
        traffic_pattern="diurnal",
        seed=0,
        total_timesteps=200_000,
    )


def _score(seeds: int = 5, suite_version: int = 1) -> EvaluationScore:
    return EvaluationScore(
        suite_name="standard",
        suite_version=suite_version,
        seeds=seeds,
        scored_at=datetime(2026, 8, 17, tzinfo=UTC),
        mean_p95_latency_ms=12.4,
        mean_throughput_mbps=310.0,
        mean_fairness=0.91,
        total_dropped_mb=44.0,
        per_pattern={"diurnal": 11.0, "flash_crowd": 13.8},
    )


@pytest.fixture
def registry(tmp_path: Path) -> PolicyRegistry:
    return PolicyRegistry(tmp_path / "registry")


# --- Identity and provenance ------------------------------------------------------------------------


def test_versions_are_addressable_and_numbered_by_the_registry(registry: PolicyRegistry) -> None:
    first = registry.register(_provenance("run-a"))
    second = registry.register(_provenance("run-b"))

    assert (first.version_id, second.version_id) == ("v1", "v2")
    assert registry.get("v1").provenance.run_id == "run-a"


def test_a_run_cannot_be_registered_twice(registry: PolicyRegistry) -> None:
    # Two ids for one checkpoint would let the same policy be compared against itself and reported as two.
    registry.register(_provenance("run-a"))

    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register(_provenance("run-a"))


def test_a_version_without_a_git_sha_cannot_exist(registry: PolicyRegistry) -> None:
    # `current_git_sha` returns "unknown" outside a repo. Registering that produces an artifact nothing
    # can reproduce, which is the one thing a provenance ledger must not contain.
    with pytest.raises(ConfigurationError, match="git SHA"):
        Provenance(
            run_id="run-a",
            git_sha="unknown",
            mdp_version=MDP,
            algorithm="dqn",
            action_mode="discrete",
            topology="mesh-6",
            flow_count=8,
            traffic_pattern="diurnal",
            seed=0,
            total_timesteps=1000,
        )


def test_a_dirty_tree_is_recorded_rather_than_rejected(registry: PolicyRegistry) -> None:
    # You will register runs from a dirty tree while iterating. The ledger says so instead of hiding it.
    version = registry.register(_provenance().model_copy(update={"git_sha": "abc1234-dirty"}))

    assert "dirty" in registry.get(version.version_id).provenance.git_sha


def test_the_description_answers_all_three_stage_i_questions(registry: PolicyRegistry) -> None:
    version = registry.record_score(registry.register(_provenance()).version_id, _score())
    promoted = registry.promote(version.version_id, Stage.PRODUCTION, current_mdp_version=MDP)

    described = promoted.describe()
    assert "production" in described  # which policy is deployed
    assert "dqn-20260817-101500" in described  # where it came from
    assert "abc1234" in described
    assert "p95" in described  # what it scored
    assert "5 seeds" in described


# --- Scores -----------------------------------------------------------------------------------------


def test_rescoring_appends_and_never_overwrites(registry: PolicyRegistry) -> None:
    # The score history is the evidence for a promotion or a retirement. Overwriting destroys it.
    version = registry.register(_provenance())
    registry.record_score(version.version_id, _score(seeds=3, suite_version=1))
    updated = registry.record_score(version.version_id, _score(seeds=5, suite_version=2))

    assert len(updated.scores) == 2
    assert updated.latest_score is not None
    assert updated.latest_score.suite_version == 2


def test_a_score_carries_the_suite_it_came_from(registry: PolicyRegistry) -> None:
    # A number detached from the scenario set that produced it is not comparable to any other number.
    version = registry.record_score(registry.register(_provenance()).version_id, _score())

    assert version.scores[0].suite_name == "standard"
    assert version.scores[0].per_pattern["flash_crowd"] == 13.8


# --- Promotion --------------------------------------------------------------------------------------


def test_promotion_requires_an_evaluation(registry: PolicyRegistry) -> None:
    version = registry.register(_provenance())

    with pytest.raises(ConfigurationError, match="never been evaluated"):
        registry.promote(version.version_id, Stage.PRODUCTION, current_mdp_version=MDP)


def test_promotion_refuses_a_single_seed_anecdote(registry: PolicyRegistry) -> None:
    version = registry.record_score(registry.register(_provenance()).version_id, _score(seeds=1))

    with pytest.raises(ConfigurationError, match="Seed variance"):
        registry.promote(version.version_id, Stage.PRODUCTION, current_mdp_version=MDP)


def test_a_deliberate_override_is_possible_but_explicit(registry: PolicyRegistry) -> None:
    version = registry.record_score(registry.register(_provenance()).version_id, _score(seeds=1))

    promoted = registry.promote(version.version_id, Stage.PRODUCTION, current_mdp_version=MDP, force=True)

    assert promoted.stage is Stage.PRODUCTION


def test_a_stale_mdp_version_cannot_be_deployed(registry: PolicyRegistry) -> None:
    # Not merely a worse policy — its action outputs mean something different under another contract.
    version = registry.record_score(registry.register(_provenance(mdp_version=MDP - 1)).version_id, _score())

    with pytest.raises(MdpVersionMismatchError):
        registry.promote(version.version_id, Stage.PRODUCTION, current_mdp_version=MDP)


def test_a_stale_mdp_version_can_still_be_archived(registry: PolicyRegistry) -> None:
    # Archiving is a bookkeeping act, not a deployment claim. Blocking it would make stale versions
    # permanently un-tidyable.
    version = registry.register(_provenance(mdp_version=MDP - 1))

    archived = registry.promote(version.version_id, Stage.ARCHIVED, current_mdp_version=MDP)

    assert archived.stage is Stage.ARCHIVED


def test_promoting_demotes_the_incumbent_to_archived_not_to_nothing(registry: PolicyRegistry) -> None:
    # An archived incumbent stays findable, which is what makes it a rollback target.
    old = registry.record_score(registry.register(_provenance("run-a")).version_id, _score())
    registry.promote(old.version_id, Stage.PRODUCTION, current_mdp_version=MDP)
    new = registry.record_score(registry.register(_provenance("run-b")).version_id, _score())

    registry.promote(new.version_id, Stage.PRODUCTION, current_mdp_version=MDP)

    assert registry.get(old.version_id).stage is Stage.ARCHIVED
    assert registry.production() is not None
    assert registry.production().version_id == new.version_id  # type: ignore[union-attr]


def test_rolling_back_is_promoting_the_archived_version_again(registry: PolicyRegistry) -> None:
    old = registry.record_score(registry.register(_provenance("run-a")).version_id, _score())
    new = registry.record_score(registry.register(_provenance("run-b")).version_id, _score())
    registry.promote(old.version_id, Stage.PRODUCTION, current_mdp_version=MDP)
    registry.promote(new.version_id, Stage.PRODUCTION, current_mdp_version=MDP)

    registry.promote(old.version_id, Stage.PRODUCTION, current_mdp_version=MDP)

    assert registry.production() is not None
    assert registry.production().version_id == old.version_id  # type: ignore[union-attr]
    assert registry.get(new.version_id).stage is Stage.ARCHIVED


def test_shadow_and_production_are_independent_stages(registry: PolicyRegistry) -> None:
    # A candidate must be able to shadow the incumbent without displacing it — that is the whole point.
    incumbent = registry.record_score(registry.register(_provenance("run-a")).version_id, _score())
    candidate = registry.record_score(registry.register(_provenance("run-b")).version_id, _score())
    registry.promote(incumbent.version_id, Stage.PRODUCTION, current_mdp_version=MDP)

    registry.promote(candidate.version_id, Stage.SHADOW, current_mdp_version=MDP)

    assert registry.production() is not None
    assert registry.production().version_id == incumbent.version_id  # type: ignore[union-attr]
    assert registry.resolve("shadow").version_id == candidate.version_id


# --- Resolution -------------------------------------------------------------------------------------


def test_a_stage_resolves_to_the_version_claiming_it(registry: PolicyRegistry) -> None:
    version = registry.record_score(registry.register(_provenance()).version_id, _score())
    registry.promote(version.version_id, Stage.PRODUCTION, current_mdp_version=MDP)

    assert registry.resolve("production").version_id == version.version_id
    assert registry.resolve(version.version_id).version_id == version.version_id


def test_resolving_an_unclaimed_stage_raises_rather_than_guessing(registry: PolicyRegistry) -> None:
    registry.register(_provenance())

    with pytest.raises(ConfigurationError, match="No policy version is at stage"):
        registry.resolve("production")


def test_nothing_deployed_is_a_valid_state_not_an_error(registry: PolicyRegistry) -> None:
    registry.register(_provenance())

    assert registry.production() is None


def test_an_unknown_version_names_the_ones_that_exist(registry: PolicyRegistry) -> None:
    registry.register(_provenance())

    with pytest.raises(ConfigurationError) as excinfo:
        registry.get("v99")

    assert excinfo.value.context["known"] == ["v1"]


def test_versions_sort_numerically_not_lexically(registry: PolicyRegistry) -> None:
    # v10 must not sort before v2 — the next assigned id is derived from this ordering.
    for index in range(11):
        registry.register(_provenance(f"run-{index}"))

    assert [record.version_id for record in registry.list_versions()][-1] == "v11"


# --- Ledger integrity -------------------------------------------------------------------------------


def test_an_empty_registry_reads_as_empty_rather_than_failing(tmp_path: Path) -> None:
    assert PolicyRegistry(tmp_path / "nothing-here").list_versions() == ()


def test_a_record_from_a_newer_schema_is_refused_not_half_read(registry: PolicyRegistry) -> None:
    version = registry.register(_provenance())
    path = registry.root / "versions" / f"{version.version_id}.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"schema_version": 1', '"schema_version": 99'))

    with pytest.raises(ConfigurationError, match="schema version 99"):
        registry.get(version.version_id)


def test_a_malformed_record_refuses_to_load_rather_than_serving_partial_provenance(
    registry: PolicyRegistry,
) -> None:
    version = registry.register(_provenance())
    path = registry.root / "versions" / f"{version.version_id}.json"
    path.write_text('{"schema_version": 1, "version_id": "v1"}', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="malformed"):
        registry.get(version.version_id)


def test_two_versions_claiming_one_stage_is_refused_not_arbitrated(registry: PolicyRegistry) -> None:
    # Only reachable by hand-editing the ledger. Picking one would serve an arbitrary policy while
    # reporting a definite stage.
    first = registry.record_score(registry.register(_provenance("run-a")).version_id, _score())
    second = registry.record_score(registry.register(_provenance("run-b")).version_id, _score())
    for record in (first, second):
        path = registry.root / "versions" / f"{record.version_id}.json"
        path.write_text(path.read_text(encoding="utf-8").replace('"stage": "none"', '"stage": "production"'))

    with pytest.raises(ConfigurationError, match="inconsistent"):
        registry.resolve("production")


def test_the_ledger_survives_a_round_trip_through_disk(tmp_path: Path) -> None:
    # The controller reads this ledger in a different process from the one that wrote it.
    written = PolicyRegistry(tmp_path / "r")
    version = written.record_score(written.register(_provenance()).version_id, _score())
    written.promote(version.version_id, Stage.PRODUCTION, current_mdp_version=MDP)

    reread = PolicyRegistry(tmp_path / "r").resolve("production")

    assert reread.version_id == version.version_id
    assert reread.provenance.git_sha == "abc1234"
    assert reread.latest_score is not None
    assert reread.latest_score.mean_p95_latency_ms == 12.4
