"""``nexus-registry`` — register, score, promote and inspect policy versions.

The operator-facing half of Stage I. Everything here is a thin shell over ``nexus_registry``: read the run
artifacts, hand the ledger a record, print what happened. The refusals that matter (stale MDP version,
promotion on too few seeds, duplicate registration) live in the registry itself so they hold no matter who
is calling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nexus_errors.base_error import NexusError
from nexus_errors.config_error import ConfigurationError
from nexus_mdp.version import MDP_VERSION
from nexus_registry.paths import default_registry_root
from nexus_registry.record import PolicyVersion, Provenance, Stage
from nexus_registry.store import PolicyRegistry
from nexus_rl.artifacts import RUNS_ROOT, read_run_metadata, run_dir
from nexus_rl.policy_loader import ActionMode, load_policy_predictor
from nexus_rl.suite import load_suite, score_policy
from nexus_telemetry.client import get_telemetry

_telemetry = get_telemetry("nexus.registry")

DEFAULT_SUITE_PATH = Path("experiments/configs/evaluation-suite.yaml")


def provenance_from_run(run_id: str, runs_root: Path) -> Provenance:
    """Build a provenance record from a run's own config snapshot.

    Every field is read from the artifact, never defaulted. A registry entry whose topology or seed was
    guessed describes a policy that does not exist.
    """
    metadata = read_run_metadata(run_dir(run_id, runs_root))

    def text(key: str) -> str:
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigurationError(
                f"Run {run_id!r} records no usable {key!r}; it cannot be registered without it.",
                {"run_id": run_id, key: repr(value)},
            )
        return value

    def number(key: str) -> int:
        value = metadata.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigurationError(
                f"Run {run_id!r} records no usable {key!r}; it cannot be registered without it.",
                {"run_id": run_id, key: repr(value)},
            )
        return value

    return Provenance(
        run_id=run_id,
        git_sha=text("git_sha"),
        mdp_version=number("mdp_version"),
        algorithm=text("algorithm"),
        action_mode=text("action_mode"),
        topology=text("topology"),
        flow_count=number("flow_count"),
        traffic_pattern=text("traffic_pattern"),
        seed=number("seed"),
        total_timesteps=number("total_timesteps"),
    )


def _action_mode(value: str, run_id: str) -> ActionMode:
    # Compared literal by literal so the return narrows properly rather than staying a plain str.
    if value == "discrete":
        return "discrete"
    if value == "continuous":
        return "continuous"
    raise ConfigurationError(f"Run {run_id!r} records an unknown action_mode {value!r}.", {"run_id": run_id})


def _render(version: PolicyVersion) -> str:
    return version.describe()


def main() -> None:
    """Entry point. Turns a refusal into a message and an exit code, never a traceback.

    The registry's refusals — unscored, too few seeds, stale MDP version — are *expected outcomes*, not
    crashes. A stack trace buries the one sentence explaining what to do next under twenty lines of
    frames, which is the CLI equivalent of a service returning an unhandled 500. Same rule as the FastAPI
    handlers and the MCP tools: a ``NexusError`` is mapped at the boundary.
    """
    try:
        _dispatch()
    except NexusError as error:
        _telemetry.capture_exception(error, "Registry command failed")
        print(f"error: {error}", file=sys.stderr)  # noqa: T201  — the CLI's output is the deliverable
        raise SystemExit(1) from error


def _dispatch() -> None:
    parser = argparse.ArgumentParser(description="Model registry for trained policies.")
    parser.add_argument("--registry-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register", help="Register a completed training run.")
    register.add_argument("run_id")
    register.add_argument("--notes", default="")

    score = subparsers.add_parser("score", help="Score a version against the evaluation suite.")
    score.add_argument("version_id")
    score.add_argument("--suite", type=Path, default=DEFAULT_SUITE_PATH)

    promote = subparsers.add_parser("promote", help="Move a version to a deployment stage.")
    promote.add_argument("version_id")
    promote.add_argument("--stage", default=Stage.PRODUCTION.value, choices=[s.value for s in Stage])
    promote.add_argument(
        "--force",
        action="store_true",
        help="Deploy despite a missing or low-seed score. Deliberate override, recorded in the log.",
    )

    subparsers.add_parser("list", help="Every registered version.")
    show = subparsers.add_parser("show", help="One version, by id or by stage.")
    show.add_argument("identifier")

    args = parser.parse_args()
    registry = PolicyRegistry(args.registry_root or default_registry_root())

    if args.command == "register":
        version = registry.register(provenance_from_run(args.run_id, args.runs_root), notes=args.notes)
        _telemetry.info("Policy registered", {"version_id": version.version_id, "run_id": args.run_id})
        print(_render(version))  # noqa: T201  — the CLI's output is the deliverable
        return

    if args.command == "score":
        version = registry.get(args.version_id)
        suite = load_suite(args.suite)
        predictor = load_policy_predictor(
            checkpoint=run_dir(version.provenance.run_id, args.runs_root) / "policy.zip",
            algorithm=version.provenance.algorithm,
            action_mode=_action_mode(version.provenance.action_mode, version.provenance.run_id),
        )
        result = score_policy(predictor, suite, label=f"policy:{version.version_id}")
        updated = registry.record_score(version.version_id, result)
        _telemetry.info(
            "Policy scored",
            {
                "version_id": version.version_id,
                "suite": f"{suite.name}@v{suite.suite_version}",
                "mean_p95_latency_ms": result.mean_p95_latency_ms,
                "per_pattern": result.per_pattern,
            },
        )
        print(_render(updated))  # noqa: T201
        return

    if args.command == "promote":
        version = registry.promote(
            args.version_id,
            Stage(args.stage),
            current_mdp_version=MDP_VERSION,
            force=args.force,
        )
        _telemetry.info(
            "Policy promoted",
            {"version_id": version.version_id, "stage": version.stage.value, "forced": args.force},
        )
        print(_render(version))  # noqa: T201
        return

    if args.command == "list":
        versions = registry.list_versions()
        if not versions:
            print("No policy versions registered.")  # noqa: T201
            return
        for version in versions:
            print(_render(version))  # noqa: T201
        return

    print(_render(registry.resolve(args.identifier)))  # noqa: T201


if __name__ == "__main__":
    main()
