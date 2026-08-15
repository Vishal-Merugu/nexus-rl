"""Evaluation entry point — ``nexus-eval``.

Enforces the four-part standard in code rather than in a checklist: a comparison always runs multiple seeds,
multiple traffic patterns, and every named baseline. Producing a single-seed, single-pattern number is not
something this tool will do for you.

Baseline-only evaluation runs without torch, so Stage A results are reachable before any training has
happened.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from nexus_baselines.allocator import available_baselines, build_baseline
from nexus_errors.config_error import ConfigurationError
from nexus_mdp.reward import RewardWeights
from nexus_mdp.version import MDP_VERSION
from nexus_rl.artifacts import RUNS_ROOT, assert_mdp_version, read_run_metadata, run_dir
from nexus_rl.policy_loader import ActionMode, load_policy_predictor
from nexus_rl.rollout import PolicyPredictor, RolloutSummary, rollout_allocator, rollout_policy
from nexus_sim.simulator import NetworkSimulator
from nexus_telemetry.client import get_telemetry

_telemetry = get_telemetry("nexus.eval")

MINIMUM_SEEDS = 3
"""Below this, seed variance routinely exceeds the effect being measured and the comparison is an anecdote."""


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """One allocator's aggregated result on one traffic pattern, across seeds."""

    label: str
    traffic_pattern: str
    seeds: int
    p95_latency_ms: tuple[float, float]
    """(mean, standard deviation). Both, always — a mean without spread hides whether the effect is real."""
    throughput_mbps: tuple[float, float]
    fairness: tuple[float, float]
    dropped_mb: tuple[float, float]

    def as_markdown_cells(self) -> list[str]:
        return [
            self.label,
            self.traffic_pattern,
            str(self.seeds),
            f"{self.p95_latency_ms[0]:.1f} ± {self.p95_latency_ms[1]:.1f}",
            f"{self.throughput_mbps[0]:.1f} ± {self.throughput_mbps[1]:.1f}",
            f"{self.fairness[0]:.3f} ± {self.fairness[1]:.3f}",
            f"{self.dropped_mb[0]:.1f} ± {self.dropped_mb[1]:.1f}",
        ]


def evaluate_baseline(
    baseline_name: str,
    traffic_pattern: str,
    seeds: tuple[int, ...],
    steps: int,
    topology: str = "mesh-6",
    flow_count: int = 8,
) -> ComparisonRow:
    """Run one baseline across seeds on one traffic pattern."""
    summaries: list[RolloutSummary] = []
    for seed in seeds:
        simulator = NetworkSimulator.from_config(topology, traffic_pattern, seed=seed, flow_count=flow_count)
        allocator = build_baseline(baseline_name, simulator.topology, flow_count)
        summaries.append(
            rollout_allocator(
                simulator=simulator,
                allocator=allocator,
                steps=steps,
                seed=seed,
                traffic_pattern=traffic_pattern,
                reward_weights=RewardWeights(),
            )
        )
    return _aggregate(baseline_name, traffic_pattern, summaries)


def evaluate_policy(
    predictor: PolicyPredictor,
    label: str,
    traffic_pattern: str,
    seeds: tuple[int, ...],
    steps: int,
    topology: str = "mesh-6",
    flow_count: int = 8,
) -> ComparisonRow:
    """Run a trained policy across seeds on one traffic pattern.

    Deliberately identical in shape to :func:`evaluate_baseline` — same simulator construction, same
    seeds, same steps, same aggregation. The comparison is only meaningful because the two differ in
    nothing but the allocator.
    """
    summaries: list[RolloutSummary] = []
    for seed in seeds:
        simulator = NetworkSimulator.from_config(topology, traffic_pattern, seed=seed, flow_count=flow_count)
        summaries.append(
            rollout_policy(
                simulator=simulator,
                predictor=predictor,
                steps=steps,
                seed=seed,
                traffic_pattern=traffic_pattern,
                label=label,
                reward_weights=RewardWeights(),
            )
        )
    return _aggregate(label, traffic_pattern, summaries)


def _aggregate(label: str, traffic_pattern: str, summaries: list[RolloutSummary]) -> ComparisonRow:
    def spread(values: list[float]) -> tuple[float, float]:
        mean = statistics.fmean(values)
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, deviation

    return ComparisonRow(
        label=label,
        traffic_pattern=traffic_pattern,
        seeds=len(summaries),
        p95_latency_ms=spread([s.mean_p95_latency_ms for s in summaries]),
        throughput_mbps=spread([s.mean_throughput_mbps for s in summaries]),
        fairness=spread([s.mean_fairness for s in summaries]),
        dropped_mb=spread([s.total_dropped_mb for s in summaries]),
    )


def render_comparison(rows: list[ComparisonRow]) -> str:
    """Render the comparison as markdown, with the honesty caveats attached.

    The caveats are part of the output, not an afterthought. A table of numbers detached from what they do
    not show is how an otherwise strong project gets undermined by its own writeup.
    """
    header = (
        "| Allocator | Pattern | Seeds | p95 latency (ms) | Throughput (Mb/s) | Fairness | Dropped (Mb) |\n"
        "| --------- | ------- | ----- | ---------------- | ----------------- | -------- | ------------ |\n"
    )
    body = "".join("| " + " | ".join(row.as_markdown_cells()) + " |\n" for row in rows)

    seeds = max((row.seeds for row in rows), default=0)
    caveat = (
        f"\n_Flow-level simulation: aggregate rates and scalar queue backlogs, no packet traces; latency "
        f"derived analytically, not measured. Not validated against physical hardware. "
        f"MDP_VERSION {MDP_VERSION}, {seeds} seeds, mean ± standard deviation._\n"
    )
    if seeds < MINIMUM_SEEDS:
        caveat += (
            f"\n> **Not a result.** Fewer than {MINIMUM_SEEDS} seeds — seed variance in RL routinely "
            f"exceeds the effect being measured. Re-run with `--seeds {MINIMUM_SEEDS}` before citing "
            f"these numbers anywhere.\n"
        )
    return header + body + caveat


def _require_str(metadata: dict[str, object], key: str, run_id: str) -> str:
    """Read a required string from a run's config snapshot.

    Raises rather than defaulting: a missing ``topology`` silently falling back to ``mesh-6`` would
    evaluate the policy on a network it never trained on and report the result as if it were comparable.
    """
    value = metadata.get(key)
    if not isinstance(value, str):
        raise ConfigurationError(
            f"Run {run_id!r} records no usable {key!r}, so its policy cannot be evaluated comparably.",
            {"run_id": run_id, key: repr(value)},
        )
    return value


def _require_int(metadata: dict[str, object], key: str, run_id: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(
            f"Run {run_id!r} records no usable {key!r}, so its policy cannot be evaluated comparably.",
            {"run_id": run_id, key: repr(value)},
        )
    return value


def _read_action_mode(metadata: dict[str, object], run_id: str) -> ActionMode:
    mode = _require_str(metadata, "action_mode", run_id)
    # Compared literal by literal so the return actually narrows to ActionMode. A membership test
    # against a tuple leaves the value a plain `str`, which `cast()` would then have to paper over.
    if mode == "discrete":
        return "discrete"
    if mode == "continuous":
        return "continuous"
    raise ConfigurationError(f"Run {run_id!r} records an unknown action_mode {mode!r}.", {"run_id": run_id})


def _read_patterns(metadata: dict[str, object]) -> tuple[str, ...] | None:
    """The run's declared evaluation patterns, if it recorded a usable list."""
    patterns = metadata.get("evaluation_patterns")
    if not isinstance(patterns, list):
        return None
    named = tuple(item for item in patterns if isinstance(item, str))
    return named or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate policies and baselines. Always multi-seed, always multi-pattern."
    )
    parser.add_argument("--run-id", default=None, help="Trained run to evaluate. Omit for baselines only.")
    parser.add_argument(
        "--traffic-pattern",
        action="append",
        default=None,
        help="Repeatable. Defaults to diurnal + flash_crowd so an off-distribution pattern is included.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=None,
        help=f"Repeatable. Defaults to all of: {', '.join(available_baselines())}.",
    )
    parser.add_argument("--seeds", type=int, default=MINIMUM_SEEDS)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write the table to docs/evaluation/results.md.",
    )
    args = parser.parse_args()

    baselines = tuple(args.baseline or available_baselines())
    seeds = tuple(range(args.seeds))

    predictor: PolicyPredictor | None = None
    policy_label = "policy"
    topology, flow_count = "mesh-6", 8
    run_patterns: tuple[str, ...] | None = None

    if args.run_id is not None:
        directory = run_dir(args.run_id, args.runs_root)
        # Refuse a stale checkpoint rather than reporting numbers that quietly mean nothing.
        assert_mdp_version(directory, MDP_VERSION)
        metadata = read_run_metadata(directory)
        _telemetry.info("Evaluating run", {"run_id": args.run_id, "metadata": metadata})

        # The policy must meet the baselines on the *run's own* network, not on this tool's defaults.
        # Evaluating a mesh-6/8-flow policy against a differently-shaped simulator silently compares
        # two different problems.
        topology = _require_str(metadata, "topology", args.run_id)
        flow_count = _require_int(metadata, "flow_count", args.run_id)
        run_patterns = _read_patterns(metadata)
        policy_label = f"policy:{args.run_id}"
        predictor = load_policy_predictor(
            checkpoint=directory / "policy.zip",
            algorithm=_require_str(metadata, "algorithm", args.run_id),
            action_mode=_read_action_mode(metadata, args.run_id),
        )

    # Precedence: explicit flag, then the run's own evaluation_patterns, then the default pair. The
    # run's list is what its config committed to holding out, so ignoring it would evaluate
    # generalisation against patterns the config never claimed.
    patterns = tuple(args.traffic_pattern or run_patterns or ("diurnal", "flash_crowd"))

    rows: list[ComparisonRow] = []
    for pattern in patterns:
        if predictor is not None:
            rows.append(
                evaluate_policy(predictor, policy_label, pattern, seeds, args.steps, topology, flow_count)
            )
        rows.extend(
            evaluate_baseline(baseline, pattern, seeds, args.steps, topology, flow_count)
            for baseline in baselines
        )

    report = render_comparison(rows)
    if args.write_report:
        destination = Path("docs/evaluation/results.md")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")
        _telemetry.info("Report written", {"path": str(destination)})
    _telemetry.info("Evaluation complete", {"rows": len(rows), "patterns": list(patterns)})
    # The table is the deliverable — stdout is correct here, unlike in library code.
    print(report)  # noqa: T201


if __name__ == "__main__":  # supports `python -m nexus_rl.evaluate` as well as the console script
    main()
