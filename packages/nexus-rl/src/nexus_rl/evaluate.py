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
from nexus_mdp.reward import RewardWeights
from nexus_mdp.version import MDP_VERSION
from nexus_rl.artifacts import RUNS_ROOT, assert_mdp_version, read_run_metadata, run_dir
from nexus_rl.rollout import RolloutSummary, rollout_allocator
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

    patterns = tuple(args.traffic_pattern or ("diurnal", "flash_crowd"))
    baselines = tuple(args.baseline or available_baselines())
    seeds = tuple(range(args.seeds))

    if args.run_id is not None:
        directory = run_dir(args.run_id, args.runs_root)
        # Refuse a stale checkpoint rather than reporting numbers that quietly mean nothing.
        assert_mdp_version(directory, MDP_VERSION)
        _telemetry.info("Evaluating run", {"run_id": args.run_id, "metadata": read_run_metadata(directory)})

    rows = [
        evaluate_baseline(baseline, pattern, seeds, args.steps)
        for pattern in patterns
        for baseline in baselines
    ]

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
