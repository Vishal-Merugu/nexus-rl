"""The reproducible evaluation suite — the fixed problem every candidate policy must face.

Two policies are comparable only if they met the same scenarios, the same seeds and the same step count.
Before this existed, "the same" meant whatever flags happened to be on the command line, which makes a
comparison table a collection of unrelated numbers that look like a ranking.

So the scenario set is a **versioned artifact** (``experiments/configs/evaluation-suite.yaml``), the
registry records ``suite_version`` alongside every score, and changing the suite is as deliberate an act as
bumping ``MDP_VERSION``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import yaml

from nexus_baselines.allocator import available_baselines
from nexus_errors.config_error import ConfigurationError
from nexus_mdp.reward import RewardWeights
from nexus_registry.record import EvaluationScore, utc_now
from nexus_rl.rollout import PolicyPredictor, RolloutSummary, rollout_policy
from nexus_sim.simulator import NetworkSimulator
from nexus_sim.traffic import available_traffic_patterns

MINIMUM_SUITE_SEEDS = 3
"""Below this, seed variance in RL routinely exceeds the effect being measured.

Mirrors ``nexus_rl.evaluate.MINIMUM_SEEDS``. A suite that declares fewer is rejected at load rather than
producing a score the registry would then refuse to promote — failing at the config is failing where the
mistake was made.
"""

MINIMUM_SUITE_PATTERNS = 2
"""A single-pattern suite measures memorisation, not competence."""


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    """A named, versioned set of scenarios."""

    name: str
    suite_version: int
    topology: str
    flow_count: int
    steps: int
    seeds: int
    patterns: tuple[str, ...]
    baselines: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.suite_version < 1:
            raise ConfigurationError("suite_version must be at least 1.", {"name": self.name})
        if self.seeds < MINIMUM_SUITE_SEEDS:
            raise ConfigurationError(
                f"An evaluation suite must run at least {MINIMUM_SUITE_SEEDS} seeds; this one declares "
                f"{self.seeds}. Scores below that are anecdotes and the registry will refuse to promote "
                f"on them.",
                {"name": self.name, "seeds": self.seeds},
            )
        if len(self.patterns) < MINIMUM_SUITE_PATTERNS:
            raise ConfigurationError(
                f"An evaluation suite needs at least {MINIMUM_SUITE_PATTERNS} traffic patterns, so a "
                f"policy is scored somewhere other than where it trained.",
                {"name": self.name, "patterns": list(self.patterns)},
            )
        if self.steps <= 0 or self.flow_count <= 0:
            raise ConfigurationError(
                "steps and flow_count must be positive.",
                {"name": self.name, "steps": self.steps, "flow_count": self.flow_count},
            )

        # Unknown names are caught here rather than partway through a scoring run, where the failure
        # arrives after minutes of compute and with the suite half-applied.
        known_patterns = set(available_traffic_patterns())
        unknown = [pattern for pattern in self.patterns if pattern not in known_patterns]
        if unknown:
            raise ConfigurationError(
                f"Unknown traffic pattern(s): {unknown}.",
                {"name": self.name, "available": sorted(known_patterns)},
            )
        known_baselines = set(available_baselines())
        unknown_baselines = [item for item in self.baselines if item not in known_baselines]
        if unknown_baselines:
            raise ConfigurationError(
                f"Unknown baseline(s): {unknown_baselines}.",
                {"name": self.name, "available": sorted(known_baselines)},
            )

    @property
    def seed_range(self) -> tuple[int, ...]:
        """The exact seeds every candidate runs on. Fixed, not sampled — that is the reproducibility."""
        return tuple(range(self.seeds))


def load_suite(path: Path) -> EvaluationSuite:
    """Load the suite definition from YAML."""
    if not path.exists():
        raise ConfigurationError(f"No evaluation suite at {path}.", {"path": str(path)})

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Evaluation suite at {path} is not a mapping.", {"path": str(path)})

    known = set(EvaluationSuite.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        # A silently ignored key is how a suite ends up not scoring what its file says it scores.
        raise ConfigurationError(
            f"Unknown suite keys: {sorted(unknown)}.", {"path": str(path), "known": sorted(known)}
        )

    for key in ("patterns", "baselines"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return EvaluationSuite(**raw)


def score_policy(predictor: PolicyPredictor, suite: EvaluationSuite, label: str) -> EvaluationScore:
    """Run a policy over the whole suite and produce the score the registry stores.

    The per-pattern breakdown is kept alongside the aggregate on purpose. An aggregate alone hides the
    policy that wins on its training distribution and collapses off it — which is the single most common
    way an RL result turns out not to be one.
    """
    per_pattern: dict[str, float] = {}
    summaries: list[RolloutSummary] = []

    for pattern in suite.patterns:
        pattern_summaries = [
            rollout_policy(
                simulator=NetworkSimulator.from_config(
                    suite.topology, pattern, seed=seed, flow_count=suite.flow_count
                ),
                predictor=predictor,
                steps=suite.steps,
                seed=seed,
                traffic_pattern=pattern,
                label=label,
                reward_weights=RewardWeights(),
            )
            for seed in suite.seed_range
        ]
        per_pattern[pattern] = statistics.fmean(s.mean_p95_latency_ms for s in pattern_summaries)
        summaries.extend(pattern_summaries)

    return EvaluationScore(
        suite_name=suite.name,
        suite_version=suite.suite_version,
        seeds=suite.seeds,
        scored_at=utc_now(),
        mean_p95_latency_ms=statistics.fmean(s.mean_p95_latency_ms for s in summaries),
        mean_throughput_mbps=statistics.fmean(s.mean_throughput_mbps for s in summaries),
        mean_fairness=statistics.fmean(s.mean_fairness for s in summaries),
        total_dropped_mb=sum(s.total_dropped_mb for s in summaries),
        per_pattern=per_pattern,
    )
