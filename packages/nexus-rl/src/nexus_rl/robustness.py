"""Robustness evaluation — does the policy survive a network it never trained on?

This is the evidence for Use Case 9, and the honest answer to "what if reality differs from your
simulator?". A policy is scored twice: on the **nominal** network at the centre of its training
distribution, and on **held-out** configurations deliberately outside it. The gap between those is the
number that means something.

Two things this refuses to do, because both would make the result decoration:

- **Score "held-out" configurations drawn from the training distribution.** That measures the same thing
  twice with more steps. ``HELD_OUT_DOMAINS`` sits beyond the default ranges on purpose, and a test
  asserts it.
- **Report a degradation without reporting the baseline's.** A policy that degrades 40% off-distribution
  looks bad until you see the static baseline degrade 60% on the same networks. Degradation is only
  interpretable against a floor that faced identical conditions.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from nexus_baselines.allocator import build_baseline
from nexus_mdp.reward import RewardWeights
from nexus_rl.rollout import PolicyPredictor, RolloutSummary, rollout_allocator, rollout_policy
from nexus_sim.randomization import HELD_OUT_DOMAINS, IDENTITY_DOMAIN, SampledDomain, apply_domain
from nexus_sim.simulator import NetworkSimulator
from nexus_sim.topology import Topology, build_topology


@dataclass(frozen=True, slots=True)
class DomainScore:
    """One allocator's mean result on one simulator configuration."""

    label: str
    domain: SampledDomain
    p95_latency_ms: float
    throughput_mbps: float
    fairness: float
    dropped_mb: float


@dataclass(frozen=True, slots=True)
class RobustnessResult:
    """Nominal versus held-out, for one allocator."""

    label: str
    nominal: DomainScore
    held_out: tuple[DomainScore, ...]

    @property
    def held_out_p95_latency_ms(self) -> float:
        return statistics.fmean(score.p95_latency_ms for score in self.held_out)

    @property
    def latency_degradation(self) -> float:
        """Fractional increase in p95 latency off-distribution. 0.2 means 20% worse.

        Returns 0.0 when the nominal score is 0 rather than dividing by it — an idle network is not
        infinitely robust, it is uninformative, and an infinity here would poison the comparison table.
        """
        if self.nominal.p95_latency_ms <= 0.0:
            return 0.0
        return (self.held_out_p95_latency_ms - self.nominal.p95_latency_ms) / self.nominal.p95_latency_ms

    @property
    def degrades_gracefully(self) -> bool:
        """Whether the policy stayed within half again its nominal latency off-distribution.

        A threshold, not a law. It exists so "degraded gracefully" is a claim with a definition attached
        rather than an impression formed from looking at a table.
        """
        return self.latency_degradation < 0.5


def evaluate_robustness(
    predictor: PolicyPredictor | None,
    label: str,
    traffic_pattern: str,
    seeds: tuple[int, ...],
    steps: int,
    topology: str = "mesh-6",
    flow_count: int = 8,
    baseline: str | None = None,
    domains: tuple[SampledDomain, ...] = HELD_OUT_DOMAINS,
) -> RobustnessResult:
    """Score one allocator on the nominal network and on each held-out configuration.

    Exactly one of ``predictor`` or ``baseline`` is used, and both go through the identical simulator
    construction and seeds — the comparison is only meaningful because they differ in nothing else.
    """
    return RobustnessResult(
        label=label,
        nominal=_score(
            predictor, baseline, label, traffic_pattern, seeds, steps, topology, flow_count, IDENTITY_DOMAIN
        ),
        held_out=tuple(
            _score(predictor, baseline, label, traffic_pattern, seeds, steps, topology, flow_count, domain)
            for domain in domains
        ),
    )


def render_robustness(results: tuple[RobustnessResult, ...]) -> str:
    """Render the comparison, with the caveat that Track D's own specification insists on.

    Domain randomization improves robustness and provides **no explicit safety guarantee**; over-wide
    ranges destabilise training rather than helping. Saying so in the output rather than in a footnote is
    the difference between a credible claim and an overclaim.
    """
    header = (
        "| Allocator | Nominal p95 (ms) | Held-out p95 (ms) | Degradation | Graceful |\n"
        "| --------- | ---------------- | ----------------- | ----------- | -------- |\n"
    )
    body = "".join(
        f"| {r.label} | {r.nominal.p95_latency_ms:.1f} | {r.held_out_p95_latency_ms:.1f} | "
        f"{r.latency_degradation:+.1%} | {'yes' if r.degrades_gracefully else 'no'} |\n"
        for r in results
    )
    caveat = (
        "\n_Held-out configurations sit outside the training randomization ranges (capacity, demand and "
        "latency scaling). Domain randomization improves robustness; it provides **no explicit safety "
        "guarantee**, and overly aggressive ranges destabilise training rather than helping. Degradation "
        "is only interpretable against a baseline that faced identical conditions._\n"
    )
    return header + body + caveat


def _score(
    predictor: PolicyPredictor | None,
    baseline: str | None,
    label: str,
    traffic_pattern: str,
    seeds: tuple[int, ...],
    steps: int,
    topology: str,
    flow_count: int,
    domain: SampledDomain,
) -> DomainScore:
    summaries: list[RolloutSummary] = []
    for seed in seeds:
        simulator = NetworkSimulator.from_config(
            topology, traffic_pattern, seed=seed, flow_count=flow_count, domain=domain
        )
        if predictor is not None:
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
        else:
            assert baseline is not None, "evaluate_robustness needs a predictor or a baseline"
            summaries.append(
                rollout_allocator(
                    simulator=simulator,
                    # The baseline is built from the *randomized* topology so it faces the same network
                    # the policy does. Building it from the nominal one would hand the policy an advantage
                    # and quietly invalidate the comparison.
                    allocator=build_baseline(baseline, _scaled_topology(topology, domain), flow_count),
                    steps=steps,
                    seed=seed,
                    traffic_pattern=traffic_pattern,
                    reward_weights=RewardWeights(),
                )
            )

    return DomainScore(
        label=label,
        domain=domain,
        p95_latency_ms=statistics.fmean(s.mean_p95_latency_ms for s in summaries),
        throughput_mbps=statistics.fmean(s.mean_throughput_mbps for s in summaries),
        fairness=statistics.fmean(s.mean_fairness for s in summaries),
        dropped_mb=statistics.fmean(s.total_dropped_mb for s in summaries),
    )


def _scaled_topology(topology_name: str, domain: SampledDomain) -> Topology:
    return apply_domain(build_topology(topology_name), domain)
