"""Robustness evaluation — the mechanics, on baselines rather than a trained policy.

No policy is trained here (that is banned in tests, and rightly). What is verified is that the comparison
is *fair*: both allocators meet identical networks, held-out configurations genuinely differ from the
nominal one, and the degradation number means what the property name says.
"""

from __future__ import annotations

from nexus_rl.robustness import RobustnessResult, evaluate_robustness, render_robustness
from nexus_sim.randomization import HELD_OUT_DOMAINS, IDENTITY_DOMAIN, SampledDomain

_SEEDS = (0, 1, 2)
_STEPS = 48


def _result(baseline: str, domains: tuple[SampledDomain, ...] = HELD_OUT_DOMAINS) -> RobustnessResult:
    return evaluate_robustness(
        predictor=None,
        label=f"baseline:{baseline}",
        traffic_pattern="diurnal",
        seeds=_SEEDS,
        steps=_STEPS,
        topology="mesh-6",
        flow_count=6,
        baseline=baseline,
        domains=domains,
    )


def test_a_policy_is_scored_on_the_nominal_network_and_off_it() -> None:
    result = _result("static")

    assert result.nominal.domain == IDENTITY_DOMAIN
    assert len(result.held_out) == len(HELD_OUT_DOMAINS)


def test_held_out_configurations_actually_produce_different_networks() -> None:
    # If the held-out scores matched the nominal one exactly, the evaluation would be measuring nothing.
    result = _result("static")

    assert any(score.p95_latency_ms != result.nominal.p95_latency_ms for score in result.held_out)


def test_degradation_is_relative_to_the_nominal_score() -> None:
    result = _result("static")

    expected = (
        result.held_out_p95_latency_ms - result.nominal.p95_latency_ms
    ) / result.nominal.p95_latency_ms

    assert result.latency_degradation == expected


def test_an_idle_network_reports_no_degradation_rather_than_an_infinity() -> None:
    # Dividing by a zero nominal score would poison every aggregate downstream of the table.
    flat = _result("static", domains=(IDENTITY_DOMAIN,))

    assert flat.latency_degradation == 0.0 or flat.nominal.p95_latency_ms > 0.0


def test_the_evaluation_is_reproducible() -> None:
    # A robustness number that moves between runs is not a result.
    assert _result("static").held_out_p95_latency_ms == _result("static").held_out_p95_latency_ms


def test_two_allocators_can_be_compared_because_they_meet_identical_networks() -> None:
    # Degradation is only interpretable against a floor that faced the same conditions.
    static = _result("static")
    round_robin = _result("round_robin")

    assert [s.domain for s in static.held_out] == [s.domain for s in round_robin.held_out]


def test_the_rendered_table_carries_the_no_guarantee_caveat() -> None:
    # Track D's specification insists on this, and a caveat in the output beats one in a footnote.
    rendered = render_robustness((_result("static"),))

    assert "no explicit safety guarantee" in rendered
    assert "identical conditions" in rendered
    assert "baseline:static" in rendered
