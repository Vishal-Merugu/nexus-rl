"""Shadow-mode comparison.

The verdicts are deliberately conservative, and most of these tests pin that down: the comparator refuses
to judge without data, disqualifies on safety before anything else, and never claims the candidate is
*better* — only that it is deployable. Superiority is the evaluation suite's verdict, on data where the
candidate's actions were actually applied.
"""

from __future__ import annotations

from nexus_safety.proposal import ProposedAction
from nexus_safety.shadow import ShadowComparator, ShadowThresholds, ShadowVerdict
from nexus_safety.violation import Violation, ViolationKind

_FAST = ShadowThresholds(minimum_samples=10, max_failure_rate=0.1, indistinguishable_below=0.02)


def _action(**shares: float) -> ProposedAction:
    return ProposedAction(share_by_flow_id=dict(shares))


def _violation() -> Violation:
    return Violation(kind=ViolationKind.NON_FINITE_SHARE, subject="f0", detail="nan")


def test_no_verdict_is_offered_before_enough_data() -> None:
    comparator = ShadowComparator(_FAST)
    for _ in range(3):
        comparator.record(_action(f0=0.5), _action(f0=0.9), ())

    assert comparator.report().verdict is ShadowVerdict.INSUFFICIENT_DATA


def test_a_meaningfully_different_safe_candidate_becomes_ready() -> None:
    comparator = ShadowComparator(_FAST)
    for _ in range(10):
        comparator.record(_action(f0=0.5, f1=0.5), _action(f0=0.9, f1=0.2), ())

    report = comparator.report()

    assert report.verdict is ShadowVerdict.READY
    assert report.mean_divergence > 0.02


def test_ready_never_claims_the_candidate_is_better() -> None:
    # Its actions were never applied, so no outcome is attributable to it. Claiming superiority here
    # would be inventing a counterfactual.
    comparator = ShadowComparator(_FAST)
    for _ in range(10):
        comparator.record(_action(f0=0.5), _action(f0=0.9), ())

    reason = comparator.report().reason

    assert "not that it is better" in reason
    assert "never applied" in reason


def test_a_candidate_that_matches_the_incumbent_is_reported_as_indistinguishable() -> None:
    # A real finding: promotion carries risk with no expected gain.
    comparator = ShadowComparator(_FAST)
    for _ in range(10):
        comparator.record(_action(f0=0.5, f1=0.5), _action(f0=0.5, f1=0.5), ())

    assert comparator.report().verdict is ShadowVerdict.INDISTINGUISHABLE


def test_an_unsafe_candidate_is_disqualified_immediately_not_after_the_sample_gate() -> None:
    # A candidate that emitted a NaN is disqualified on the evidence available, not once enough
    # timesteps have accumulated to be polite about it.
    comparator = ShadowComparator(_FAST)

    comparator.record(_action(f0=0.5), _action(f0=0.5), (_violation(),))

    report = comparator.report()
    assert report.verdict is ShadowVerdict.UNSAFE
    assert "must not be promoted" in report.reason


def test_safety_outranks_every_other_verdict() -> None:
    comparator = ShadowComparator(_FAST)
    for _ in range(20):
        comparator.record(_action(f0=0.5), _action(f0=0.5), ())
    comparator.record(_action(f0=0.5), _action(f0=0.5), (_violation(),))

    assert comparator.report().verdict is ShadowVerdict.UNSAFE


def test_a_candidate_that_cannot_keep_up_is_not_deployable_whatever_its_quality() -> None:
    comparator = ShadowComparator(_FAST)
    for _ in range(10):
        comparator.record(_action(f0=0.5), _action(f0=0.9), ())
    for _ in range(5):
        comparator.record_failure()

    report = comparator.report()

    assert report.verdict is ShadowVerdict.UNRESPONSIVE
    assert report.failure_rate > 0.1


def test_omitting_flows_reads_as_divergence_not_agreement() -> None:
    # Omission is the cheapest way for a candidate to look identical to the incumbent.
    comparator = ShadowComparator(_FAST)
    for _ in range(10):
        comparator.record(_action(f0=0.5, f1=0.5), _action(f0=0.5), ())

    assert comparator.report().verdict is ShadowVerdict.READY
    assert comparator.report().mean_divergence > 0.0


def test_failures_are_counted_even_when_no_comparison_was_possible() -> None:
    comparator = ShadowComparator(_FAST)
    for _ in range(10):
        comparator.record_failure()

    report = comparator.report()

    assert report.samples == 0
    assert report.failures == 10
    assert report.failure_rate == 1.0
