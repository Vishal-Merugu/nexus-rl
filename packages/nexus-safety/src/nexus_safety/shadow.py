"""Shadow-mode comparison — what a candidate policy *would* have done.

Use Case 10. A new policy runs alongside the incumbent against live traffic, its decisions recorded and
never applied. The operator promotes it only if it wins.

The thing that makes this more than logging is deciding what "wins" means, and the honest answer is that
**shadow mode cannot tell you.** The candidate's actions were never applied, so the network never
responded to them; there is no counterfactual latency to compare. Anything claiming otherwise is
inventing a number.

What shadow mode genuinely measures is narrower and still valuable:

- **Would it have been safe?** The candidate's actions go through the same bounds as the incumbent's.
  A candidate that would have emitted NaNs or starved a flow is disqualified on evidence, before it ever
  touches traffic. This is the part that actually protects anyone.
- **How different is it?** A candidate that agrees with the incumbent everywhere is not worth the
  promotion risk. One that diverges wildly is worth a much closer look.
- **Did it keep up?** A candidate that times out under live load fails deployment regardless of quality.

So the verdict this produces is about *readiness*, never about superiority. Superiority is what the
offline evaluation suite is for, where both policies really are applied to the same network.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import StrEnum

from nexus_safety.proposal import ProposedAction
from nexus_safety.violation import Violation


class ShadowVerdict(StrEnum):
    """Whether a candidate is ready to be considered for promotion."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Too few timesteps observed to say anything. The default, and the honest one."""

    UNSAFE = "unsafe"
    """The candidate would have violated a hard bound. Disqualifying on its own."""

    UNRESPONSIVE = "unresponsive"
    """The candidate failed to answer often enough to be deployable, whatever its quality."""

    INDISTINGUISHABLE = "indistinguishable"
    """Safe, responsive, and behaviourally almost identical to the incumbent. Promotion carries risk with
    no expected gain — which is a real finding, not a null result."""

    READY = "ready"
    """Safe, responsive, and meaningfully different. **Not** "better" — that is the evaluation suite's
    verdict to give, on data where the candidate's actions were actually applied."""


@dataclass(frozen=True, slots=True)
class ShadowThresholds:
    minimum_samples: int = 200
    """Timesteps before any verdict but INSUFFICIENT_DATA is offered."""

    max_failure_rate: float = 0.05
    """Fraction of timesteps the candidate may fail to answer."""

    indistinguishable_below: float = 0.02
    """Mean per-flow share divergence below which the candidate is behaviourally the incumbent."""


@dataclass(frozen=True, slots=True)
class ShadowReport:
    """What the candidate did, and whether it is ready to be considered."""

    samples: int
    failures: int
    mean_divergence: float
    """Mean absolute per-flow share difference from the incumbent, in [0, 1]."""

    max_divergence: float
    violations: tuple[Violation, ...] = field(default_factory=tuple)
    verdict: ShadowVerdict = ShadowVerdict.INSUFFICIENT_DATA
    reason: str = ""

    @property
    def failure_rate(self) -> float:
        total = self.samples + self.failures
        return self.failures / total if total else 0.0


class ShadowComparator:
    """Records candidate decisions against the incumbent's, and issues a readiness verdict.

    Holds no policy and applies nothing. The supervisor gives it both actions; it never asks for either.
    """

    def __init__(self, thresholds: ShadowThresholds | None = None) -> None:
        self._thresholds = thresholds or ShadowThresholds()
        self._divergences: list[float] = []
        self._violations: list[Violation] = []
        self._failures = 0

    def record_failure(self) -> None:
        """The candidate did not answer this timestep."""
        self._failures += 1

    def record(
        self,
        incumbent: ProposedAction,
        candidate: ProposedAction,
        candidate_violations: tuple[Violation, ...],
    ) -> None:
        """Record one timestep of side-by-side behaviour.

        Divergence is measured over the union of both flow sets, so a candidate that simply *omits* flows
        reads as divergent rather than as agreement — omission is the cheapest way to look identical.
        """
        self._violations.extend(candidate_violations)

        flow_ids = set(incumbent.share_by_flow_id) | set(candidate.share_by_flow_id)
        if not flow_ids:
            return
        self._divergences.append(
            statistics.fmean(
                abs(
                    incumbent.share_by_flow_id.get(flow_id, 0.0)
                    - candidate.share_by_flow_id.get(flow_id, 0.0)
                )
                for flow_id in flow_ids
            )
        )

    def report(self) -> ShadowReport:
        """The current verdict, erring toward refusing to judge."""
        samples = len(self._divergences)
        mean = statistics.fmean(self._divergences) if self._divergences else 0.0
        peak = max(self._divergences, default=0.0)
        base = ShadowReport(
            samples=samples,
            failures=self._failures,
            mean_divergence=mean,
            max_divergence=peak,
            violations=tuple(self._violations),
        )

        # Safety first, and before the sample-count gate: a candidate that emitted a NaN is disqualified
        # on the evidence available, not once enough timesteps have accumulated to be polite about it.
        if self._violations:
            kinds = sorted({v.kind.value for v in self._violations})
            return _verdict(
                base,
                ShadowVerdict.UNSAFE,
                f"The candidate would have violated hard bounds ({', '.join(kinds)}) across "
                f"{len(self._violations)} occurrence(s). It must not be promoted.",
            )

        if samples + self._failures < self._thresholds.minimum_samples:
            return _verdict(
                base,
                ShadowVerdict.INSUFFICIENT_DATA,
                f"{samples + self._failures} of {self._thresholds.minimum_samples} timesteps observed. "
                f"No verdict yet.",
            )

        if base.failure_rate > self._thresholds.max_failure_rate:
            return _verdict(
                base,
                ShadowVerdict.UNRESPONSIVE,
                f"The candidate failed to answer on {base.failure_rate:.1%} of timesteps, above the "
                f"{self._thresholds.max_failure_rate:.1%} limit. Not deployable regardless of quality.",
            )

        if mean < self._thresholds.indistinguishable_below:
            return _verdict(
                base,
                ShadowVerdict.INDISTINGUISHABLE,
                f"Mean divergence {mean:.4f} is below {self._thresholds.indistinguishable_below}. The "
                f"candidate behaves like the incumbent, so promotion carries risk with no expected gain.",
            )

        return _verdict(
            base,
            ShadowVerdict.READY,
            f"Safe and responsive over {samples} timesteps, with mean divergence {mean:.4f}. Ready to be "
            f"*considered* — this says the candidate is deployable, not that it is better. Its actions "
            f"were never applied, so no outcome here is attributable to it. Compare on the evaluation "
            f"suite before promoting.",
        )


def _verdict(base: ShadowReport, verdict: ShadowVerdict, reason: str) -> ShadowReport:
    return ShadowReport(
        samples=base.samples,
        failures=base.failures,
        mean_divergence=base.mean_divergence,
        max_divergence=base.max_divergence,
        violations=base.violations,
        verdict=verdict,
        reason=reason,
    )
