"""State-distribution drift — has the live network moved away from what the policy was trained on?

The v2 specification calls this "the earliest warning that the policy is operating outside what it
learned", and that framing is the important part: **drift is not a failure, it is a leading indicator.**
A drifted policy may still be performing perfectly. By the time QoS thresholds are breaching, the warning
has come too late to be useful.

So this reports a number and never acts on it. Fallback activation is
:mod:`nexus_safety.fallback`'s decision, taken on observed outcomes; drift is evidence for a human
deciding whether to retrain.

The measure is deliberately simple: per-feature standardised deviation from a training-time reference,
maxed across features. Not a KS test or an MMD — those need a retained sample of the training
distribution, which this system does not keep, and a sophisticated statistic nobody can interpret at 2am
is worse than a crude one everybody can.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass

from nexus_types.network_state import NetworkState

DEFAULT_WINDOW = 128
"""Timesteps of live state to average over. Long enough that a single burst does not read as drift."""

DEFAULT_WARN_SIGMA = 3.0
"""Standardised deviations from the training mean before the feature is reported as drifted."""


@dataclass(frozen=True, slots=True)
class FeatureReference:
    """Training-time mean and spread for one observed feature."""

    mean: float
    stdev: float

    def sigma_from(self, value: float) -> float:
        """How many standard deviations ``value`` sits from the training mean.

        A zero-spread reference means the feature never varied in training. Any movement is then
        infinitely surprising in principle, which is useless as a number — so it reports 0 when the value
        matches and a large finite constant when it does not, rather than an infinity that poisons every
        aggregate downstream.
        """
        if self.stdev <= 0.0:
            return 0.0 if math.isclose(value, self.mean) else DEFAULT_WARN_SIGMA * 2
        return abs(value - self.mean) / self.stdev


@dataclass(frozen=True, slots=True)
class DriftReference:
    """What the network looked like during training.

    Recorded from the training distribution and shipped alongside the policy. An empty reference disables
    drift reporting rather than producing a fabricated baseline — reporting "no drift" against a reference
    nobody established would be worse than reporting nothing.
    """

    mean_utilization: FeatureReference | None = None
    mean_latency_ms: FeatureReference | None = None
    mean_demand_mbps: FeatureReference | None = None

    @property
    def is_established(self) -> bool:
        return any(
            reference is not None
            for reference in (self.mean_utilization, self.mean_latency_ms, self.mean_demand_mbps)
        )


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Per-feature drift, and whether any feature crossed the warning line."""

    per_feature_sigma: dict[str, float]
    warn_sigma: float
    samples: int

    @property
    def max_sigma(self) -> float:
        return max(self.per_feature_sigma.values(), default=0.0)

    @property
    def is_drifting(self) -> bool:
        return self.max_sigma >= self.warn_sigma

    @property
    def drifted_features(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, sigma in self.per_feature_sigma.items() if sigma >= self.warn_sigma)
        )


class DriftDetector:
    """Rolling comparison of live state against a training-time reference."""

    def __init__(
        self,
        reference: DriftReference,
        window: int = DEFAULT_WINDOW,
        warn_sigma: float = DEFAULT_WARN_SIGMA,
    ) -> None:
        self._reference = reference
        self._warn_sigma = warn_sigma
        self._utilization: deque[float] = deque(maxlen=window)
        self._latency: deque[float] = deque(maxlen=window)
        self._demand: deque[float] = deque(maxlen=window)

    def observe(self, state: NetworkState) -> None:
        """Record one timestep. Cheap by design — this runs inside the fast loop."""
        if state.links:
            self._utilization.append(statistics.fmean(link.utilization for link in state.links))
            self._latency.append(statistics.fmean(link.latency_ms for link in state.links))
        if state.flows:
            self._demand.append(statistics.fmean(flow.demand_mbps for flow in state.flows))

    def report(self) -> DriftReport:
        """Current drift. Reports nothing rather than something invented when there is no reference."""
        per_feature: dict[str, float] = {}
        if not self._reference.is_established:
            return DriftReport(per_feature_sigma={}, warn_sigma=self._warn_sigma, samples=0)

        for name, reference, samples in (
            ("mean_utilization", self._reference.mean_utilization, self._utilization),
            ("mean_latency_ms", self._reference.mean_latency_ms, self._latency),
            ("mean_demand_mbps", self._reference.mean_demand_mbps, self._demand),
        ):
            if reference is None or not samples:
                continue
            per_feature[name] = reference.sigma_from(statistics.fmean(samples))

        return DriftReport(
            per_feature_sigma=per_feature,
            warn_sigma=self._warn_sigma,
            samples=len(self._utilization),
        )
