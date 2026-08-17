"""Fallback activation — when to stop trusting the policy, and when to trust it again.

The v2 specification's guardrail: *a deterministic heuristic the system reverts to automatically if the
RL policy's metrics breach thresholds.* Two things make that harder than a threshold check:

- **Breaches are noisy.** A single bad timestep under a traffic burst is not a failing policy. Reverting
  on it produces a system that flaps between allocators, which is worse than either allocator alone.
- **Recovery must not be instant either.** If the fallback is engaged because latency is high, and the
  fallback lowers latency, then the moment it recovers the policy is handed back the network it just
  broke. That is an oscillator, not a safety mechanism.

So activation needs consecutive breaches, and deactivation needs a cooldown of consecutive *healthy*
steps. Both are configurable and both default to values that assume noise.

The **fallback activation rate is the clearest failure signal the system has** (v2 spec §8.4), so every
transition is recorded rather than merely acted on.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FallbackReason(StrEnum):
    """Why the deterministic allocator is driving."""

    NONE = "none"
    UNSAFE_ACTION = "unsafe_action"
    """The policy produced something no correction could repair — a non-finite share."""

    CONTROLLER_UNREACHABLE = "controller_unreachable"
    """No action arrived at all. Distinct from a bad action: nothing is wrong with the policy, and the
    fix is elsewhere."""

    QOS_BREACH = "qos_breach"
    """Live metrics crossed a threshold for the configured number of consecutive timesteps."""


@dataclass(frozen=True, slots=True)
class QosThresholds:
    """Live limits the *outcome* must stay within, whatever the policy intended.

    Distinct from action bounds: those constrain what the policy may ask for, these constrain what the
    network actually does. A policy can emit perfectly in-range shares and still drive the network into
    the ground, and only this catches that.
    """

    max_p95_latency_ms: float = 250.0
    max_drop_rate: float = 0.35
    """Fraction of offered demand dropped, in [0, 1]."""

    min_fairness: float = 0.25
    """Jain's fairness index. A policy serving one flow and starving the rest scores near 1/n."""

    consecutive_breaches_to_engage: int = 10
    """Sustained breach before falling back. One bad step under a burst is not a failing policy."""

    consecutive_healthy_to_release: int = 60
    """Healthy steps before the policy is trusted again. Longer than the engage window on purpose —
    otherwise the fallback fixes the metric, immediately hands back, and the system oscillates."""


@dataclass(frozen=True, slots=True)
class QosSample:
    """One timestep of observed outcome, as the supervisor reads it off the state."""

    p95_latency_ms: float
    drop_rate: float
    fairness: float

    def breaches(self, thresholds: QosThresholds) -> tuple[str, ...]:
        """Which limits this sample crosses. Empty means healthy."""
        crossed: list[str] = []
        if self.p95_latency_ms > thresholds.max_p95_latency_ms:
            crossed.append(f"p95 latency {self.p95_latency_ms:.1f}ms > {thresholds.max_p95_latency_ms:.1f}ms")
        if self.drop_rate > thresholds.max_drop_rate:
            crossed.append(f"drop rate {self.drop_rate:.3f} > {thresholds.max_drop_rate:.3f}")
        if self.fairness < thresholds.min_fairness:
            crossed.append(f"fairness {self.fairness:.3f} < {thresholds.min_fairness:.3f}")
        return tuple(crossed)


@dataclass(frozen=True, slots=True)
class FallbackTransition:
    """A change of who is driving. Recorded because the rate of these is the failure signal."""

    engaged: bool
    reason: FallbackReason
    detail: str
    timestep: int


class FallbackController:
    """Decides whether the deterministic allocator is driving, and records every transition.

    Deliberately holds no allocator and produces no action: it answers *whether*, the supervisor service
    answers *what*. Keeping the decision separable is what makes it testable without a network.
    """

    def __init__(self, thresholds: QosThresholds | None = None) -> None:
        self._thresholds = thresholds or QosThresholds()
        self._engaged = False
        self._reason = FallbackReason.NONE
        self._consecutive_breaches = 0
        self._consecutive_healthy = 0
        self._transitions: list[FallbackTransition] = []
        self._activations = 0

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def reason(self) -> FallbackReason:
        return self._reason

    @property
    def activation_count(self) -> int:
        """How many times the fallback has taken over. A rising number is the clearest failure signal."""
        return self._activations

    @property
    def transitions(self) -> tuple[FallbackTransition, ...]:
        return tuple(self._transitions)

    def engage_now(self, reason: FallbackReason, detail: str, timestep: int) -> None:
        """Fall back immediately, without waiting for a breach window.

        For failures that are not a matter of degree: an unrepairable action, or no action at all. Waiting
        ten timesteps to react to a NaN would mean applying it ten times.
        """
        self._consecutive_healthy = 0
        if self._engaged and self._reason is reason:
            return
        self._engaged = True
        self._reason = reason
        self._activations += 1
        self._record(True, reason, detail, timestep)

    def observe(self, sample: QosSample, timestep: int) -> None:
        """Record one timestep of observed outcome and update the decision."""
        crossed = sample.breaches(self._thresholds)

        if crossed:
            self._consecutive_breaches += 1
            self._consecutive_healthy = 0
            if (
                not self._engaged
                and self._consecutive_breaches >= self._thresholds.consecutive_breaches_to_engage
            ):
                self._engaged = True
                self._reason = FallbackReason.QOS_BREACH
                self._activations += 1
                self._record(
                    True,
                    FallbackReason.QOS_BREACH,
                    f"{self._consecutive_breaches} consecutive breaches: {'; '.join(crossed)}",
                    timestep,
                )
            return

        self._consecutive_breaches = 0
        self._consecutive_healthy += 1
        if self._engaged and self._consecutive_healthy >= self._thresholds.consecutive_healthy_to_release:
            released_from = self._reason
            self._engaged = False
            self._reason = FallbackReason.NONE
            self._record(
                False,
                released_from,
                f"{self._consecutive_healthy} consecutive healthy timesteps; returning control to the "
                f"policy.",
                timestep,
            )

    def status(self) -> dict[str, object]:
        """Everything an operator needs to answer "is the policy driving, and how often is it not?"."""
        return {
            "fallback_engaged": self._engaged,
            "reason": self._reason.value,
            "activation_count": self._activations,
            "consecutive_breaches": self._consecutive_breaches,
            "consecutive_healthy": self._consecutive_healthy,
            "thresholds": {
                "max_p95_latency_ms": self._thresholds.max_p95_latency_ms,
                "max_drop_rate": self._thresholds.max_drop_rate,
                "min_fairness": self._thresholds.min_fairness,
                "consecutive_breaches_to_engage": self._thresholds.consecutive_breaches_to_engage,
                "consecutive_healthy_to_release": self._thresholds.consecutive_healthy_to_release,
            },
        }

    def _record(self, engaged: bool, reason: FallbackReason, detail: str, timestep: int) -> None:
        self._transitions.append(
            FallbackTransition(engaged=engaged, reason=reason, detail=detail, timestep=timestep)
        )
        # Bounded: this is an observability aid, not a datastore. Same rule as the controller's ring buffer.
        if len(self._transitions) > 256:
            del self._transitions[0]
