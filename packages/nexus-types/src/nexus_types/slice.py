"""Network slices — logical networks with their own QoS guarantees over shared physical capacity.

Slicing is the canonical real-world framing for multi-agent resource orchestration, and it is what gives
several agents something meaningful to *disagree* about. v1's single undifferentiated traffic pool had no
notion of a promise: the reward wanted low latency and high throughput everywhere, so every flow's
interests pointed the same direction and "coordination" was an empty word.

A slice adds the missing thing: **a commitment that can be kept or broken**, per tier, with tiers that
genuinely conflict under contention. Premium latency can only be protected by taking capacity from
somewhere, and the somewhere is another slice whose own target then moves.

The three tiers follow the setup used in current network-slicing research: a premium tier with strict
latency guarantees, a mid tier with balanced requirements, and a best-effort tier that degrades first.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus_errors.config_error import ConfigurationError
from nexus_types.traffic_class import TrafficClass


class SliceTier(StrEnum):
    """Service tier. Ordering matters: it is the contention priority."""

    PREMIUM = "premium"
    """Strict latency guarantee. Protected first under contention, and the last to be degraded."""

    STANDARD = "standard"
    """Balanced latency and throughput requirements."""

    BEST_EFFORT = "best_effort"
    """No guarantee beyond not being starved outright. Degrades first, by design — that is what makes the
    premium guarantee affordable."""


class SliceSla(BaseModel):
    """What a slice is promised.

    Both a ceiling and a floor, because either alone is gameable. A latency-only SLA is trivially met by
    admitting almost no traffic; a throughput-only SLA is met by flooding the network and ignoring delay.
    """

    model_config = ConfigDict(frozen=True)

    max_latency_ms: float = Field(gt=0.0, description="p95 latency ceiling for the slice's flows.")
    min_satisfaction: float = Field(
        ge=0.0,
        le=1.0,
        description="Minimum fraction of offered demand the slice must actually carry.",
    )
    max_drop_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class NetworkSlice(BaseModel):
    """One logical network: a traffic class, a tier, and the promise attached to it."""

    model_config = ConfigDict(frozen=True)

    slice_id: str = Field(min_length=1)
    tier: SliceTier
    traffic_class: TrafficClass = Field(
        description="Which flows belong to this slice. One class per slice keeps membership derivable "
        "from the state the simulator already sends, so no service has to be told separately."
    )
    sla: SliceSla

    @property
    def priority(self) -> int:
        """Contention priority, higher wins. Derived from the tier rather than configured separately —
        two fields that must agree are a field that will eventually disagree."""
        return {SliceTier.PREMIUM: 2, SliceTier.STANDARD: 1, SliceTier.BEST_EFFORT: 0}[self.tier]


class SliceSet(BaseModel):
    """The slices in force. Immutable; the simulator and the MDP read the same one."""

    model_config = ConfigDict(frozen=True)

    slices: tuple[NetworkSlice, ...]

    @model_validator(mode="after")
    def _one_slice_per_class(self) -> SliceSet:
        """Membership is derived from a flow's traffic class, so two slices cannot claim one class.

        Allowing it would make "which slice is this flow in" ambiguous, and every per-slice number
        downstream would silently depend on iteration order.
        """
        if not self.slices:
            raise ConfigurationError("A slice set must contain at least one slice.", {})

        seen: set[TrafficClass] = set()
        for network_slice in self.slices:
            if network_slice.traffic_class in seen:
                raise ConfigurationError(
                    f"Two slices claim traffic class {network_slice.traffic_class.value}; slice "
                    f"membership would be ambiguous.",
                    {"traffic_class": network_slice.traffic_class.value},
                )
            seen.add(network_slice.traffic_class)

        ids = [s.slice_id for s in self.slices]
        if len(set(ids)) != len(ids):
            raise ConfigurationError("Slice ids must be unique.", {"slice_ids": ids})
        return self

    def for_class(self, traffic_class: TrafficClass) -> NetworkSlice | None:
        return next((s for s in self.slices if s.traffic_class is traffic_class), None)

    def by_id(self, slice_id: str) -> NetworkSlice | None:
        return next((s for s in self.slices if s.slice_id == slice_id), None)

    @property
    def ordered(self) -> tuple[NetworkSlice, ...]:
        """Slices by descending priority. A stable order the observation encoding can rely on."""
        return tuple(sorted(self.slices, key=lambda s: (-s.priority, s.slice_id)))


DEFAULT_SLICES = SliceSet(
    slices=(
        NetworkSlice(
            slice_id="premium",
            tier=SliceTier.PREMIUM,
            traffic_class=TrafficClass.LATENCY_SENSITIVE,
            # Tight but reachable on an uncongested mesh-6. An unreachable SLA is not a hard problem, it
            # is a constant penalty the policy learns to ignore.
            sla=SliceSla(max_latency_ms=25.0, min_satisfaction=0.95, max_drop_rate=0.02),
        ),
        NetworkSlice(
            slice_id="standard",
            tier=SliceTier.STANDARD,
            traffic_class=TrafficClass.BULK,
            sla=SliceSla(max_latency_ms=120.0, min_satisfaction=0.80, max_drop_rate=0.10),
        ),
        NetworkSlice(
            slice_id="best-effort",
            tier=SliceTier.BEST_EFFORT,
            traffic_class=TrafficClass.BACKGROUND,
            sla=SliceSla(max_latency_ms=600.0, min_satisfaction=0.35, max_drop_rate=0.50),
        ),
    )
)
"""The standard three-tier setup.

Targets are deliberately reachable when the network is uncongested and mutually unreachable when it is
not — which is exactly the contention Use Case 6 demonstrates. If every slice could always be satisfied,
there would be nothing for a multi-agent policy to negotiate.
"""
