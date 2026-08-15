"""Traffic patterns — seeded demand generators.

Every pattern is a pure function of ``(timestep, base_demand, rng)``. No internal state survives a reset, no
wall-clock, no global RNG. Same seed ⇒ same demand sequence, always.

Patterns are the axis along which this project demonstrates generalisation. A policy evaluated only on the
pattern it trained against has demonstrated memorisation, not competence — so at least one pattern is always
held out of training.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from nexus_errors.simulation_error import UnknownTrafficPatternError
from nexus_sim.constants import (
    BURST_MULTIPLIER,
    BURST_PROBABILITY,
    DEMAND_NOISE_STD,
    DIURNAL_PERIOD_TIMESTEPS,
    FLASH_CROWD_MULTIPLIER,
    FLASH_CROWD_START,
)


class TrafficPattern(Protocol):
    """A seeded demand generator."""

    name: str

    def demand_mbps(
        self,
        timestep: int,
        base_demand: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        """Per-flow demand for this timestep, in Mb/s. Non-negative and finite."""
        ...


def _noise(rng: np.random.Generator, size: int) -> NDArray[np.float32]:
    """Multiplicative jitter, clipped so demand can never go negative."""
    return np.clip(rng.normal(1.0, DEMAND_NOISE_STD, size), 0.0, None).astype(np.float32)


class SteadyTraffic:
    """Constant demand with light jitter.

    Baseline competence. Every allocator should do well here; one that does not has a bug, not a tuning
    problem.
    """

    name = "steady"

    def demand_mbps(
        self, timestep: int, base_demand: NDArray[np.float32], rng: np.random.Generator
    ) -> NDArray[np.float32]:
        del timestep
        return base_demand * _noise(rng, base_demand.size)


class DiurnalTraffic:
    """A slow sinusoidal cycle between roughly 20% and 100% of base demand.

    Stresses slow adaptation: can the policy track a smooth trend rather than reacting to each step?
    """

    name = "diurnal"

    def demand_mbps(
        self, timestep: int, base_demand: NDArray[np.float32], rng: np.random.Generator
    ) -> NDArray[np.float32]:
        phase = 2.0 * np.pi * (timestep % DIURNAL_PERIOD_TIMESTEPS) / DIURNAL_PERIOD_TIMESTEPS
        envelope = np.float32(0.6 + 0.4 * np.sin(phase))
        return base_demand * envelope * _noise(rng, base_demand.size)


class BurstyTraffic:
    """Independent per-flow bursts.

    Stresses reaction time and queue management — the policy must absorb a spike it could not have seen
    coming.
    """

    name = "bursty"

    def demand_mbps(
        self, timestep: int, base_demand: NDArray[np.float32], rng: np.random.Generator
    ) -> NDArray[np.float32]:
        del timestep
        bursting = rng.random(base_demand.size) < BURST_PROBABILITY
        multiplier = np.where(bursting, BURST_MULTIPLIER, 1.0).astype(np.float32)
        return base_demand * multiplier * _noise(rng, base_demand.size)


class FlashCrowdTraffic:
    """A sustained spike concentrated on the first half of the flows, from a fixed timestep.

    Stresses the fairness/throughput trade-off under overload: total demand far exceeds capacity, so the
    policy must decide who suffers. This is the pattern that separates a fair allocator from a
    throughput-greedy one, and it is a good default hold-out.
    """

    name = "flash_crowd"

    def demand_mbps(
        self, timestep: int, base_demand: NDArray[np.float32], rng: np.random.Generator
    ) -> NDArray[np.float32]:
        multiplier = np.ones(base_demand.size, dtype=np.float32)
        if timestep >= FLASH_CROWD_START:
            affected = base_demand.size // 2
            multiplier[:affected] = FLASH_CROWD_MULTIPLIER
        return base_demand * multiplier * _noise(rng, base_demand.size)


_PATTERNS: dict[str, TrafficPattern] = {
    pattern.name: pattern
    for pattern in (SteadyTraffic(), DiurnalTraffic(), BurstyTraffic(), FlashCrowdTraffic())
}


def get_traffic_pattern(name: str) -> TrafficPattern:
    """Resolve a pattern by name.

    The registry is the extension point — do not branch on pattern names at call sites.
    """
    pattern = _PATTERNS.get(name)
    if pattern is None:
        raise UnknownTrafficPatternError(
            f"Unknown traffic pattern {name!r}.", {"available": sorted(_PATTERNS)}
        )
    return pattern


def available_traffic_patterns() -> tuple[str, ...]:
    return tuple(sorted(_PATTERNS))
