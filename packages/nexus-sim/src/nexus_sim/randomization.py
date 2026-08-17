"""Domain randomization — training against a distribution of networks, not one network.

A policy trained against a single fixed simulator configuration learns that configuration. The standard
mitigation for the sim-to-real gap is to randomize the simulator's parameters across training episodes so
the policy learns to be robust across a *range* of conditions instead.

Two things this file is careful about.

**Determinism is not negotiable.** Randomizing the environment must not make a run irreproducible. Every
draw comes from an explicitly passed ``numpy.random.Generator``, so `(master seed, episode index)` still
determines the network exactly — the trajectory varies across episodes and replays identically across
runs. A randomization scheme that broke that would trade a real property for a hoped-for one.

**It is not a safety guarantee, and the docs say so.** Domain randomization improves robustness; it
provides no explicit guarantee, and over-aggressive ranges destabilise training rather than helping. The
defaults here are deliberately modest, and :class:`RandomizationRanges` is the one place to widen them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nexus_errors.config_error import ConfigurationError
from nexus_sim.topology import Link, Topology


@dataclass(frozen=True, slots=True)
class RandomizationRanges:
    """Multiplicative ranges applied per episode.

    Multiplicative rather than absolute so one set of ranges is meaningful across topologies with
    different capacities. ``(1.0, 1.0)`` disables a dimension without needing a separate flag.
    """

    capacity: tuple[float, float] = (0.7, 1.3)
    """Per-link capacity scaling. Wide enough that the policy meets both a congested and a roomy network,
    narrow enough that the action space stays meaningful in both."""

    demand: tuple[float, float] = (0.8, 1.25)
    """Traffic demand scaling for the episode."""

    latency: tuple[float, float] = (0.8, 1.4)
    """Base propagation-latency scaling."""

    def __post_init__(self) -> None:
        for name, (low, high) in (
            ("capacity", self.capacity),
            ("demand", self.demand),
            ("latency", self.latency),
        ):
            if low <= 0.0 or high <= 0.0:
                raise ConfigurationError(
                    f"{name} randomization range must be positive; a zero or negative scale produces a "
                    f"network that cannot carry traffic.",
                    {"range": name, "low": low, "high": high},
                )
            if low > high:
                raise ConfigurationError(
                    f"{name} randomization range is inverted.", {"range": name, "low": low, "high": high}
                )

    @property
    def is_enabled(self) -> bool:
        """Whether any dimension actually varies. All-identity ranges mean randomization is off."""
        return any(low != 1.0 or high != 1.0 for low, high in (self.capacity, self.demand, self.latency))


@dataclass(frozen=True, slots=True)
class SampledDomain:
    """One episode's drawn parameters.

    Recorded rather than merely applied: a robustness result that cannot say *which* configurations a
    policy met is an anecdote, and a failure that cannot be reproduced with its own parameters cannot be
    debugged.
    """

    capacity_scale: float
    demand_scale: float
    latency_scale: float

    def as_dict(self) -> dict[str, float]:
        return {
            "capacity_scale": self.capacity_scale,
            "demand_scale": self.demand_scale,
            "latency_scale": self.latency_scale,
        }


IDENTITY_DOMAIN = SampledDomain(capacity_scale=1.0, demand_scale=1.0, latency_scale=1.0)
"""The nominal network. What a non-randomized run trains on, and the centre of the training distribution."""


def sample_domain(rng: np.random.Generator, ranges: RandomizationRanges) -> SampledDomain:
    """Draw one episode's parameters.

    Takes the generator explicitly — no module-level RNG, no ``np.random.*``. This is the rule that keeps
    "same seed, same trajectory" true even while the environment varies.
    """
    return SampledDomain(
        capacity_scale=float(rng.uniform(*ranges.capacity)),
        demand_scale=float(rng.uniform(*ranges.demand)),
        latency_scale=float(rng.uniform(*ranges.latency)),
    )


def apply_domain(topology: Topology, domain: SampledDomain) -> Topology:
    """Return a new topology with the sampled scaling applied.

    Never mutates: the nominal topology stays intact so the next episode draws from the same centre rather
    than compounding scale on scale — a bug that turns a bounded range into a random walk.
    """
    return Topology(
        name=topology.name,
        nodes=topology.nodes,
        links=tuple(
            Link(
                link_id=link.link_id,
                source=link.source,
                target=link.target,
                capacity_mbps=link.capacity_mbps * domain.capacity_scale,
            )
            for link in topology.links
        ),
    )


HELD_OUT_DOMAINS: tuple[SampledDomain, ...] = (
    SampledDomain(capacity_scale=0.5, demand_scale=1.5, latency_scale=1.8),
    SampledDomain(capacity_scale=1.6, demand_scale=0.6, latency_scale=0.7),
    SampledDomain(capacity_scale=0.6, demand_scale=1.4, latency_scale=0.9),
)
"""Configurations deliberately **outside** the default training ranges.

This is the evidence for Use Case 9. Evaluating on the training distribution measures how well a policy
learned it; evaluating here measures whether it generalises at all. Each entry sits beyond at least one
default range on purpose — a "held-out" set drawn from the same distribution as training would be a
harder-sounding test of the same thing.
"""
