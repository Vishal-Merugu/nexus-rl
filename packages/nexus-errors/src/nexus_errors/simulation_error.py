"""Errors raised by the simulator and the environment."""

from __future__ import annotations

from nexus_errors.base_error import NexusError


class SimulationError(NexusError):
    """Base for anything wrong inside the simulation."""

    code = "simulation_error"


class InvariantViolationError(SimulationError):
    """A physical invariant of the network was violated.

    Raised when utilization leaves [0, 1], a queue goes negative, or allocations exceed capacity. These are
    always bugs in the simulation math — never valid states the caller should handle. Treat as CRITICAL:
    RL will find and exploit any invariant that is not enforced, and the resulting policy looks successful
    while doing something physically impossible.
    """

    code = "invariant_violation"


class TopologyError(SimulationError):
    """The network topology is unusable — disconnected, empty, or missing a path a flow requires."""

    code = "topology_error"


class UnknownTrafficPatternError(SimulationError):
    """A traffic pattern was requested by a name that is not in the registry."""

    code = "unknown_traffic_pattern"
