"""Simulation state ownership.

One simulator, one writer, one lock. Every reader — the controller, the agent layer's read tools, the
dashboard — goes through here, so they cannot disagree about what timestep it is. Two surfaces showing
different timesteps is the most common source of "the explanation doesn't match what I saw", and it is
designed out rather than tested for.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

from nexus_sim.metrics import StepMetrics
from nexus_sim.simulator import NetworkSimulator
from nexus_types.action import AllocationAction
from nexus_types.constants import DEFAULT_HISTORY_SIZE
from nexus_types.incident import Incident
from nexus_types.network_state import NetworkState


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One past timestep, kept for agent explanations."""

    state: NetworkState
    metrics: StepMetrics
    issued_by: str
    """Provenance of the action applied: 'controller', 'baseline:<name>', or 'none'."""


class SimulationService:
    """Owns the live simulation.

    History is a bounded ring buffer, not a datastore. It exists so the agent layer can answer "why did that
    happen" about the recent past; anything longer-lived belongs in a run artifact.
    """

    def __init__(
        self,
        simulator: NetworkSimulator,
        history_size: int = DEFAULT_HISTORY_SIZE,
    ) -> None:
        self._simulator = simulator
        self._history: deque[HistoryEntry] = deque(maxlen=history_size)
        self._lock = asyncio.Lock()
        self._incidents: list[Incident] = []

    @property
    def state(self) -> NetworkState:
        """The current state. Frozen, so callers may hold it safely."""
        return self._simulator.state

    @property
    def metrics(self) -> StepMetrics:
        return self._simulator.metrics

    @property
    def simulator(self) -> NetworkSimulator:
        return self._simulator

    async def step(self, action: AllocationAction | None) -> HistoryEntry:
        """Advance one timestep under the lock.

        Serialised deliberately: concurrent steps would interleave writes to the queue state and produce a
        trajectory that no seed can reproduce.
        """
        async with self._lock:
            result = self._simulator.step(action)
            entry = HistoryEntry(
                state=result.state,
                metrics=result.metrics,
                issued_by=action.issued_by if action is not None else "none",
            )
            self._history.append(entry)
            return entry

    async def reset(self) -> NetworkState:
        async with self._lock:
            self._history.clear()
            return self._simulator.reset()

    def history(self, limit: int = 50) -> tuple[HistoryEntry, ...]:
        """The most recent entries, oldest first."""
        entries = list(self._history)
        return tuple(entries[-limit:])

    def active_incidents(self) -> tuple[Incident, ...]:
        timestep = self._simulator.timestep
        return tuple(incident for incident in self._incidents if incident.is_active_at(timestep))

    async def inject_incident(self, incident: Incident) -> Incident:
        """Schedule a degradation.

        Incidents are environment, not action space — the controller reacts to one, and can neither cause
        nor clear it. Injection is an operator/demo affordance, exposed here and nowhere near the policy.
        """
        async with self._lock:
            self._incidents.append(incident)
            self._simulator.set_incidents(tuple(self._incidents))
            return incident
