"""Incidents — deliberate degradations injected into the simulator.

An incident is part of the environment, not of the action space. The controller reacts to one; it can neither
cause nor clear one. That asymmetry is what makes Use Case 4 a genuine test of the learned policy.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class IncidentKind(StrEnum):
    LINK_DEGRADED = "LINK_DEGRADED"
    """Capacity reduced to a fraction of nominal."""

    LINK_FAILED = "LINK_FAILED"
    """Capacity to zero. Traffic must reroute or drop."""

    NODE_FAILED = "NODE_FAILED"
    """Every link incident to the node fails."""


class Incident(BaseModel):
    """A scheduled degradation."""

    model_config = ConfigDict(frozen=True)

    incident_id: str
    kind: IncidentKind
    target_id: str = Field(description="link_id for link incidents, node id for NODE_FAILED.")
    start_timestep: int = Field(ge=0)
    duration_timesteps: int = Field(gt=0)
    capacity_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Remaining fraction of nominal capacity. Used by LINK_DEGRADED; 0 for failures.",
    )

    def is_active_at(self, timestep: int) -> bool:
        return self.start_timestep <= timestep < self.start_timestep + self.duration_timesteps
