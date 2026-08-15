"""Incident application — how a scheduled degradation reduces link capacity.

Incidents are part of the environment, not the action space. The controller reacts to one; it cannot cause
or clear one.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nexus_sim.topology import Topology
from nexus_types.incident import Incident, IncidentKind


def effective_capacities_mbps(
    nominal_capacity: NDArray[np.float32],
    topology: Topology,
    incidents: tuple[Incident, ...],
    timestep: int,
) -> tuple[NDArray[np.float32], tuple[str, ...]]:
    """Apply active incidents to nominal capacities.

    Overlapping incidents take the **most severe** outcome rather than compounding: two independent 50%
    degradations on one link describe one degraded link, not a 25% one.

    Returns:
        The per-link effective capacity, and the ids of the incidents that were active.
    """
    effective = nominal_capacity.copy()
    index_of = topology.link_index()
    active: list[str] = []

    for incident in incidents:
        if not incident.is_active_at(timestep):
            continue

        if incident.kind is IncidentKind.NODE_FAILED:
            affected = topology.links_incident_to(incident.target_id)
            factor = 0.0
        else:
            affected = (incident.target_id,)
            factor = 0.0 if incident.kind is IncidentKind.LINK_FAILED else incident.capacity_factor

        matched = False
        for link_id in affected:
            position = index_of.get(link_id)
            if position is None:
                continue
            matched = True
            effective[position] = min(effective[position], nominal_capacity[position] * factor)

        # Only count it as active if it actually degraded something. An incident targeting a link the
        # topology does not have used to appear in active_incident_ids while changing no capacity — the
        # dashboard showed "Incident active" and the agent would tell an operator the network was degraded
        # when nothing was.
        if matched:
            active.append(incident.incident_id)

    return effective.astype(np.float32), tuple(active)
