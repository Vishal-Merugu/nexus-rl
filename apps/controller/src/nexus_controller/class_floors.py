"""Per-class rate floors — enforcing the third kind of constraint.

``Constraint.class_floor_mbps`` was accepted, validated as a real constraint, and advertised in ADR-0002,
SECURITY.md and CONTEXT.md — while no code read it. A floor-only constraint would be accepted, confirmed
to the operator, expire on schedule, and do nothing. That is the empty-success failure at the constraint
level, which is precisely what ADR-0002 exists to prevent.

This applies floors the same way every other constraint works: it shapes what the controller may do, and
the controller still owns the allocation. Raising a class toward its floor never exceeds a flow's own
demand, so a floor cannot manufacture traffic — it can only stop a class being starved.
"""

from __future__ import annotations

from nexus_types.action import AllocationAction
from nexus_types.network_state import NetworkState
from nexus_types.traffic_class import TrafficClass

_EPSILON = 1e-9
"""Local numerical guard. Deliberately not imported from the simulator: the controller decodes from the
state it receives and does not depend on simulation internals."""


def apply_class_floors(
    action: AllocationAction,
    state: NetworkState,
    floors: dict[TrafficClass, float],
) -> AllocationAction:
    """Raise shares so each class reaches its guaranteed aggregate rate, where its demand allows.

    Shares are only ever raised, never lowered: a floor is a guarantee for one class, not a cap on the
    others. If a class's total demand is already below its floor, the floor is satisfied by definition —
    the operator asked for a minimum, not for traffic to be invented.
    """
    if not floors:
        return action

    demand_by_flow = {flow.flow_id: flow.demand_mbps for flow in state.flows}
    class_by_flow = {flow.flow_id: flow.traffic_class for flow in state.flows}
    shares = dict(action.share_by_flow_id)

    for traffic_class, floor_mbps in floors.items():
        members = [f for f, c in class_by_flow.items() if c is traffic_class]
        if not members:
            continue

        class_demand = sum(demand_by_flow[f] for f in members)
        if class_demand <= _EPSILON:
            continue  # nothing to serve; a floor cannot create demand

        granted = sum(demand_by_flow[f] * shares.get(f, 1.0) for f in members)
        target = min(floor_mbps, class_demand)  # cannot guarantee more than the class actually wants
        if granted >= target - _EPSILON:
            continue

        # Scale the class up uniformly toward the floor, clamped at each flow's full demand.
        scale = target / max(granted, _EPSILON) if granted > _EPSILON else 1.0
        for flow_id in members:
            current = shares.get(flow_id, 1.0)
            shares[flow_id] = min(1.0, current * scale if granted > _EPSILON else 1.0)

    return action.model_copy(update={"share_by_flow_id": shares})


def merge_floors(floors: list[dict[TrafficClass, float]]) -> dict[TrafficClass, float]:
    """Combine floors from several active constraints, taking the strongest guarantee per class.

    Floors accumulate rather than replace, matching ``forbidden_link_ids``: a restriction or guarantee from
    one operator instruction must not be silently weakened by a later, unrelated one.
    """
    merged: dict[TrafficClass, float] = {}
    for floor in floors:
        for traffic_class, value in floor.items():
            merged[traffic_class] = max(merged.get(traffic_class, 0.0), value)
    return merged
