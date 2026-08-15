"""Action encoding and decoding.

**Discrete (v1, DQN).** The policy picks one of a small set of named allocation *modes* rather than emitting
a number per flow. Two reasons:

1. A per-flow discrete space grows combinatorially with flow count and is not learnable within any sane
   sample budget.
2. Modes are legible. When the agent layer explains "the controller throttled bulk traffic", that maps onto
   something the controller actually chose, rather than a vector of numbers nobody can interpret.

The cost is **reachability**, and it is a real one: a mode set cannot express "60% to this flow, 40% to
that". Some optimal allocations are outside the space, so a discrete policy will plateau below what a
continuous one could reach. That plateau is expected, not a training bug — this is exactly the trade-off
ADR-0004 makes deliberately, and exactly what the Stage G PPO upgrade exists to lift.

**Continuous (Stage G, PPO).** ``Box(flow_count,)`` of per-flow shares in [0, 1]. Fully expressive.

Decoding is **total** under both: every action in the space decodes to a valid, capacity-respecting
allocation. Nothing relies on the policy learning to avoid invalid actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from nexus_errors.control_error import ActionDecodingError
from nexus_types.action import AllocationAction
from nexus_types.network_state import NetworkState
from nexus_types.traffic_class import TrafficClass


class AllocationMode(IntEnum):
    """The discrete action set. Index order is part of the MDP contract — append, never reorder."""

    OPEN = 0
    """Every flow gets its full demand. The do-nothing action."""

    THROTTLE_MODERATE = 1
    """Everything to 60%. Congestion avoidance without picking winners."""

    THROTTLE_AGGRESSIVE = 2
    """Everything to 30%. Sheds load hard to drain backlog."""

    PRIORITISE_LATENCY_SENSITIVE = 3
    """Latency-sensitive flows at full rate, everything else at 40%."""

    PRIORITISE_BULK = 4
    """Bulk flows at full rate, everything else at 40%."""

    SHED_BACKGROUND = 5
    """Background traffic to 10%, everything else full. The cheapest way to free capacity."""

    REROUTE_LEAST_LOADED = 6
    """Keep full rates, move every flow onto its least-utilized candidate path."""

    PRIORITISE_LATENCY_AND_REROUTE = 7
    """Combines mode 3 and mode 6 — the response to a congestion event on a latency-sensitive path."""


DISCRETE_ACTION_COUNT: int = len(AllocationMode)

_CLASS_SHARES: dict[AllocationMode, dict[TrafficClass, float]] = {
    AllocationMode.OPEN: {
        TrafficClass.LATENCY_SENSITIVE: 1.0,
        TrafficClass.BULK: 1.0,
        TrafficClass.BACKGROUND: 1.0,
    },
    AllocationMode.THROTTLE_MODERATE: {
        TrafficClass.LATENCY_SENSITIVE: 0.6,
        TrafficClass.BULK: 0.6,
        TrafficClass.BACKGROUND: 0.6,
    },
    AllocationMode.THROTTLE_AGGRESSIVE: {
        TrafficClass.LATENCY_SENSITIVE: 0.3,
        TrafficClass.BULK: 0.3,
        TrafficClass.BACKGROUND: 0.3,
    },
    AllocationMode.PRIORITISE_LATENCY_SENSITIVE: {
        TrafficClass.LATENCY_SENSITIVE: 1.0,
        TrafficClass.BULK: 0.4,
        TrafficClass.BACKGROUND: 0.4,
    },
    AllocationMode.PRIORITISE_BULK: {
        TrafficClass.LATENCY_SENSITIVE: 0.4,
        TrafficClass.BULK: 1.0,
        TrafficClass.BACKGROUND: 0.4,
    },
    AllocationMode.SHED_BACKGROUND: {
        TrafficClass.LATENCY_SENSITIVE: 1.0,
        TrafficClass.BULK: 1.0,
        TrafficClass.BACKGROUND: 0.1,
    },
    AllocationMode.REROUTE_LEAST_LOADED: {
        TrafficClass.LATENCY_SENSITIVE: 1.0,
        TrafficClass.BULK: 1.0,
        TrafficClass.BACKGROUND: 1.0,
    },
    AllocationMode.PRIORITISE_LATENCY_AND_REROUTE: {
        TrafficClass.LATENCY_SENSITIVE: 1.0,
        TrafficClass.BULK: 0.4,
        TrafficClass.BACKGROUND: 0.4,
    },
}

_REROUTING_MODES = frozenset(
    {AllocationMode.REROUTE_LEAST_LOADED, AllocationMode.PRIORITISE_LATENCY_AND_REROUTE}
)


@dataclass(frozen=True, slots=True)
class FlowRoutingOptions:
    """A flow's candidate paths, as the decoder needs them.

    Passed in rather than read from the simulator so decoding stays a pure function — which is what makes it
    testable without a running simulation.
    """

    flow_id: str
    candidate_paths: tuple[tuple[str, ...], ...]


def decode_discrete_action(
    action_index: int,
    state: NetworkState,
    routing: tuple[FlowRoutingOptions, ...],
    forbidden_link_ids: frozenset[str] = frozenset(),
) -> AllocationAction:
    """Decode a discrete mode into a concrete allocation.

    Out-of-range indices clamp to the last mode rather than raising: decoding must be total, and a policy
    emitting an index the environment does not recognise is a bug to find, not one to crash a training run
    over.

    ``forbidden_link_ids`` comes from active operator constraints. It restricts *routing* — the controller
    still chooses rates freely within what remains, which is the whole point of constrained authority.
    """
    mode = AllocationMode(min(max(action_index, 0), DISCRETE_ACTION_COUNT - 1))
    shares_by_class = _CLASS_SHARES[mode]

    share_by_flow_id = {flow.flow_id: shares_by_class[flow.traffic_class] for flow in state.flows}

    path_by_flow_id = reroute_paths(
        state=state,
        routing=routing,
        forbidden_link_ids=forbidden_link_ids,
        reroute_all=mode in _REROUTING_MODES,
    )
    return AllocationAction(share_by_flow_id=share_by_flow_id, path_by_flow_id=path_by_flow_id)


def reroute_paths(
    state: NetworkState,
    routing: tuple[FlowRoutingOptions, ...],
    forbidden_link_ids: frozenset[str] = frozenset(),
    reroute_all: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Choose paths for the flows that need one.

    A rerouting *mode* moves every flow onto its least-loaded path — that is the action the policy chose.
    A *constraint* is narrower: it moves only the flows whose current path actually touches a forbidden
    link. Rerouting everything on a constraint would let "avoid n0->n1" silently reshuffle traffic that has
    nothing to do with that link, which is a far bigger change than the operator asked for.

    Returns an empty mapping when nothing needs to move, leaving every flow on its current path.
    """
    if not reroute_all and not forbidden_link_ids:
        return {}

    utilization_by_link = {link.link_id: link.utilization for link in state.links}
    current_path_by_flow = {flow.flow_id: frozenset(flow.path) for flow in state.flows}

    chosen: dict[str, tuple[str, ...]] = {}
    for options in routing:
        if not reroute_all:
            on_forbidden = current_path_by_flow.get(options.flow_id, frozenset()) & forbidden_link_ids
            if not on_forbidden:
                continue
        chosen[options.flow_id] = _choose_path(
            options.candidate_paths, utilization_by_link, forbidden_link_ids
        )
    return chosen


def _choose_path(
    candidate_paths: tuple[tuple[str, ...], ...],
    utilization_by_link: dict[str, float],
    forbidden_link_ids: frozenset[str],
) -> tuple[str, ...]:
    """Pick the least-loaded candidate path, preferring ones that avoid forbidden links.

    If every candidate touches a forbidden link, the least-loaded one is used anyway. A constraint must
    never leave a flow with no path at all — that would let an operator instruction black-hole traffic,
    which is exactly the outcome constrained authority exists to prevent.
    """
    allowed = [path for path in candidate_paths if not any(link_id in forbidden_link_ids for link_id in path)]
    considered = allowed or list(candidate_paths)
    return min(
        considered,
        key=lambda path: max((utilization_by_link.get(link_id, 0.0) for link_id in path), default=0.0),
    )


def decode_continuous_action(
    shares: NDArray[np.float32],
    state: NetworkState,
) -> AllocationAction:
    """Decode a continuous per-flow share vector.

    Clipped into [0, 1] rather than validated: a policy exploring outside the box is normal early in
    training, and rejecting it would stall learning for no benefit. Length mismatches are handled by
    position, with missing entries defaulting to full demand.
    """
    if not np.all(np.isfinite(shares)):
        # Clipping leaves NaN as NaN, which then trips the allocation validator with a message about
        # shape mismatches — pointing at the wrong cause. A non-finite action means the policy diverged.
        raise ActionDecodingError(
            "Policy emitted a non-finite allocation share. The network has diverged — check the learning "
            "rate and gradient clipping; this is not an MDP_VERSION or shape problem.",
            {"non_finite_count": int(np.count_nonzero(~np.isfinite(shares)))},
        )
    clipped = np.clip(shares, 0.0, 1.0)
    return AllocationAction(
        share_by_flow_id={
            flow.flow_id: float(clipped[index]) if index < clipped.size else 1.0
            for index, flow in enumerate(state.flows)
        }
    )
