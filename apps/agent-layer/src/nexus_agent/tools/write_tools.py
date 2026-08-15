"""Write tools — constraints only.

**There is no function in this module that produces an allocation.** Every write tool builds a
:class:`~nexus_types.constraint.Constraint` and posts it to the controller, which optimizes within it. That
is not a policy enforced by a check; it is enforced by the absence of any other code path (ADR-0002).

A prompt injection that convinces the model to "set node 3 to 100% bandwidth" produces, at worst, a
constraint the controller optimizes inside — never a direct allocation.

Every tool here:

- takes a bounded, mandatory ``ttl_timesteps``
- records ``issued_by`` and the operator's own words
- raises on failure rather than returning empty success
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from nexus_agent.client import OrchestratorClient
from nexus_errors.control_error import ToolExecutionError
from nexus_types.constants import MAX_CONSTRAINT_TTL, MAX_OBJECTIVE_WEIGHT
from nexus_types.constraint import Constraint, ObjectiveWeights
from nexus_types.traffic_class import TrafficClass

_ISSUER = "agent-layer"

DEFAULT_TTL_TIMESTEPS = 600
"""Roughly a minute of simulated time at the default tick rate. Long enough to observe an effect, short
enough that a forgotten instruction expires on its own."""


class PrioritiseTrafficClassInput(BaseModel):
    """Raise one traffic class's weight in the controller's objective, temporarily."""

    traffic_class: TrafficClass = Field(
        description="Class to favour. LATENCY_SENSITIVE for interactive traffic, BULK for transfers."
    )
    ttl_timesteps: int = Field(
        default=DEFAULT_TTL_TIMESTEPS,
        gt=0,
        le=MAX_CONSTRAINT_TTL,
        description="How long this stays in force. Mandatory — every operator override expires.",
    )
    operator_utterance: str = Field(
        default="",
        description="The operator's own words. Recorded so the change stays attributable afterwards.",
    )


class SetObjectiveWeightsInput(BaseModel):
    """Reweight what the controller optimizes for, temporarily.

    This does not retrain the policy — it changes the objective its actions are scored and masked against.
    """

    throughput: float = Field(default=1.0, ge=0.0, le=MAX_OBJECTIVE_WEIGHT)
    latency: float = Field(default=1.0, ge=0.0, le=MAX_OBJECTIVE_WEIGHT)
    drops: float = Field(default=1.0, ge=0.0, le=MAX_OBJECTIVE_WEIGHT)
    fairness: float = Field(default=1.0, ge=0.0, le=MAX_OBJECTIVE_WEIGHT)
    ttl_timesteps: int = Field(default=DEFAULT_TTL_TIMESTEPS, gt=0, le=MAX_CONSTRAINT_TTL)
    operator_utterance: str = Field(default="")


class AvoidLinksInput(BaseModel):
    """Tell the controller to route new traffic away from specific links.

    A restriction on routing, not an allocation. If every candidate path for a flow is forbidden, the
    controller still routes it — a constraint must never be able to black-hole traffic.
    """

    link_ids: list[str] = Field(
        min_length=1,
        max_length=32,
        description="Links to avoid, e.g. ['n0->n1']. Used ahead of maintenance, or to steer around a "
        "degrading link.",
    )
    ttl_timesteps: int = Field(default=DEFAULT_TTL_TIMESTEPS, gt=0, le=MAX_CONSTRAINT_TTL)
    operator_utterance: str = Field(default="")


class RevokeConstraintInput(BaseModel):
    constraint_id: str = Field(min_length=1, description="Constraint to lift before its TTL elapses.")


async def prioritise_traffic_class(
    client: OrchestratorClient,
    traffic_class: TrafficClass,
    ttl_timesteps: int = DEFAULT_TTL_TIMESTEPS,
    operator_utterance: str = "",
) -> dict[str, object]:
    """Favour one traffic class in the controller's objective for a bounded period.

    Use for "prioritise low-latency traffic for the next five minutes". Returns the constraint that was
    created, including when it expires — tell the operator both what changed and when it reverts.
    """
    weights = (
        ObjectiveWeights(latency=3.0, throughput=1.0, drops=1.0, fairness=0.5)
        if traffic_class is TrafficClass.LATENCY_SENSITIVE
        else ObjectiveWeights(throughput=3.0, latency=0.5, drops=1.0, fairness=0.5)
    )
    return await _apply(
        client,
        objective_weights=weights,
        ttl_timesteps=ttl_timesteps,
        operator_utterance=operator_utterance,
        summary=f"prioritise {traffic_class.value}",
    )


async def set_objective_weights(
    client: OrchestratorClient,
    throughput: float = 1.0,
    latency: float = 1.0,
    drops: float = 1.0,
    fairness: float = 1.0,
    ttl_timesteps: int = DEFAULT_TTL_TIMESTEPS,
    operator_utterance: str = "",
) -> dict[str, object]:
    """Reweight the controller's objective directly, for finer control than the class shortcut gives."""
    return await _apply(
        client,
        objective_weights=ObjectiveWeights(
            throughput=throughput, latency=latency, drops=drops, fairness=fairness
        ),
        ttl_timesteps=ttl_timesteps,
        operator_utterance=operator_utterance,
        summary="reweight objectives",
    )


async def avoid_links(
    client: OrchestratorClient,
    link_ids: list[str],
    ttl_timesteps: int = DEFAULT_TTL_TIMESTEPS,
    operator_utterance: str = "",
) -> dict[str, object]:
    """Steer new traffic off specific links for a bounded period."""
    return await _apply(
        client,
        forbidden_link_ids=tuple(link_ids),
        ttl_timesteps=ttl_timesteps,
        operator_utterance=operator_utterance,
        summary=f"avoid {', '.join(link_ids)}",
    )


async def revoke_constraint(client: OrchestratorClient, constraint_id: str) -> dict[str, object]:
    """Lift a constraint before its TTL elapses."""
    result = await client.revoke_constraint(constraint_id)
    if not result.get("removed"):
        raise ToolExecutionError(
            f"No active constraint {constraint_id!r} to revoke. It may have already expired — tell the "
            f"operator that rather than implying you removed something.",
            {"constraint_id": constraint_id},
        )
    return {"revoked": constraint_id}


async def _apply(
    client: OrchestratorClient,
    ttl_timesteps: int,
    operator_utterance: str,
    summary: str,
    objective_weights: ObjectiveWeights | None = None,
    forbidden_link_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build and post a constraint.

    The current timestep is read from the simulator rather than assumed: a constraint stamped with the wrong
    timestep expires at an unpredictable moment, which is exactly the sort of drift that makes an operator
    stop trusting the system.
    """
    state = await client.get_network_state()
    timestep = state.get("timestep")
    if not isinstance(timestep, int):
        raise ToolExecutionError(
            "Could not read the current timestep, so a constraint cannot be stamped correctly.",
            {"got": repr(timestep)},
        )

    constraint = Constraint(
        constraint_id=f"c-{uuid.uuid4().hex[:8]}",
        issued_by=_ISSUER,
        operator_utterance=operator_utterance,
        issued_at_timestep=timestep,
        ttl_timesteps=ttl_timesteps,
        objective_weights=objective_weights,
        forbidden_link_ids=forbidden_link_ids,
    )

    applied = await client.post_constraint(constraint.model_dump(mode="json"), current_timestep=timestep)
    return {
        "applied": applied,
        "summary": summary,
        "expires_at_timestep": constraint.expires_at_timestep(),
        "ttl_timesteps": ttl_timesteps,
    }
