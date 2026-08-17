"""Write tools — A1 policies only.

The agent layer is an **rApp** on the Non-RT tier. Its entire write surface is the A1 interface: it issues
policy *guidance* to the xApp, which validates it against the live network, decides what it can enforce,
and retains sole authority to allocate. See ADR-0002 for the authority rule and ADR-0008 for the interface
it is expressed through.

**There is no function in this module that produces an allocation**, and none that writes a `Constraint`
directly either. That is not a policy enforced by a check; it is enforced by the absence of any other code
path. A prompt injection that convinces the model to "set node 3 to 100% bandwidth" produces, at worst, an
A1 policy the xApp is free to refuse.

Every issuing tool here:

- takes a bounded, mandatory ``ttl_timesteps`` — every operator override expires
- records ``issued_by`` and the operator's own words, so the change stays attributable
- returns the xApp's **decision**, verbatim, including refusals

That last point is the whole reason A1 replaced the direct constraint write. A refusal arrives as a normal
response, not an exception: raising would make the model treat a considered "no" as a broken call and retry
it, and returning a cheerful payload would let it report a guarantee that is not in force. Both are worse
than relaying what the xApp actually said.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, ValidationError

from nexus_a1.decision import A1PolicyDecision
from nexus_a1.policy import A1Policy, A1PolicyScope, A1PolicyStatement, A1PolicyType, QosObjective
from nexus_agent.client import OrchestratorClient
from nexus_errors.control_error import ToolExecutionError
from nexus_types.constants import MAX_CONSTRAINT_TTL, MAX_OBJECTIVE_WEIGHT
from nexus_types.traffic_class import TrafficClass

_ISSUER = "rapp:agent-layer"
"""Provenance stamped on every policy. Names the tier, not just the service — an operator reading the
decision log should be able to tell Non-RT guidance from anything else without cross-referencing."""

DEFAULT_TTL_TIMESTEPS = 600
"""Roughly a minute of simulated time at the default tick rate. Long enough to observe an effect, short
enough that a forgotten instruction expires on its own."""

MAX_CLASS_FLOOR_MBPS = 100_000.0
"""Schema-level sanity bound on a requested floor.

Deliberately generous: whether a floor is *achievable* is the xApp's judgement, made against the live
topology, and answering it here would mean the rApp holding a model of the network it has no business
holding (ADR-0006). This bound only keeps an absurd number off the wire.
"""

TOOL_POLICY_TYPES: dict[str, A1PolicyType] = {
    "prioritise_traffic_class": A1PolicyType.QOS_OBJECTIVE,
    "set_objective_weights": A1PolicyType.QOS_OBJECTIVE,
    "avoid_links": A1PolicyType.TRAFFIC_STEERING,
    "set_class_floor": A1PolicyType.CLASS_FLOOR,
}
"""Which A1 policy type each issuing tool emits.

Kept as data so a test can assert the tool surface only issues types the xApp advertises. A tool emitting
an unadvertised type would be refused on every call — an authority surface that looks larger than it is.
"""


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
    controller still routes it — a policy must never be able to black-hole traffic, and the xApp reports
    which flows that applies to.
    """

    link_ids: list[str] = Field(
        min_length=1,
        max_length=32,
        description="Links to avoid, e.g. ['n0->n1']. Used ahead of maintenance, or to steer around a "
        "degrading link.",
    )
    ttl_timesteps: int = Field(default=DEFAULT_TTL_TIMESTEPS, gt=0, le=MAX_CONSTRAINT_TTL)
    operator_utterance: str = Field(default="")


class SetClassFloorInput(BaseModel):
    """Ask for a minimum aggregate rate for one traffic class."""

    traffic_class: TrafficClass = Field(description="Class the floor applies to.")
    floor_mbps: float = Field(
        ge=0.0,
        le=MAX_CLASS_FLOOR_MBPS,
        description="Minimum aggregate rate for the class, in Mb/s. The xApp checks this against the "
        "network's real capacity and refuses a floor it cannot guarantee.",
    )
    ttl_timesteps: int = Field(default=DEFAULT_TTL_TIMESTEPS, gt=0, le=MAX_CONSTRAINT_TTL)
    operator_utterance: str = Field(default="")


class WithdrawPolicyInput(BaseModel):
    policy_id: str = Field(min_length=1, description="A1 policy to withdraw before its TTL elapses.")


async def prioritise_traffic_class(
    client: OrchestratorClient,
    traffic_class: TrafficClass,
    ttl_timesteps: int = DEFAULT_TTL_TIMESTEPS,
    operator_utterance: str = "",
) -> dict[str, object]:
    """Favour one traffic class in the controller's objective for a bounded period.

    Use for "prioritise low-latency traffic for the next five minutes". Returns the xApp's decision: tell
    the operator what it said it would do and when the guidance expires.
    """
    objective = (
        QosObjective(latency=3.0, throughput=1.0, drops=1.0, fairness=0.5)
        if traffic_class is TrafficClass.LATENCY_SENSITIVE
        else QosObjective(throughput=3.0, latency=0.5, drops=1.0, fairness=0.5)
    )
    return await _offer(
        client,
        policy_type=A1PolicyType.QOS_OBJECTIVE,
        statement=A1PolicyStatement(qos_objective=objective),
        scope=A1PolicyScope(traffic_classes=(traffic_class,)),
        ttl_timesteps=ttl_timesteps,
        operator_utterance=operator_utterance,
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
    return await _offer(
        client,
        policy_type=A1PolicyType.QOS_OBJECTIVE,
        statement=A1PolicyStatement(
            qos_objective=QosObjective(throughput=throughput, latency=latency, drops=drops, fairness=fairness)
        ),
        ttl_timesteps=ttl_timesteps,
        operator_utterance=operator_utterance,
    )


async def avoid_links(
    client: OrchestratorClient,
    link_ids: list[str],
    ttl_timesteps: int = DEFAULT_TTL_TIMESTEPS,
    operator_utterance: str = "",
) -> dict[str, object]:
    """Steer new traffic off specific links for a bounded period.

    The xApp may accept this only in part — a named link that does not exist, or a flow with no alternative
    path, comes back in ``rejected_elements``. Report both lists; a partial acceptance is not a success.
    """
    return await _offer(
        client,
        policy_type=A1PolicyType.TRAFFIC_STEERING,
        statement=A1PolicyStatement(avoid_link_ids=tuple(link_ids)),
        scope=A1PolicyScope(link_ids=tuple(link_ids)),
        ttl_timesteps=ttl_timesteps,
        operator_utterance=operator_utterance,
    )


async def set_class_floor(
    client: OrchestratorClient,
    traffic_class: TrafficClass,
    floor_mbps: float,
    ttl_timesteps: int = DEFAULT_TTL_TIMESTEPS,
    operator_utterance: str = "",
) -> dict[str, object]:
    """Request a minimum aggregate rate for a traffic class, for a bounded period.

    The one tool whose request the network can make impossible. If the floor exceeds what the topology can
    carry, the xApp refuses it and returns the capacity figure and an achievable alternative — relay both
    rather than implying the guarantee is in force.
    """
    return await _offer(
        client,
        policy_type=A1PolicyType.CLASS_FLOOR,
        statement=A1PolicyStatement(class_floor_mbps={traffic_class: floor_mbps}),
        scope=A1PolicyScope(traffic_classes=(traffic_class,)),
        ttl_timesteps=ttl_timesteps,
        operator_utterance=operator_utterance,
    )


async def withdraw_policy(client: OrchestratorClient, policy_id: str) -> dict[str, object]:
    """Withdraw an A1 policy before its TTL elapses."""
    result = await client.delete_a1_policy(policy_id)
    if not result.get("removed"):
        raise ToolExecutionError(
            f"No A1 policy {policy_id!r} in force to withdraw. It may have already expired — tell the "
            f"operator that rather than implying you removed something.",
            {"policy_id": policy_id},
        )
    return {"withdrawn": policy_id}


async def _offer(
    client: OrchestratorClient,
    policy_type: A1PolicyType,
    statement: A1PolicyStatement,
    ttl_timesteps: int,
    operator_utterance: str,
    scope: A1PolicyScope | None = None,
) -> dict[str, object]:
    """Build a policy, offer it to the xApp, and return the decision.

    The current timestep is read from the simulator rather than assumed: a policy stamped with the wrong
    timestep is refused outright by the xApp's timing check, and one stamped slightly wrong would expire at
    an unpredictable moment — the sort of drift that makes an operator stop trusting the system.
    """
    state = await client.get_network_state()
    timestep = state.get("timestep")
    if not isinstance(timestep, int):
        raise ToolExecutionError(
            "Could not read the current timestep, so a policy cannot be stamped correctly.",
            {"got": repr(timestep)},
        )

    policy = A1Policy(
        policy_id=f"p-{uuid.uuid4().hex[:8]}",
        policy_type=policy_type,
        scope=scope or A1PolicyScope(),
        statement=statement,
        issued_by=_ISSUER,
        operator_utterance=operator_utterance,
        issued_at_timestep=timestep,
        ttl_timesteps=ttl_timesteps,
    )

    payload = await client.put_a1_policy(policy.policy_id, policy.model_dump(mode="json"))
    decision = _parse_decision(payload, policy.policy_id)

    return {
        "policy_id": policy.policy_id,
        "policy_type": policy_type.value,
        "applied": decision.is_enforced,
        "status": decision.status.value,
        "reason": decision.reason,
        "enforced_elements": list(decision.enforced_elements),
        "rejected_elements": list(decision.rejected_elements),
        "achievable_alternative": decision.achievable_alternative,
        # Only meaningful when something is actually in force. Reporting an expiry for guidance that was
        # refused would imply it is running.
        "expires_at_timestep": policy.expires_at_timestep() if decision.is_enforced else None,
        "ttl_timesteps": ttl_timesteps if decision.is_enforced else None,
    }


def _parse_decision(payload: dict[str, object], policy_id: str) -> A1PolicyDecision:
    """Narrow the xApp's response to a decision, or fail loudly.

    A refusal is a decision and must be relayed. A response this rApp *cannot interpret* is not — it is a
    failed call, and reporting an outcome for it would be inventing one. The two are deliberately handled
    differently; collapsing them is how a schema drift between the tiers turns into a confident wrong
    answer to an operator.
    """
    try:
        return A1PolicyDecision.model_validate(payload)
    except ValidationError as error:
        raise ToolExecutionError(
            "The controller returned something that is not an A1 policy decision, so what happened to "
            "this policy is unknown. Tell the operator the outcome could not be confirmed — do not "
            "report it as applied or refused.",
            {"policy_id": policy_id, "payload": payload, "errors": error.error_count()},
        ) from error
