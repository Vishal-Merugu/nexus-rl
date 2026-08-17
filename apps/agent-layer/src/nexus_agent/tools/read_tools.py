"""Read tools — unrestricted.

Exposing more state costs nothing and makes explanations better: an agent that can see the recent history,
the active constraints and the incident log answers "why is node 4 getting less bandwidth" with real numbers
instead of a plausible story.

Every tool returns **structured data, never prose**. A tool that returns a formatted sentence has pre-empted
the model's reasoning and stripped out the numbers it was supposed to cite.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexus_agent.client import OrchestratorClient
from nexus_errors.control_error import ToolExecutionError


class GetNetworkStateInput(BaseModel):
    """No parameters — the current state is the current state."""


class GetRecentHistoryInput(BaseModel):
    limit: int = Field(
        default=20, ge=1, le=200, description="How many recent timesteps to return, oldest first."
    )


class GetRecentDecisionsInput(BaseModel):
    limit: int = Field(
        default=20, ge=1, le=200, description="How many recent controller decisions to return."
    )


class GetFlowDetailInput(BaseModel):
    flow_id: str = Field(description="Flow to describe, e.g. 'f3'.", min_length=1)


async def get_network_state(client: OrchestratorClient) -> dict[str, object]:
    """Current network state: every link's utilization, backlog and latency; every flow's demand and
    allocation. Join anything else on the returned ``timestep``."""
    state = await client.get_network_state()
    metrics = await client.get_metrics()
    return {"state": state, "metrics": metrics}


async def get_recent_history(client: OrchestratorClient, limit: int = 20) -> dict[str, object]:
    """Recent timesteps with their QoS metrics. Use this to answer questions about trends and about what
    changed, rather than describing the current instant as though it were the whole story."""
    return {"history": await client.get_history(limit=limit)}


async def get_recent_decisions(client: OrchestratorClient, limit: int = 20) -> dict[str, object]:
    """What the controller actually did, timestep by timestep, and which constraints were in force.

    This is the evidence behind any explanation of controller behaviour. Note what it does *not* contain:
    the policy's reasons. Those are not observable from outside the network, and inventing them is exactly
    the overclaim this project must avoid.
    """
    return {"decisions": await client.get_decisions(limit=limit)}


async def get_active_constraints(client: OrchestratorClient) -> dict[str, object]:
    """Constraints currently in force — what the controller is actually **enforcing**, with who issued each
    one, the operator utterance behind it, and when it expires.

    The enforcement view. For what was *asked for*, and what the controller agreed to, read the A1 policies
    instead: a partially-accepted policy appears here only as the part that was accepted.
    """
    return await client.get_constraints(timestep=await _current_timestep(client))


async def get_active_policies(client: OrchestratorClient) -> dict[str, object]:
    """A1 policies currently in force, and the policy types this controller advertises it can enforce.

    The guidance view: what the Non-RT tier asked for. Use it to answer "what instructions are running?",
    to find a ``policy_id`` before withdrawing one, and to check a policy type is supported before issuing
    it rather than discovering the limit by having it refused.
    """
    timestep = await _current_timestep(client)
    policies = await client.get_a1_policies(timestep=timestep)
    advertised = await client.get_a1_policy_types()
    return {
        "timestep": timestep,
        "policies": policies,
        "advertised_policy_types": advertised.get("policy_type_ids"),
        "a1_schema_version": advertised.get("schema_version"),
    }


async def _current_timestep(client: OrchestratorClient) -> int:
    """The simulator's timestep, or a tool error.

    "Currently in force" is meaningless without it, and defaulting to 0 would report expired guidance as
    active — the whole set, since every policy is issued at a later timestep than that.
    """
    state = await client.get_network_state()
    timestep = state.get("timestep")
    if not isinstance(timestep, int):
        raise ToolExecutionError(
            "Could not read the current timestep, so 'currently in force' cannot be determined. Say so "
            "rather than reporting guidance as active.",
            {"got": repr(timestep)},
        )
    return timestep


async def get_active_incidents(client: OrchestratorClient) -> dict[str, object]:
    """Degradations currently affecting the network — failed or degraded links and nodes."""
    return {"incidents": await client.get_incidents()}


async def get_controller_status(client: OrchestratorClient) -> dict[str, object]:
    """Which allocator is live: a trained policy, or the static baseline.

    Check this before explaining "policy behaviour". If ``trained_policy_loaded`` is false, the behaviour
    being described is a fixed baseline, and saying otherwise would be wrong.
    """
    return await client.get_controller_health()


async def get_flow_detail(client: OrchestratorClient, flow_id: str) -> dict[str, object]:
    """One flow's demand, allocation, drops, path and latency, plus the utilization of every link it
    traverses. The tool to reach for when an operator asks about a specific flow or node."""
    payload = await client.get_network_state()
    flows = payload.get("flows", [])
    links = payload.get("links", [])
    if not isinstance(flows, list) or not isinstance(links, list):
        raise ToolExecutionError("Network state had an unexpected shape.", {})

    flow = next((item for item in flows if item.get("flow_id") == flow_id), None)
    if flow is None:
        raise ToolExecutionError(
            f"No flow {flow_id!r} in the current network. Tell the operator which flows exist rather "
            f"than guessing at one.",
            {"requested": flow_id, "available": [item.get("flow_id") for item in flows]},
        )

    path = flow.get("path", [])
    path_links = [link for link in links if link.get("link_id") in path]
    return {"flow": flow, "path_links": path_links, "timestep": payload.get("timestep")}
