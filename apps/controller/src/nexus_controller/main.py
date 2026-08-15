"""Controller service HTTP surface.

Two responsibilities: serve allocations from the loaded policy, and own the constraints the agent layer
writes. Everything else — simulation, LLM reasoning, rendering — belongs elsewhere.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from nexus_config.settings import Settings, get_settings
from nexus_controller.class_floors import apply_class_floors
from nexus_controller.constraint_store import ConstraintStore
from nexus_controller.policy_host import PolicyHost, build_default_host, describe_fallback_reason
from nexus_errors.control_error import ConstraintRejectedError
from nexus_mdp.version import MDP_VERSION
from nexus_telemetry.client import get_telemetry
from nexus_types.action import AllocationAction
from nexus_types.constants import DEFAULT_HISTORY_SIZE
from nexus_types.constraint import Constraint, ConstraintSet
from nexus_types.network_state import NetworkState

_telemetry = get_telemetry("nexus.controller")

_host: PolicyHost | None = None
_constraints = ConstraintStore()
_decisions: deque[dict[str, object]] = deque(maxlen=DEFAULT_HISTORY_SIZE)
"""Recent decisions, for agent explanations. Bounded — an explanation aid, not a datastore."""

_last_timestep: int = 0
"""Simulation time as of the last `/act`. The controller's only notion of "now".

Read paths use this rather than accepting a timestep argument, so a caller cannot make the controller
reason about — or garbage-collect against — a moment that never happened.
"""


def get_host() -> PolicyHost:
    if _host is None:
        raise HTTPException(status_code=503, detail="Controller is still starting.")
    return _host


def require_api_key(
    settings: Settings = Depends(get_settings),
    x_nexus_key: str | None = Header(default=None),
) -> None:
    """Guard constraint writes outside development."""
    if settings.is_development:
        return
    if x_nexus_key != settings.require_api_key():
        raise HTTPException(status_code=401, detail="Invalid or missing X-Nexus-Key.")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _host
    settings = get_settings()

    # The controller holds no model of the network. Everything needed to decode an action — the flow set
    # and each flow's candidate paths — arrives in the state passed to /act, so the controller cannot
    # disagree with the simulator about what it is controlling.
    _host = build_default_host()
    _host.try_load(settings.policy_path)

    _telemetry.info("Controller ready", {"policy": _host.policy_label, "mdp_version": MDP_VERSION})
    yield
    _telemetry.info("Controller stopped", {})


app = FastAPI(title="NEXUS-RL Controller", version="0.1.0", lifespan=lifespan)


class HealthResponse(BaseModel):
    """Reports which allocator is live — check this before debugging 'odd policy behaviour'."""

    status: str
    policy: str
    """'dqn:policy@mdp1' when a checkpoint is serving, 'baseline:static' when it is not."""
    trained_policy_loaded: bool
    fallback_reason: str | None
    mdp_version: int
    last_timestep: int
    """Simulation time as of the last `/act`. Stuck at 0 means the control loop is not reaching us."""
    active_constraints: int
    stored_constraints: int
    """Includes any not yet expired-and-collected. Reported alongside `active` so a mismatch is visible."""


class ActRequest(BaseModel):
    state: NetworkState


class DecisionRecord(BaseModel):
    """One past decision, as the agent layer's explanation tool reads it."""

    timestep: int
    policy: str
    shares: dict[str, float]
    active_constraint_ids: list[str] = Field(default_factory=list)


@app.get("/health", response_model=HealthResponse)
def health(
    host: PolicyHost = Depends(get_host), settings: Settings = Depends(get_settings)
) -> HealthResponse:
    # Read-only: snapshot() does not mutate. Health used to prune here, which meant a liveness probe
    # deleted every constraint issued after the last decision.
    return HealthResponse(
        status="ok",
        policy=host.policy_label,
        trained_policy_loaded=host.has_trained_policy,
        fallback_reason=(None if host.has_trained_policy else describe_fallback_reason(settings.policy_path)),
        mdp_version=MDP_VERSION,
        last_timestep=_last_timestep,
        active_constraints=len(_constraints.snapshot(_last_timestep).active()),
        stored_constraints=_constraints.stored_count(),
    )


@app.post("/act", response_model=AllocationAction)
def act(request: ActRequest, host: PolicyHost = Depends(get_host)) -> AllocationAction:
    """Produce an allocation for the given state.

    Constraints are resolved here, at decision time, rather than being baked in earlier: a constraint that
    expired between issue and use must not still apply, and resolving late is the only way to guarantee it.
    """
    global _last_timestep
    state = request.state
    _last_timestep = state.timestep

    # The control path is the only place expiry is garbage-collected: this is the one caller whose
    # timestep is real, advancing simulation time.
    _constraints.prune(state.timestep)
    forbidden = _constraints.forbidden_link_ids(state.timestep)
    action = host.act(state, forbidden_link_ids=forbidden)
    # The third constraint kind. Applied after the policy acts: the controller still chooses the
    # allocation, and the floor only lifts a starved class toward its guarantee.
    action = apply_class_floors(action, state, _constraints.class_floors(state.timestep))

    _decisions.append(
        {
            "timestep": state.timestep,
            "policy": host.policy_label,
            "shares": dict(action.share_by_flow_id),
            "active_constraint_ids": [
                constraint.constraint_id for constraint in _constraints.snapshot(state.timestep).active()
            ],
        }
    )
    return action


@app.get("/decisions", response_model=list[DecisionRecord])
def decisions(limit: int = 20) -> list[DecisionRecord]:
    """Recent decisions, oldest first. Backs the agent layer's 'why did you do that' explanations."""
    return [DecisionRecord(**record) for record in list(_decisions)[-limit:]]


@app.get("/constraints", response_model=ConstraintSet)
def list_constraints(timestep: int | None = None) -> ConstraintSet:
    """Constraints in force at a timestep, defaulting to the controller's current time.

    Purely a read: nothing is deleted. The previous version defaulted to timestep 0 and pruned, so a
    parameterless call destroyed every constraint issued after the start of the run.

    Returns only constraints actually in force — a client reading the ``constraints`` field must not see
    expired ones, or the agent layer will report lapsed rules as active.
    """
    return _constraints.active_snapshot(_last_timestep if timestep is None else timestep)


@app.post("/constraints", response_model=Constraint, dependencies=[Depends(require_api_key)])
def add_constraint(constraint: Constraint, current_timestep: int | None = None) -> Constraint:
    """Accept a constraint from the agent layer.

    This is the **only** write the outer tier has. There is no endpoint here that accepts an allocation —
    that boundary is the project's central design claim (ADR-0002), and it is enforced by the absence of a
    route, not by a check inside one.
    """
    try:
        return _constraints.add(constraint, _last_timestep if current_timestep is None else current_timestep)
    except ConstraintRejectedError as error:
        # Surfaced as an error so the agent reports the refusal rather than assuming success. An empty
        # success here is how the model ends up telling an operator a change happened that did not.
        _telemetry.capture_exception(error, "Constraint rejected")
        raise HTTPException(status_code=422, detail=error.to_dict()) from error


@app.delete("/constraints/{constraint_id}")
def revoke_constraint(constraint_id: str) -> dict[str, bool]:
    """Revoke a constraint early. Constraints also expire on their own — this is for 'never mind'."""
    return {"removed": _constraints.remove(constraint_id)}
