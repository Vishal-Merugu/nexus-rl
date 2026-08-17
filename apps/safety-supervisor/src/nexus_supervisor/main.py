"""Safety supervisor service — an independent guardrail in the fast control path.

**It speaks the controller's own `/act` contract.** That is the whole deployment story: point the
simulator's ``NEXUS_CONTROLLER_URL`` at this service instead, and it sits in the path with no change to
either neighbour. The controller does not know it exists and cannot be made to bypass it.

Why a separate service rather than a module inside the controller (v2 spec, Decision 9): if the guardrail
lives inside the thing it guards, a bug in the policy-serving code can bypass it. Independence *is* the
value — a supervisor that shares a process with the policy shares its failure modes.

The cost is honest and worth stating: one more network hop in a loop that must be fast, and one more
service that can fail. The hop is a local call against a simulator ticking at hundreds of milliseconds,
and the failure mode is designed for — if the supervisor cannot reach the controller it runs the
deterministic allocator rather than passing the problem on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from nexus_config.settings import Settings, get_settings
from nexus_errors.base_error import error_to_string
from nexus_errors.control_error import ActionDecodingError
from nexus_safety.bounds import ActionBounds
from nexus_safety.fallback import QosThresholds
from nexus_safety.proposal import ProposedAction, parse_proposed_action
from nexus_supervisor.supervisor import SafetySupervisor
from nexus_telemetry.client import get_telemetry
from nexus_types.action import AllocationAction
from nexus_types.network_state import NetworkState

_telemetry = get_telemetry("nexus.supervisor.service")

REQUEST_TIMEOUT_SECONDS = 2.0
"""Deliberately tight. This sits in the fast loop: waiting five seconds for a controller that is not
answering is itself a failure, and the fallback allocator is right there."""

_supervisor = SafetySupervisor()
_client: httpx.AsyncClient | None = None
_controller_url = ""
_shadow_url = ""
"""A second controller serving the shadow-staged policy version, or empty.

Empty is the normal state. Shadow mode is opt-in because it doubles the calls in the fast loop, and a
feature that silently costs a control-loop round trip is one nobody can budget for."""


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _client, _controller_url, _shadow_url, _supervisor
    settings = get_settings()

    _controller_url = settings.controller_url.rstrip("/")
    _shadow_url = (settings.shadow_controller_url or "").rstrip("/")
    _client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
    _supervisor = SafetySupervisor(
        bounds=ActionBounds(
            starvation_share=settings.safety_starvation_share,
            starvation_window=settings.safety_starvation_window,
        ),
        thresholds=QosThresholds(
            max_p95_latency_ms=settings.safety_max_p95_latency_ms,
            max_drop_rate=settings.safety_max_drop_rate,
            min_fairness=settings.safety_min_fairness,
            consecutive_breaches_to_engage=settings.safety_breaches_to_engage,
            consecutive_healthy_to_release=settings.safety_healthy_to_release,
        ),
    )

    _telemetry.info(
        "Safety supervisor ready",
        {"controller_url": _controller_url, "shadow_url": _shadow_url or None},
    )
    yield
    if _client is not None:
        await _client.aclose()
    _telemetry.info("Safety supervisor stopped", {})


app = FastAPI(title="NEXUS-RL Safety Supervisor", version="0.1.0", lifespan=lifespan)


class ActRequest(BaseModel):
    """Identical to the controller's. The supervisor is transparent to its callers by construction."""

    state: NetworkState


class HealthResponse(BaseModel):
    status: str
    controller_url: str
    fallback_engaged: bool
    """True means the deterministic allocator is driving, not the policy. Check this first."""
    fallback_reason: str
    activation_count: int
    supervised_timesteps: int


@app.post("/act", response_model=AllocationAction)
async def act(request: ActRequest) -> AllocationAction:
    """Get an action for one timestep — checked, corrected, or replaced.

    Always returns an action. There is no failure mode in which the simulator receives nothing: that path
    made the network run unmanaged at full demand, which is precisely what this service exists to replace
    with a deliberate, deterministic fallback.
    """
    proposal = await _propose(request.state, _controller_url)
    # The candidate is queried after the incumbent and its result reaches only the comparator. There is
    # no path from here into the returned action — shadow mode's guarantee is structural, not a flag.
    shadow_proposal = await _propose(request.state, _shadow_url, shadow=True) if _shadow_url else None
    decision = _supervisor.supervise(
        request.state, proposal, shadow_proposal, shadow_configured=bool(_shadow_url)
    )
    return decision.action


@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    del settings
    status = _supervisor.status()
    return HealthResponse(
        status="ok",
        controller_url=_controller_url,
        fallback_engaged=bool(status["fallback_engaged"]),
        fallback_reason=str(status["reason"]),
        activation_count=int(str(status["activation_count"])),
        supervised_timesteps=int(str(status["supervised_timesteps"])),
    )


@app.get("/safety")
def safety() -> dict[str, object]:
    """Policy-level observability — violation rates, corrections, fallback activations, drift.

    Deliberately not merged into ``/health``. A liveness probe asks "is this process up"; this asks "is
    the policy behaving", and the second question is the one nobody thinks to instrument.
    """
    return _supervisor.status()


@app.get("/shadow")
def shadow() -> dict[str, object]:
    """The shadow candidate's readiness verdict — Use Case 10.

    Reports readiness, never superiority. The candidate's actions were never applied, so the network never
    responded to them and no outcome here is attributable to it. Which policy is *better* is the offline
    evaluation suite's verdict, on data where both were really applied.
    """
    if not _shadow_url:
        return {"shadow_configured": False, "detail": "NEXUS_SHADOW_CONTROLLER_URL is unset."}
    return {"shadow_configured": True, "shadow_url": _shadow_url, **_supervisor.shadow_report()}


async def _propose(state: NetworkState, url: str, shadow: bool = False) -> ProposedAction | None:
    """Ask the controller for an action, parsed leniently. ``None`` on any failure.

    Lenient on purpose: ``AllocationAction`` rejects out-of-range and non-finite shares in its own
    validator, so parsing into it here would turn every interesting policy failure into a transport error
    the supervisor could only report as "malformed response". The guardrail has to see the bad number to
    say anything useful about it.
    """
    if _client is None or not url:
        return None
    try:
        response = await _client.post(f"{url}/act", json={"state": state.model_dump(mode="json")})
        response.raise_for_status()
        return parse_proposed_action(response.json())
    except (httpx.HTTPError, ValueError, ActionDecodingError) as error:
        if shadow:
            # A failing candidate must never disturb live traffic — it counts against its own readiness
            # verdict and nothing else.
            _telemetry.info(
                "Shadow candidate did not answer; recorded against its readiness verdict",
                {"shadow_url": url, "error": error_to_string(error)},
            )
            return None
        _telemetry.warning(
            "No usable action from the controller — deterministic allocator will drive this timestep",
            {"controller_url": url, "error": error_to_string(error)},
        )
        return None
