"""Simulator service HTTP surface.

Thin by design: parse, delegate, serialize. Simulation math lives in ``nexus-sim``, state ownership in
``service.py``. If logic starts appearing in a handler here, it belongs in a package instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from nexus_config.settings import Settings, get_settings
from nexus_errors.base_error import NexusError
from nexus_sim.simulator import NetworkSimulator
from nexus_simulator.control_loop import ControlLoop
from nexus_simulator.service import SimulationService
from nexus_telemetry.client import get_telemetry
from nexus_types.constants import DASHBOARD_DOWNSAMPLE
from nexus_types.incident import Incident
from nexus_types.network_state import NetworkState

_telemetry = get_telemetry("nexus.simulator")

_service: SimulationService | None = None
_loop: ControlLoop | None = None


def get_service() -> SimulationService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Simulator is still starting.")
    return _service


def require_api_key(
    settings: Settings = Depends(get_settings),
    x_nexus_key: str | None = Header(default=None),
) -> None:
    """Guard mutating endpoints outside development.

    Injecting an incident or resetting the simulation changes what every other service sees, so it is not
    something an unauthenticated caller should be able to do in a deployed environment.
    """
    if settings.is_development:
        return
    if x_nexus_key != settings.require_api_key():
        raise HTTPException(status_code=401, detail="Invalid or missing X-Nexus-Key.")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _service, _loop
    settings = get_settings()

    simulator = NetworkSimulator.from_config(
        topology_name=settings.topology,
        traffic_pattern_name="diurnal",
        seed=settings.seed,
    )
    _service = SimulationService(simulator)
    _loop = ControlLoop(_service, controller_url=settings.controller_url)
    await _loop.start()

    _telemetry.info(
        "Simulator ready",
        {"topology": settings.topology, "seed": settings.seed, "links": len(simulator.topology.links)},
    )
    yield

    await _loop.stop()
    _telemetry.info("Simulator stopped", {})


app = FastAPI(title="NEXUS-RL Simulator", version="0.1.0", lifespan=lifespan)


class HealthResponse(BaseModel):
    """Health that actually says what the service is doing.

    A check that returns 200 unconditionally is a liability: during a demo it will report healthy while the
    control loop sits disconnected and the network runs unmanaged.
    """

    status: str
    timestep: int
    topology: str
    controller_reachable: bool
    control_loop_alive: bool
    """False means the simulation is frozen. Without this the service answers normally while nothing runs."""
    last_error: str | None
    flows: int
    links: int


class StepRequest(BaseModel):
    action: dict[str, object] | None = Field(
        default=None, description="Optional allocation. Omitted means every flow gets its full demand."
    )


@app.get("/health", response_model=HealthResponse)
def health(service: SimulationService = Depends(get_service)) -> HealthResponse:
    state = service.state
    return HealthResponse(
        status="ok",
        timestep=state.timestep,
        topology=service.simulator.topology.name,
        controller_reachable=_loop.controller_reachable if _loop is not None else False,
        control_loop_alive=_loop.alive if _loop is not None else False,
        last_error=_loop.last_error if _loop is not None else None,
        flows=len(state.flows),
        links=len(state.links),
    )


@app.get("/state", response_model=NetworkState)
def current_state(service: SimulationService = Depends(get_service)) -> NetworkState:
    """The current network state. Join everything else on its ``timestep``."""
    return service.state


@app.get("/metrics")
def current_metrics(service: SimulationService = Depends(get_service)) -> dict[str, float]:
    return service.metrics.as_dict()


@app.get("/history")
def history(limit: int = 50, service: SimulationService = Depends(get_service)) -> list[dict[str, object]]:
    """Recent timesteps, oldest first. Bounded — this is an explanation aid, not a datastore."""
    return [
        {
            "timestep": entry.state.timestep,
            "metrics": entry.metrics.as_dict(),
            "issued_by": entry.issued_by,
            "active_incident_ids": list(entry.state.active_incident_ids),
        }
        for entry in service.history(limit=limit)
    ]


@app.post("/reset", response_model=NetworkState, dependencies=[Depends(require_api_key)])
async def reset(service: SimulationService = Depends(get_service)) -> NetworkState:
    return await service.reset()


@app.post("/incidents", response_model=Incident, dependencies=[Depends(require_api_key)])
async def inject_incident(incident: Incident, service: SimulationService = Depends(get_service)) -> Incident:
    """Schedule a degradation — Use Case 4.

    Incidents are environment, not action space. The controller reacts; it cannot cause or clear one.
    """
    _telemetry.info("Incident injected", {"incident": incident.model_dump(mode="json")})
    return await service.inject_incident(incident)


@app.get("/incidents", response_model=list[Incident])
def active_incidents(service: SimulationService = Depends(get_service)) -> list[Incident]:
    return list(service.active_incidents())


def should_send_frame(timestep: int, last_sent: int | None) -> bool:
    """Whether the dashboard should receive this timestep.

    Extracted so it is testable without a live socket — the logic has three cases and two of them were
    wrong: `None` sends the first frame immediately instead of waiting a whole downsample window, and
    `timestep < last_sent` detects a simulator reset. Without the reset case the stream went permanently
    silent after `/reset`, because the timestep never climbed back past the old high-water mark.
    """
    return last_sent is None or timestep < last_sent or timestep >= last_sent + DASHBOARD_DOWNSAMPLE


@app.websocket("/ws")
async def stream_state(websocket: WebSocket) -> None:
    """Stream downsampled state to the dashboard.

    Downsampled because the simulator steps far faster than a browser can paint. Streaming every timestep
    gains nothing visually and stalls the client.
    """
    await websocket.accept()
    service = get_service()
    last_sent: int | None = None
    try:
        while True:
            state = service.state
            if should_send_frame(state.timestep, last_sent):
                await websocket.send_json(
                    {
                        "state": state.model_dump(mode="json"),
                        "metrics": service.metrics.as_dict(),
                    }
                )
                last_sent = state.timestep
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        _telemetry.info("Dashboard disconnected", {"last_timestep": last_sent})
    except NexusError as error:
        _telemetry.capture_exception(error, "WebSocket stream failed")
        with suppress(RuntimeError):
            await websocket.close(code=1011)
