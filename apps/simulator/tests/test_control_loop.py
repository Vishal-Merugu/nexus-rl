"""Control-loop resilience tests.

The loop must survive every way the controller can misbehave. An unhandled exception here kills the task
while the service keeps answering requests — the simulation freezes and `/health` still reports ok, which
is the single worst failure mode this system has.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from nexus_sim.simulator import NetworkSimulator
from nexus_simulator.control_loop import ControlLoop
from nexus_simulator.service import SimulationService


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _loop(handler: object) -> tuple[ControlLoop, SimulationService]:
    service = SimulationService(NetworkSimulator.from_config("mesh-6", "steady", seed=1))
    loop = ControlLoop(
        service,
        "http://controller",
        tick_seconds=0.01,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),  # type: ignore[arg-type]
    )
    return loop, service


async def _run_briefly(loop: ControlLoop) -> None:
    await loop.start()
    await asyncio.sleep(0.2)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("label", "response"),
    [
        ("server error", httpx.Response(500, json={"detail": "boom"})),
        ("wrong schema", httpx.Response(200, json={"nonsense": True})),
        # The one that actually killed the loop: the model validator raises ActionDecodingError, a
        # NexusError rather than a ValueError, so the original except clause missed it entirely.
        ("out-of-range share", httpx.Response(200, json={"share_by_flow_id": {"f0": 7.5}})),
        ("not json at all", httpx.Response(200, content=b"<html>nope</html>")),
        ("empty body", httpx.Response(200, json={})),
    ],
)
async def test_the_loop_keeps_stepping_through_controller_failures(
    label: str, response: httpx.Response
) -> None:
    loop, service = _loop(lambda _request: response)
    try:
        await _run_briefly(loop)

        assert loop.alive, f"loop died on: {label}"
        assert service.state.timestep > 0, f"simulation froze on: {label}"
        assert loop.controller_reachable is False
    finally:
        await loop.stop()


@pytest.mark.anyio
async def test_a_healthy_controller_is_reported_reachable() -> None:
    loop, service = _loop(lambda _request: httpx.Response(200, json={"share_by_flow_id": {"f0": 0.5}}))
    try:
        await _run_briefly(loop)

        assert loop.controller_reachable is True
        assert loop.last_error is None
        assert service.state.timestep > 0
    finally:
        await loop.stop()


@pytest.mark.anyio
async def test_reachability_is_not_claimed_before_the_first_tick() -> None:
    # Optimistic health during startup is misleading exactly when a demo is coming up.
    loop, _ = _loop(lambda _request: httpx.Response(200, json={"share_by_flow_id": {}}))

    assert loop.controller_reachable is False
    assert loop.alive is False


@pytest.mark.anyio
async def test_a_failure_is_recorded_for_health_to_report() -> None:
    loop, _ = _loop(lambda _request: httpx.Response(500, json={"detail": "boom"}))
    try:
        await _run_briefly(loop)

        assert loop.last_error is not None
    finally:
        await loop.stop()


@pytest.mark.anyio
async def test_the_loop_recovers_when_the_controller_returns() -> None:
    state = {"failing": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["failing"]:
            return httpx.Response(503, json={"detail": "starting"})
        return httpx.Response(200, json={"share_by_flow_id": {"f0": 0.5}})

    loop, _ = _loop(handler)
    try:
        await _run_briefly(loop)
        assert loop.controller_reachable is False

        state["failing"] = False
        await asyncio.sleep(0.1)

        # Read through explicitly-annotated locals: the earlier `is False` assert narrows the attribute
        # to Literal[False], and mypy has no way to know the background loop mutates it in between.
        reachable_after_recovery: bool = loop.controller_reachable
        error_after_recovery: str | None = loop.last_error
        assert reachable_after_recovery is True
        assert error_after_recovery is None
    finally:
        await loop.stop()


def test_a_non_positive_tick_is_refused() -> None:
    # tick_seconds=0 spun the loop as fast as the CPU allowed — 518 timesteps in 0.15s.
    from nexus_errors.config_error import ConfigurationError

    service = SimulationService(NetworkSimulator.from_config("mesh-6", "steady", seed=1))
    for tick in (0.0, -1.0):
        with pytest.raises(ConfigurationError, match="tick_seconds must be positive"):
            ControlLoop(service, "http://c", tick_seconds=tick)
