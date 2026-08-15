"""The control loop — simulator drives, controller decides.

Runs as a background task inside the simulator service. Each tick: read state, ask the controller for an
action, apply it, repeat.

The simulator drives rather than the controller because the simulator is the single writer of state. If the
controller drove, two clients calling it concurrently would interleave writes and produce a trajectory no
seed can reproduce.

**The degraded path is designed first.** If the controller is slow, erroring, or absent, the loop applies no
action — every flow gets its full demand — and keeps stepping. The network degrades to unmanaged operation
rather than freezing. A stalled simulation looks like a crashed system; an unmanaged one looks like exactly
what it is, and the dashboard can say so.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

import httpx

from nexus_errors.base_error import NexusError, error_to_string
from nexus_errors.config_error import ConfigurationError
from nexus_simulator.service import SimulationService
from nexus_telemetry.client import get_telemetry
from nexus_types.action import AllocationAction

_telemetry = get_telemetry("nexus.simulator.loop")

CONTROLLER_TIMEOUT_SECONDS = 2.0
"""Short on purpose. The control loop must not block on a slow controller — it steps without it instead."""

WARN_AFTER_CONSECUTIVE_FAILURES = 5
"""Log once when the controller goes away, then stay quiet. A per-tick error log at 10 Hz is unreadable."""


class ControlLoop:
    """Advances the simulation, consulting the controller each tick."""

    def __init__(
        self,
        service: SimulationService,
        controller_url: str,
        tick_seconds: float = 0.1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if tick_seconds <= 0:
            raise ConfigurationError(
                "tick_seconds must be positive; zero spins the control loop as fast as the CPU allows.",
                {"tick_seconds": tick_seconds},
            )
        self._service = service
        self._controller_url = controller_url.rstrip("/")
        self._tick_seconds = tick_seconds
        self._client = client or httpx.AsyncClient(timeout=CONTROLLER_TIMEOUT_SECONDS)
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._consecutive_failures = 0
        self._last_action_ok: bool | None = None
        self._last_error: str | None = None

    @property
    def controller_reachable(self) -> bool:
        """Whether the last tick actually got an action.

        False before the first tick: until one has run we do not know the controller is reachable, and
        claiming otherwise makes /health optimistic exactly when a demo is starting up.
        """
        return self._last_action_ok is True

    @property
    def alive(self) -> bool:
        """Whether the loop task is still running.

        Surfaced on /health because a dead loop is invisible otherwise — the service answers requests
        normally while the simulation sits frozen.
        """
        return self._task is not None and not self._task.done()

    @property
    def last_error(self) -> str | None:
        """Most recent failure, so /health can say *why* the loop is degraded."""
        return self._last_error

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        _telemetry.info("Control loop started", {"controller_url": self._controller_url})

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task  # cancellation is the expected path on shutdown
            self._task = None
        await self._client.aclose()

    async def _run(self) -> None:
        """Tick forever.

        Nothing short of cancellation may end this loop. An unhandled exception here kills the task while
        the service keeps answering requests, so the simulation freezes and /health still reports ok — the
        exact failure this module exists to avoid. A malformed action used to do precisely that.
        """
        while self._running:
            try:
                action = await self._request_action()
                await self._service.step(action)
            except NexusError as error:
                # Stepping failed (e.g. an invariant violation). Log and keep ticking; the next timestep
                # may well succeed, and a frozen simulation is worse than a logged bad one.
                self._last_error = error_to_string(error)
                _telemetry.capture_exception(error, "Control loop step failed — continuing")
            await asyncio.sleep(self._tick_seconds)

    async def _request_action(self) -> AllocationAction | None:
        """Ask the controller for an action. ``None`` on any failure — the loop keeps going regardless."""
        try:
            response = await self._client.post(
                f"{self._controller_url}/act",
                json={"state": self._service.state.model_dump(mode="json")},
            )
            response.raise_for_status()
            action = AllocationAction.model_validate(response.json())
        # NexusError matters here: AllocationAction's validator raises ActionDecodingError, which is not a
        # ValueError. Omitting it let one malformed response kill the loop permanently.
        except (httpx.HTTPError, ValueError, NexusError) as error:
            self._consecutive_failures += 1
            self._last_action_ok = False
            self._last_error = error_to_string(error)
            if self._consecutive_failures == WARN_AFTER_CONSECUTIVE_FAILURES:
                _telemetry.warning(
                    "Controller unreachable — stepping unmanaged. The network is running without a "
                    "policy; the dashboard will show full-demand allocation.",
                    {"controller_url": self._controller_url, "error": error_to_string(error)},
                )
            return None

        if self._consecutive_failures >= WARN_AFTER_CONSECUTIVE_FAILURES:
            _telemetry.info("Controller reachable again", {"controller_url": self._controller_url})
        self._consecutive_failures = 0
        self._last_action_ok = True
        self._last_error = None
        return action
