"""HTTP client for the simulator and controller.

Every MCP tool goes through here. The single rule this module exists to enforce: **a failed call raises**.
It never returns an empty dict, a default, or a partial payload.

Given nothing, a language model produces a fluent, confident, entirely invented answer that is
indistinguishable from a real one — and the whole value of the agent layer is that its explanations cite
numbers that actually came from the system.
"""

from __future__ import annotations

import httpx

from nexus_errors.base_error import error_to_string
from nexus_errors.control_error import ToolExecutionError
from nexus_telemetry.client import get_telemetry

_telemetry = get_telemetry("nexus.agent.client")

REQUEST_TIMEOUT_SECONDS = 5.0

type QueryParams = dict[str, str | int]
"""Query-string values this client actually sends. Narrower than ``dict[str, object]`` because httpx
accepts only scalars here, and widening it would push the error to the wire instead of the call site."""


class OrchestratorClient:
    """Reads simulator and controller state; writes constraints to the controller."""

    def __init__(
        self,
        simulator_url: str,
        controller_url: str,
        client: httpx.AsyncClient | None = None,
        api_key: str = "",
    ) -> None:
        self._simulator_url = simulator_url.rstrip("/")
        self._controller_url = controller_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
        self._api_key = api_key

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_network_state(self) -> dict[str, object]:
        return await self._get(f"{self._simulator_url}/state", "current network state")

    async def get_metrics(self) -> dict[str, object]:
        return await self._get(f"{self._simulator_url}/metrics", "current QoS metrics")

    async def get_history(self, limit: int = 20) -> list[dict[str, object]]:
        payload = await self._request(
            "GET", f"{self._simulator_url}/history", "recent network history", params={"limit": limit}
        )
        return _expect_list_of_mappings(payload, "recent network history")

    async def get_incidents(self) -> list[dict[str, object]]:
        payload = await self._request("GET", f"{self._simulator_url}/incidents", "active incidents")
        return _expect_list_of_mappings(payload, "active incidents")

    async def get_controller_health(self) -> dict[str, object]:
        return await self._get(f"{self._controller_url}/health", "controller health")

    async def get_decisions(self, limit: int = 20) -> list[dict[str, object]]:
        payload = await self._request(
            "GET",
            f"{self._controller_url}/decisions",
            "recent controller decisions",
            params={"limit": limit},
        )
        return _expect_list_of_mappings(payload, "recent controller decisions")

    async def get_constraints(self, timestep: int) -> dict[str, object]:
        return await self._get(
            f"{self._controller_url}/constraints", "active constraints", params={"timestep": timestep}
        )

    async def post_constraint(
        self, constraint: dict[str, object], current_timestep: int
    ) -> dict[str, object]:
        """Write a constraint. The only write the agent layer has."""
        return await self._post(
            f"{self._controller_url}/constraints",
            constraint,
            "apply constraint",
            params={"current_timestep": current_timestep},
        )

    async def revoke_constraint(self, constraint_id: str) -> dict[str, object]:
        return await self._delete(f"{self._controller_url}/constraints/{constraint_id}", "revoke constraint")

    async def _get(self, url: str, description: str, params: QueryParams | None = None) -> dict[str, object]:
        return _expect_mapping(await self._request("GET", url, description, params=params), description)

    async def _post(
        self,
        url: str,
        payload: dict[str, object],
        description: str,
        params: QueryParams | None = None,
    ) -> dict[str, object]:
        response = await self._request("POST", url, description, json=payload, params=params)
        return _expect_mapping(response, description)

    async def _delete(self, url: str, description: str) -> dict[str, object]:
        return _expect_mapping(await self._request("DELETE", url, description), description)

    async def _request(
        self,
        method: str,
        url: str,
        description: str,
        json: dict[str, object] | None = None,
        params: QueryParams | None = None,
    ) -> object:
        """Perform the request and return **parsed JSON of unknown shape**.

        Deliberately ``object`` rather than ``dict[str, object]``: ``/history``, ``/incidents`` and
        ``/decisions`` all return JSON arrays. Declaring a mapping here made the callers' own
        ``isinstance(payload, list)`` guards statically unreachable — the annotation contradicted the
        endpoints. Callers narrow through ``_expect_mapping`` / ``_expect_list_of_mappings``.
        """
        headers = {"X-Nexus-Key": self._api_key} if self._api_key else {}
        try:
            response = await self._client.request(method, url, json=json, params=params, headers=headers)
            response.raise_for_status()
            parsed: object = response.json()
            return parsed
        except httpx.HTTPStatusError as error:
            detail = _safe_detail(error.response)
            _telemetry.capture_exception(error, f"Failed to {description}")
            raise ToolExecutionError(
                f"Could not {description}: the service returned {error.response.status_code}. "
                f"Report this to the operator — do not answer from memory.",
                {"url": url, "status": error.response.status_code, "detail": detail},
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            _telemetry.capture_exception(error, f"Failed to {description}")
            raise ToolExecutionError(
                f"Could not {description}: {error_to_string(error)}. Report this to the operator — do "
                f"not answer from memory.",
                {"url": url},
            ) from error


def _expect_mapping(payload: object, description: str) -> dict[str, object]:
    """Narrow parsed JSON to an object, or fail loudly.

    A shape surprise has to raise rather than coerce: handing the model a silently-emptied payload is the
    empty-success failure this whole module exists to prevent.
    """
    if not isinstance(payload, dict):
        raise ToolExecutionError(
            f"The {description} endpoint returned a JSON {type(payload).__name__}, not an object.",
            {"got": type(payload).__name__},
        )
    return {str(key): value for key, value in payload.items()}


def _expect_list_of_mappings(payload: object, description: str) -> list[dict[str, object]]:
    """Narrow parsed JSON to a list of objects, or fail loudly."""
    if not isinstance(payload, list):
        raise ToolExecutionError(
            f"The {description} endpoint returned a JSON {type(payload).__name__}, not a list.",
            {"got": type(payload).__name__},
        )
    return [_expect_mapping(item, description) for item in payload]


def _safe_detail(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text[:200]
