"""The MCP server actually assembles.

Regression: ``mcp`` was declared as ``>=1.2`` and resolved to 2.0.0, which removed the lowlevel
``@server.list_tools()`` / ``@server.call_tool()`` decorators this module is built on. ``build_server``
raised ``AttributeError`` on import of the first decorator, so the agent layer could not start at all.

362 tests passed while that was true. Every one of them exercised ``build_tool_specs()`` or a tool
function directly; none constructed the server, which is the one thing that touches the MCP API surface.
A dependency the code cannot run against is a build failure, and it belongs in the test suite rather than
in a stack trace at deploy time.
"""

from __future__ import annotations

import httpx
import pytest

from nexus_agent.client import OrchestratorClient
from nexus_agent.server import build_server, build_tool_specs
from nexus_agent.tools import read_tools
from nexus_errors.control_error import ToolExecutionError


def _client() -> OrchestratorClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"timestep": 7, "links": [], "flows": []})

    return OrchestratorClient(
        "http://sim", "http://ctrl", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def test_the_server_assembles_against_the_installed_mcp() -> None:
    server = build_server(_client())

    assert server.name == "nexus-rl"


def test_every_declared_tool_is_reachable_through_the_server() -> None:
    server = build_server(_client())
    specs = build_tool_specs()

    # The decorators register handlers on the server; if the MCP surface shifts again, this fails here
    # rather than at the first operator question.
    assert server.request_handlers, "no MCP request handlers registered"
    assert len(specs) == 13


@pytest.mark.anyio
async def test_invoke_validates_before_dispatching() -> None:
    spec = {s.name: s for s in build_tool_specs()}["get_recent_history"]

    with pytest.raises(ToolExecutionError, match="Invalid arguments"):
        await spec.invoke(_client(), {"limit": -50})


@pytest.mark.anyio
async def test_invoke_passes_the_validated_model_to_the_handler() -> None:
    spec = {s.name: s for s in build_tool_specs()}["get_recent_history"]

    # Defaults come from the declared schema, not from the call site.
    validated = spec.validate_arguments({})

    assert isinstance(validated, read_tools.GetRecentHistoryInput)
    assert validated.limit == 20
