"""MCP server exposing the orchestrator to an LLM.

Read tools are unrestricted. Write tools produce constraints and nothing else. The registry below is the
complete surface the model can reach — if a capability is not here, the model cannot do it, which is a
stronger guarantee than any instruction in the prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import BaseModel, ValidationError

from nexus_agent.client import OrchestratorClient
from nexus_agent.tools import read_tools, write_tools
from nexus_config.settings import get_settings
from nexus_errors.base_error import NexusError
from nexus_errors.control_error import ToolExecutionError
from nexus_telemetry.client import get_telemetry

_telemetry = get_telemetry("nexus.agent.server")


class ToolInvoker(Protocol):
    """The non-generic face of a tool spec — what the registry and the MCP server actually need.

    ``ToolSpec`` is generic in its input model so each handler receives its own validated type. That
    generic parameter would otherwise make the registry heterogeneous and force every dispatch site back
    to ``object``. Keeping it off this protocol lets the specs live in one tuple while each handler stays
    precisely typed.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def writes(self) -> bool: ...

    @property
    def input_schema(self) -> dict[str, object]: ...

    def validate_arguments(self, arguments: dict[str, object]) -> BaseModel: ...

    def invoke(
        self, client: OrchestratorClient, arguments: dict[str, object]
    ) -> Awaitable[dict[str, object]]: ...


@dataclass(frozen=True)
class ToolSpec[TInput: BaseModel]:
    """One MCP tool: its schema, its handler, and whether it can change anything.

    Generic in the input model so ``handler`` receives a validated ``TInput`` rather than a raw dict.
    Handing the handler an untyped dict meant re-coercing every field at the call site — ``int(args[...])``
    over ``object`` — which discarded the very types validation had just established and needed a
    ``type: ignore`` per argument to compile.
    """

    name: str
    description: str
    input_model: type[TInput]
    """The schema is **enforced**, not just advertised. See ``validate_arguments``."""
    handler: Callable[[OrchestratorClient, TInput], Awaitable[dict[str, object]]]
    writes: bool
    """True only for constraint-writing tools. Used for the authority audit test and for logging."""

    @property
    def input_schema(self) -> dict[str, object]:
        return dict(self.input_model.model_json_schema())

    def validate_arguments(self, arguments: dict[str, object]) -> TInput:
        """Validate a tool call against its declared schema, as a tool error on failure.

        Previously the ``*Input`` models existed only to *generate* JSON schema and handlers received the
        raw dict, so every declared bound was decorative: ``avoid_links`` accepted 500 links against a
        declared maximum of 32, and ``limit`` accepted -50 against a declared range of 1-200. A missing
        required field surfaced as a bare ``KeyError`` that escaped the tool-error path entirely.

        Bounds on a tool the model can call are an authority question, not a hygiene one — they have to run.
        """
        try:
            return self.input_model.model_validate(arguments)
        except ValidationError as error:
            raise ToolExecutionError(
                f"Invalid arguments for {self.name!r}: {error.error_count()} problem(s). Correct them and "
                f"call the tool again — do not answer as though it had succeeded.",
                {"tool": self.name, "errors": error.errors(include_url=False), "arguments": arguments},
            ) from error

    async def invoke(self, client: OrchestratorClient, arguments: dict[str, object]) -> dict[str, object]:
        """Validate then dispatch. The only path from a model's tool call into the orchestrator."""
        return await self.handler(client, self.validate_arguments(arguments))


def build_tool_specs() -> tuple[ToolInvoker, ...]:
    """The complete tool surface.

    Every write tool here produces a ``Constraint``. Adding one that does anything else is an ADR, not a
    code change — see ADR-0002 and `/add-mcp-tool`.
    """
    return (
        ToolSpec(
            name="get_network_state",
            description=read_tools.get_network_state.__doc__ or "",
            input_model=read_tools.GetNetworkStateInput,
            handler=lambda client, _: read_tools.get_network_state(client),
            writes=False,
        ),
        ToolSpec(
            name="get_recent_history",
            description=read_tools.get_recent_history.__doc__ or "",
            input_model=read_tools.GetRecentHistoryInput,
            handler=lambda client, args: read_tools.get_recent_history(client, limit=args.limit),
            writes=False,
        ),
        ToolSpec(
            name="get_recent_decisions",
            description=read_tools.get_recent_decisions.__doc__ or "",
            input_model=read_tools.GetRecentDecisionsInput,
            handler=lambda client, args: read_tools.get_recent_decisions(client, limit=args.limit),
            writes=False,
        ),
        ToolSpec(
            name="get_active_constraints",
            description=read_tools.get_active_constraints.__doc__ or "",
            input_model=read_tools.GetNetworkStateInput,
            handler=lambda client, _: read_tools.get_active_constraints(client),
            writes=False,
        ),
        ToolSpec(
            name="get_active_incidents",
            description=read_tools.get_active_incidents.__doc__ or "",
            input_model=read_tools.GetNetworkStateInput,
            handler=lambda client, _: read_tools.get_active_incidents(client),
            writes=False,
        ),
        ToolSpec(
            name="get_controller_status",
            description=read_tools.get_controller_status.__doc__ or "",
            input_model=read_tools.GetNetworkStateInput,
            handler=lambda client, _: read_tools.get_controller_status(client),
            writes=False,
        ),
        ToolSpec(
            name="get_flow_detail",
            description=read_tools.get_flow_detail.__doc__ or "",
            input_model=read_tools.GetFlowDetailInput,
            handler=lambda client, args: read_tools.get_flow_detail(client, flow_id=args.flow_id),
            writes=False,
        ),
        ToolSpec(
            name="prioritise_traffic_class",
            description=write_tools.prioritise_traffic_class.__doc__ or "",
            input_model=write_tools.PrioritiseTrafficClassInput,
            handler=lambda client, args: write_tools.prioritise_traffic_class(
                client,
                traffic_class=args.traffic_class,
                ttl_timesteps=args.ttl_timesteps,
                operator_utterance=args.operator_utterance,
            ),
            writes=True,
        ),
        ToolSpec(
            name="set_objective_weights",
            description=write_tools.set_objective_weights.__doc__ or "",
            input_model=write_tools.SetObjectiveWeightsInput,
            handler=lambda client, args: write_tools.set_objective_weights(
                client,
                throughput=args.throughput,
                latency=args.latency,
                drops=args.drops,
                fairness=args.fairness,
                ttl_timesteps=args.ttl_timesteps,
                operator_utterance=args.operator_utterance,
            ),
            writes=True,
        ),
        ToolSpec(
            name="avoid_links",
            description=write_tools.avoid_links.__doc__ or "",
            input_model=write_tools.AvoidLinksInput,
            handler=lambda client, args: write_tools.avoid_links(
                client,
                link_ids=args.link_ids,
                ttl_timesteps=args.ttl_timesteps,
                operator_utterance=args.operator_utterance,
            ),
            writes=True,
        ),
        ToolSpec(
            name="revoke_constraint",
            description=write_tools.revoke_constraint.__doc__ or "",
            input_model=write_tools.RevokeConstraintInput,
            handler=lambda client, args: write_tools.revoke_constraint(
                client, constraint_id=args.constraint_id
            ),
            writes=True,
        ),
    )


def build_server(client: OrchestratorClient) -> Server:
    """Wire the tool specs into an MCP server."""
    server: Server = Server("nexus-rl")
    specs = {spec.name: spec for spec in build_tool_specs()}

    # mcp's lowlevel decorators are themselves untyped, so mypy treats anything they wrap as untyped.
    # The wrapped functions below are fully annotated; the gap is in the library, not here.
    @server.list_tools()  # type: ignore[untyped-decorator,no-untyped-call]  # untyped mcp decorator
    async def list_tools() -> list[Tool]:
        return [
            Tool(name=spec.name, description=spec.description.strip(), inputSchema=spec.input_schema)
            for spec in specs.values()
        ]

    @server.call_tool()  # type: ignore[untyped-decorator]  # untyped mcp decorator, as above
    async def call_tool(name: str, arguments: dict[str, object]) -> list[TextContent]:
        spec = specs.get(name)
        if spec is None:
            raise ToolExecutionError(f"Unknown tool {name!r}.", {"available": sorted(specs)})

        try:
            result = await spec.invoke(client, arguments)
        except NexusError as error:
            # Surfaced as a tool error, never as an empty success. Given nothing, the model produces a
            # fluent invented answer indistinguishable from a real one.
            _telemetry.capture_exception(error, f"Tool {name} failed")
            raise

        if spec.writes:
            _telemetry.info(
                "Constraint applied via agent layer",
                {"tool": name, "arguments": arguments, "result": result},
            )
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    return server


async def run_stdio() -> None:
    settings = get_settings()
    client = OrchestratorClient(
        simulator_url=settings.simulator_url,
        controller_url=settings.controller_url,
        api_key=settings.api_key,
    )
    server = build_server(client)
    _telemetry.info(
        "Agent layer ready",
        {
            "simulator": settings.simulator_url,
            "controller": settings.controller_url,
            "tools": [spec.name for spec in build_tool_specs()],
        },
    )
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS-RL agent layer MCP server.")
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the tool surface and exit. Useful for auditing what the model can reach.",
    )
    args = parser.parse_args()

    if args.list_tools:
        for spec in build_tool_specs():
            kind = "WRITE (constraints only)" if spec.writes else "read"
            _telemetry.info("tool", {"name": spec.name, "kind": kind})
        return

    asyncio.run(run_stdio())
