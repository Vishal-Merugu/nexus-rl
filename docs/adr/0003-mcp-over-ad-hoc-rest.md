# 0003 — Expose the orchestrator to the LLM via MCP, not hand-written REST tool-calling

Date: 2026-08-14
Status: Accepted

## Context

The agent layer needs to read orchestrator state and write constraints. Two ways to give a model that
ability:

1. **Hand-written function calling over REST.** Define tool schemas inline, call the controller's HTTP API
   from the handlers, wire them into whatever the provider's tool-calling format is.
2. **An MCP server** exposing the same capabilities as protocol-level tools.

Option 1 is less code today. Option 2 costs a little more up front.

## Decision

**Build the MCP server.**

`apps/agent-layer/src/nexus_agent/server.py` exposes seven read tools and four constraint-writing tools. Tool
specs are declared as data (`ToolSpec`), so the same registry feeds both the MCP surface and a direct
chat-completions loop — the transport is not baked into the tool definitions.

## Consequences

- **The tool surface is enumerable and auditable.** `nexus-agent-server --list-tools` prints exactly what
  the model can reach, marked read or write. A test asserts the write set is exactly the four expected tools,
  so a new capability cannot appear without someone noticing — which is what makes ADR-0002's authority claim
  checkable rather than aspirational.
- **The agent layer is usable from any MCP client**, including Claude Desktop and Claude Code, without
  building a chat UI first. During development that is the fastest way to try a tool by hand.
- **Schemas are pydantic models, so bounds are declared once** and enforced at the boundary rather than
  inside handlers. A TTL cap in a schema cannot be forgotten by a handler.
- **It costs an extra dependency and an extra process boundary.** For a four-service system that is a real,
  if small, addition to what has to be running for a demo.
- **Transport-neutral specs mean two consumers to keep working.** The MCP path and the direct
  chat-completions path (used by the dashboard chat, per ADR-0005) share the registry, so a badly-shaped tool
  breaks both. That shared surface is the point, but it is worth stating.
- **It connects this project to the original MCP platform idea** the specification set out to reuse — the
  agent layer is that idea, given a concrete and finishable home.

## Alternatives considered

- **Hand-written REST tool-calling.** Rejected: less code now, but the tool surface ends up scattered across
  handler functions and provider-specific schema dicts, which makes the authority audit harder exactly where
  it matters most. It also ties the tool definitions to one provider's format, which ADR-0005 rules out.
- **Expose the controller's OpenAPI spec directly to the model.** Rejected: it would expose `/act` and the
  raw shape of `AllocationAction` — the model would see an allocation-shaped hole in the API and try to use
  it. The tool surface should be smaller than the HTTP surface, deliberately.
- **MCP for reads, direct REST for writes.** Rejected: writes are precisely where the schema bounds, TTL
  enforcement and audit trail matter. Splitting transports would put the important half on the less
  structured path.
