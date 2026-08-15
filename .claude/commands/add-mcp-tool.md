---
description: 'Add a new MCP tool to the agent layer. Usage: /add-mcp-tool <tool-name> <what it should do>'
---

Add an MCP tool: `$ARGUMENTS`

Spawn `mcp-tool-designer`. Work through the authority question **before** writing any code.

### Step 1: Authority verdict — the gate

Classify the tool:

- **Read tool** → proceed freely. Exposing more state makes the agent's explanations better and carries no risk.
- **Write tool** → it may produce a `Constraint` and nothing else. Objective reweighting, action mask, per-class floor. Each with a bounded TTL.
- **Neither** → it wants to emit an allocation, mutate simulation state, or bypass the controller. **Stop.** That is ADR-0002 territory: write the ADR, do not write the tool.

Test the verdict adversarially: what could a maximally hostile operator utterance achieve through this tool? If
the answer includes destabilising the network, the design is wrong.

### Step 2: Schema

One file per tool in `apps/agent-layer/src/nexus_agent/tools/<tool_name>.py`:

- Explicit pydantic input model. Every field typed, bounded, documented.
- No `dict[str, object]` parameters — that is how unbounded authority arrives.
- Numerics clamped at the schema, not in the handler.
- Write tools: mandatory `ttl_timesteps` (≤ `MAX_CONSTRAINT_TTL`) and `issued_by`.
- Name it for the operator's intent (`prioritize_traffic_class`), not the mechanism (`set_objective_weight_vector`).
- Use `CONTEXT.md` vocabulary — tool names are user-facing surface.

### Step 3: Return structured data, never prose

The tool returns numbers and objects; the model writes the sentences. A tool returning a formatted sentence has
pre-empted the model's reasoning and stripped out the numbers it should have cited.

### Step 4: The failure path — mandatory

The tool raises a `NexusError` subclass on failure. It **never** returns `{}`, a default, or an empty-success
payload. Given nothing, the model produces fluent invented numbers indistinguishable from real ones.

Write this test first:

```
test_<tool>_raises_when_<dependency>_unavailable
```

### Step 5: Register and document

- Register in the MCP server's tool list
- Add a row to `docs/architecture/03_agent_layer.md`
- If it is a write tool, add it to the authority table in `SECURITY.md`

### Step 6: Verify end-to-end

- [ ] Schema rejects out-of-range input
- [ ] Failure path raises (test passes)
- [ ] Write tool: constraint expires on schedule, controller drops it without being told
- [ ] Provenance recorded — the operator utterance is traceable to the constraint
- [ ] Ask the live agent something requiring the tool; confirm it cites real retrieved numbers

Then `/post-implement-check`.
