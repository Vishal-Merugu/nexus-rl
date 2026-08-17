---
description: 'Add a new MCP tool to the agent layer. Usage: /add-mcp-tool <tool-name> <what it should do>'
---

Add an MCP tool: `$ARGUMENTS`

Spawn `mcp-tool-designer`. Work through the authority question **before** writing any code.

### Step 1: Authority verdict — the gate

Classify the tool:

- **Read tool** → proceed freely. Exposing more state makes the agent's explanations better and carries no risk.
- **Write tool** → it may issue an `A1Policy` and nothing else, of a type the xApp already advertises (`QOS_OBJECTIVE`, `TRAFFIC_STEERING`, `CLASS_FLOOR`). Bounded TTL, mandatory. Writing a `Constraint` directly bypasses A1 validation and the decision — that is not a write tool, it is ADR territory.
- **Neither** → it wants to emit an allocation, mutate simulation state, or bypass the xApp. **Stop.** That is ADR-0002 / ADR-0008 territory: write the ADR, do not write the tool.

A tool issuing an unadvertised policy type would be refused on every call — an authority surface that looks larger than it is. If the type does not exist yet, adding it to `nexus-a1` and the xApp's validator comes first.

Test the verdict adversarially: what could a maximally hostile operator utterance achieve through this tool? If
the answer includes destabilising the network, the design is wrong.

### Step 2: Schema

One file per tool in `apps/agent-layer/src/nexus_agent/tools/<tool_name>.py`:

- Explicit pydantic input model. Every field typed, bounded, documented.
- No `dict[str, object]` parameters — that is how unbounded authority arrives.
- Numerics clamped at the schema, not in the handler.
- Write tools: mandatory `ttl_timesteps` (≤ `MAX_CONSTRAINT_TTL`) and `issued_by`. Feasibility is the xApp's judgement, not the schema's — bound the input for sanity, never pre-judge what the network can carry.
- Name it for the operator's intent (`prioritize_traffic_class`), not the mechanism (`set_objective_weight_vector`).
- Use `CONTEXT.md` vocabulary — tool names are user-facing surface.

### Step 3: Return structured data, never prose

The tool returns numbers and objects; the model writes the sentences. A tool returning a formatted sentence has
pre-empted the model's reasoning and stripped out the numbers it should have cited.

### Step 4: The failure path — mandatory

The tool raises a `NexusError` subclass on failure. It **never** returns `{}`, a default, or an empty-success
payload. Given nothing, the model produces fluent invented numbers indistinguishable from real ones.

**A refused policy is not a failure.** `REJECTED` arrives as a 200 carrying the xApp's reason; return it, with
`applied: false`. Raising would have the model retry a considered "no". Only a transport error or a response
the rApp cannot parse raises.

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
- [ ] Write tool: no call to `/constraints` — the A1 path only
- [ ] Write tool: policy expires on schedule, xApp drops it without being told
- [ ] Write tool: all three decision statuses relayed correctly, partial acceptance not reported as success
- [ ] Provenance recorded — the operator utterance is traceable to the policy
- [ ] Ask the live agent something requiring the tool; confirm it cites real retrieved numbers

Then `/post-implement-check`.
