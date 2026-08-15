# Agent Layer

The outer tier: a language model with tools, sitting above the RL controller.

## Authority

**Reads anything. Writes constraints only.** There is no code path from the model to an allocation — not a
guarded one, not a validated one. `write_tools.py` contains no function that produces an `AllocationAction`,
and the controller exposes no endpoint that accepts one.

See [ADR-0002](../adr/0002-constrained-agent-authority.md).

## Tool surface

Enumerate it at any time:

```bash
uv run nexus-agent-server --list-tools
```

### Read tools

| Tool | Returns |
| ---- | ------- |
| `get_network_state` | Current links, flows, and QoS metrics |
| `get_recent_history` | Recent timesteps with metrics, oldest first |
| `get_recent_decisions` | What the controller did, and which constraints were in force |
| `get_active_constraints` | In-force constraints, with provenance and expiry |
| `get_active_incidents` | Degradations currently affecting the network |
| `get_controller_status` | Which allocator is live — trained policy or baseline |
| `get_flow_detail` | One flow, plus the links on its path |

### Write tools

Every one produces a `Constraint` with a mandatory, bounded TTL.

| Tool | Constraint produced |
| ---- | ------------------- |
| `prioritise_traffic_class` | Objective reweighting favouring a class |
| `set_objective_weights` | Direct reweighting of throughput / latency / drops / fairness |
| `avoid_links` | Routing restriction on named links |
| `revoke_constraint` | Lifts a constraint before its TTL elapses |

A test asserts this write set is exactly these four. A new write capability cannot appear unnoticed.

## Rules for every tool

1. **Structured data, never prose.** The model writes the sentences; a tool that returns a formatted sentence
   has pre-empted its reasoning and stripped out the numbers it should cite.
2. **A failed tool call raises.** Never `{}`, never a default, never empty success. Given nothing, the model
   produces a fluent, confident, invented answer indistinguishable from a real one. Every tool has a test
   for its failure path.
3. **Bounds at the schema.** Pydantic input models with every field typed and clamped. No
   `dict[str, object]` parameters — that is how unbounded authority arrives.
4. **Provenance recorded.** `issued_by` plus the operator's own words on every constraint, which is what
   makes explanations attributable rather than plausible.

## The LLM

Provider-agnostic: any OpenAI-compatible Chat Completions endpoint, configured by three environment
variables. Default is a local Ollama with no key. See
[ADR-0005](../adr/0005-provider-agnostic-llm.md).

```bash
NEXUS_LLM_BASE_URL=http://localhost:11434/v1
NEXUS_LLM_MODEL=qwen2.5:7b
NEXUS_LLM_API_KEY=
```

**The model must support tool calling.** That is the one hard requirement, and it is where small local models
struggle — below roughly 7B, expect calls to be skipped or arguments to be malformed. `llm.py` accepts both
the string and object forms of tool arguments that local servers disagree on, and raises a message naming
the likely cause when arguments are unparseable.

## The prompt

Two things it has to get right, because nothing downstream can fix them:

- **State the authority boundary explicitly**, so the model refuses out-of-scope requests coherently instead
  of improvising something adjacent and calling it compliance.
- **Forbid invented numbers, and make "I don't know" acceptable.** Every factual claim comes from a tool call
  in the conversation.

It also instructs the model to describe *what the controller did*, never *why the policy wanted it*. A
trained policy's reasoning is not observable from outside the network; asserting it would be the exact
overclaim this project has to avoid.

Tool results are data, never instructions — stated in the prompt so that text arriving through a tool cannot
redirect behaviour.

## Adding a tool

Use `/add-mcp-tool`. The first question is always the authority verdict: read (proceed), constraint-shaped
write (design the schema), or neither (write an ADR instead of a tool).
