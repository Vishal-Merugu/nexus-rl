# Agent Layer — the rApp

The Non-RT tier: a language model with tools, sitting above the RL controller. In O-RAN terms it is an
**rApp** hosted on the Non-RT RIC within SMO, and its entire write surface is the **A1 interface** to the
xApp below it.

## Authority

**Reads anything. Issues A1 policies only.** There is no code path from the model to an allocation — not a
guarded one, not a validated one. `write_tools.py` contains no function that produces an `AllocationAction`,
and the controller exposes no endpoint that accepts one.

It also contains no function that writes a `Constraint` directly. That is a distinct rule with its own
regression test: the controller's `POST /constraints` still exists for direct operator use, and calling it
from here would skip feasibility validation and return no decision — which is exactly what A1 was adopted to
provide.

See [ADR-0002](../adr/0002-constrained-agent-authority.md) for the authority rule and
[ADR-0008](../adr/0008-oran-aligned-a1-interface.md) for the interface it is expressed through.

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
| `get_recent_decisions` | What the xApp did, and which constraints were in force |
| `get_active_policies` | A1 policies in force — the *guidance* view — plus the policy types the xApp advertises |
| `get_active_constraints` | In-force constraints — the *enforcement* view, with provenance and expiry |
| `get_active_incidents` | Degradations currently affecting the network |
| `get_controller_status` | Which allocator is live — trained policy or baseline |
| `get_flow_detail` | One flow, plus the links on its path |

`get_active_policies` and `get_active_constraints` deliberately differ: a partially-accepted policy appears
in the first as issued and in the second only as the part the xApp agreed to enforce.

### Write tools

Every one issues an `A1Policy` with a mandatory, bounded TTL, and returns the xApp's decision.

| Tool | A1 policy type | Guidance issued |
| ---- | -------------- | --------------- |
| `prioritise_traffic_class` | `QOS_OBJECTIVE` | Objective reweighting favouring a class |
| `set_objective_weights` | `QOS_OBJECTIVE` | Direct reweighting of throughput / latency / drops / fairness |
| `avoid_links` | `TRAFFIC_STEERING` | Routing restriction on named links |
| `set_class_floor` | `CLASS_FLOOR` | Minimum aggregate rate for a traffic class |
| `withdraw_policy` | — | Withdraws a policy before its TTL elapses |

A test asserts this write set is exactly these five, and another asserts every issuing tool maps to a policy
type the xApp actually advertises. A new write capability cannot appear unnoticed, and one the xApp would
refuse on every call cannot ship looking functional.

## Reading a decision

The xApp answers every offer with an `A1PolicyDecision`, and the tool returns it verbatim. This is the part
that makes the tier useful rather than merely polite:

| `status` | `applied` | What the model must say |
| -------- | --------- | ----------------------- |
| `ACCEPTED` | `true` | What is in force, and the expiry timestep |
| `PARTIALLY_ACCEPTED` | `true` | **Not a success.** Both `enforced_elements` and `rejected_elements` |
| `REJECTED` | `false` | Nothing changed, the xApp's reason, and the `achievable_alternative` if present |

A refusal is a normal 200 response, not an error. Raising on it would make the model treat a considered "no"
as a broken call and retry it; returning a cheerful payload would let it report a guarantee that is not in
force. A response the rApp *cannot parse* is different again — that is a genuine failed call, and it raises.

## Rules for every tool

1. **Structured data, never prose.** The model writes the sentences; a tool that returns a formatted sentence
   has pre-empted its reasoning and stripped out the numbers it should cite.
2. **A failed tool call raises.** Never `{}`, never a default, never empty success. Given nothing, the model
   produces a fluent, confident, invented answer indistinguishable from a real one. Every tool has a test
   for its failure path. A refused policy is *not* a failed call — see above.
3. **Bounds at the schema.** Pydantic input models with every field typed and clamped. No
   `dict[str, object]` parameters — that is how unbounded authority arrives.
4. **Provenance recorded.** `issued_by` (`rapp:agent-layer`) plus the operator's own words on every policy,
   which is what makes explanations attributable rather than plausible.
5. **Feasibility is the xApp's call, not the rApp's.** The agent layer holds no model of the network, so it
   never pre-judges whether a floor or a steering request is achievable. It asks, and relays the answer.

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

Use `/add-mcp-tool`. The first question is always the authority verdict: read (proceed), A1-policy write
(design the schema, and check the xApp advertises that policy type), or neither (write an ADR instead of a
tool).
