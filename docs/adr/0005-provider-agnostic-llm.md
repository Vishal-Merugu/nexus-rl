# 0005 — The agent layer talks to any OpenAI-compatible endpoint, defaulting to a local model

Date: 2026-08-15
Status: Accepted

## Context

The agent layer needs a tool-calling LLM. The obvious move is to pick a provider, install its SDK, and read
its API key from the environment. The first draft of this repo did exactly that with Anthropic.

That has three problems for this project specifically:

1. **Cost gates development.** This is a portfolio project built alongside study and a job search. Iterating
   on the agent layer means many dozens of tool-calling round trips per debugging session. A metered API
   turns "let me try that prompt again" into a decision.
2. **It makes a demo depend on a credential.** Anyone cloning the repo — a reviewer, an interviewer,
   a future employer — hits a wall unless they have a key for whichever vendor was chosen.
3. **It bakes a reversible decision into code.** Section 7, Decision 4 of the project spec chose a hosted
   API for v1 and listed self-hosting an open-weights model as future scope. Coupling to one vendor's SDK
   makes that future scope a refactor rather than a config change, which is precisely how "future scope"
   quietly becomes "never".

The interesting engineering in this layer is the tool design and the authority boundary (ADR-0002), not
which company serves the tokens.

## Decision

The agent layer speaks the **OpenAI-compatible Chat Completions API** and nothing else. Provider selection
is three environment variables:

```
NEXUS_LLM_BASE_URL   # endpoint root, including /v1
NEXUS_LLM_MODEL      # model id as that endpoint names it
NEXUS_LLM_API_KEY    # empty for local servers
```

The default is a **local Ollama at `http://localhost:11434/v1` with no key**, so a fresh clone runs the
complete system with no account anywhere.

No vendor SDK is a dependency. `apps/agent-layer/src/nexus_agent/llm.py` is a thin `httpx` client over one
POST with a `tools` array — the entire surface this project uses.

This works because the OpenAI Chat Completions shape has become the de facto interface: Ollama, llama.cpp,
LM Studio, vLLM, OpenRouter, Groq, Together, DeepInfra, OpenAI and Anthropic (via its compatibility
endpoint) all serve it.

## Consequences

- **The project runs at zero marginal cost.** Local inference means unlimited iteration on prompts and tool
  design, which is where the actual work is.
- **A reviewer can run the whole thing.** `ollama pull qwen2.5:7b` and the demo works — no signup.
- **Provider choice becomes a runtime decision.** Local model while building, a fast hosted one for a
  recorded demo, whatever is cheapest later. Two variables, no code change, no redeploy of anything but env.
- **Self-hosting stops being future scope** and becomes the default, which strengthens the data-locality
  story rather than deferring it.
- **Tool-calling quality now varies with the model, and that is a real cost.** A 7B local model calls tools
  less reliably than a frontier model: it will occasionally answer without calling a tool, or emit malformed
  arguments. Mitigations: the system prompt is explicit about citing retrieved numbers; `llm.py` accepts
  both the string and object forms of tool arguments that local servers disagree on; malformed arguments
  raise an error naming the likely cause. Models below ~7B are documented as unreliable for this.
- **We inherit the OpenAI schema's limits.** No access to provider-specific features — prompt caching,
  extended thinking, vendor-native tool formats. For this workload (short operator questions, a handful of
  tool calls, structured answers) none of those are load-bearing.
- **Endpoints diverge in small ways.** Ollama and llama.cpp are close to the spec but not identical, so the
  response parser is deliberately tolerant. Each divergence found is covered by a test in `test_llm.py`.

## Alternatives considered

- **Anthropic SDK directly** (the original draft). Rejected: best-in-class tool calling, but it gates
  development on spend and gates the demo on a credential, for a component whose interest is the tool design
  rather than the model.
- **LiteLLM or a similar routing layer.** Rejected: it solves the same problem with a dependency and a
  configuration format to learn. The subset used here is one endpoint shape; a 200-line client covers it and
  keeps the failure modes visible, which matters more when the endpoint is a local process that might not be
  running.
- **Support both a native SDK and an OpenAI-compatible path.** Rejected: two code paths, one of which is
  always the less-tested one. If a provider-specific feature ever earns its place, that is a new ADR.
- **Ollama's native `/api/chat` rather than its `/v1` compatibility endpoint.** Rejected: it would couple
  the code to Ollama specifically, which is the mistake this ADR exists to avoid — just with a different
  vendor.
