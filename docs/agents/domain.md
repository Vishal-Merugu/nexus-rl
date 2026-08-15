# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo uses a **single-context** layout: one `CONTEXT.md` and one `docs/adr/` directory at the repo root,
even though it is a polyglot workspace with several packages and services.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — the glossary. Inner tier vs. outer tier, constraint vs. allocation,
  flow vs. link, MDP version. These distinctions are load-bearing; using the wrong one produces confidently
  wrong analysis.
- **`docs/adr/`** — read the ADRs touching the area you are about to work in. In particular:
  - Simulator work → ADR-0001 (flow-level fidelity)
  - Agent-layer work → ADR-0002 (constrained authority), ADR-0003 (MCP), ADR-0005 (provider-agnostic LLM)
  - MDP or training work → ADR-0004 (discrete first)

If any of these files don't exist, **proceed silently**. Don't flag their absence or suggest creating them
upfront.

## File structure

```
/
├── CONTEXT.md
├── docs/
│   ├── adr/
│   ├── architecture/
│   ├── mdp/
│   └── evaluation/
└── apps/, packages/
```

## Use the glossary's vocabulary

When your output names a domain concept — an issue title, a refactor proposal, a hypothesis, a test name —
use the term as `CONTEXT.md` defines it. Don't drift to the synonyms it explicitly avoids:

- "the agent" for the RL controller (ambiguous with the LLM agent) → say **controller** or **inner tier**
- "policy change" for a constraint (the policy is not modified) → say **constraint**
- "priority" for a flow's class (priority is an outcome, class is an input) → say **traffic class**
- "campaign"-style legacy terms have no equivalent here, but the principle does: if the glossary marks a
  term as avoided, it is avoided everywhere including comments and telemetry

If the concept you need isn't in the glossary, that's a signal — either you're inventing language the project
doesn't use (reconsider), or there's a real gap worth recording.

## Flag ADR conflicts

If your output contradicts an ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0002 (constrained agent authority) — but worth reopening because…_

Two ADRs are effectively load-bearing for the project's identity and should be treated as very hard to
overturn: **ADR-0002** (the agent writes constraints, never allocations) and **ADR-0001** (flow-level
fidelity). A change to either is a change to what this project is.
