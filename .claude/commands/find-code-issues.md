---
description: '3-agent adversarial bug hunt over a file, package, or the current diff. Usage: /find-code-issues [target]'
---

Adversarial bug hunt over: `$ARGUMENTS` (defaults to the current diff).

Spawn **three hunters in parallel**, each with a distinct attack angle. Independent passes find more than one
thorough pass.

### Hunter 1 — Determinism & state (`bug-hunter-analyst`)

- Global RNG anywhere in the path? `import random`, `np.random.<fn>`, unseeded generators
- Wall-clock time, dict/set iteration-order dependence
- `NetworkState` mutated after being handed out
- Shared mutable defaults, module-level state that survives a reset
- Timestep skew: can two surfaces read different timesteps and both look right?

### Hunter 2 — Numerical & invariant (`simulator-architect`)

- Division by zero: zero-capacity link, zero-demand flow, empty flow set
- Clamping: can utilization exceed 1, can a queue go negative, can allocations exceed capacity?
- Conservation: does allocated + queued + dropped account for offered load?
- Float accumulation over long episodes
- numpy shape mismatches that broadcast silently instead of raising
- Off-by-one on timestep and episode boundaries

### Hunter 3 — Boundary & authority (`mcp-tool-designer` + `python-code-reviewer`)

- Any path from the agent layer to a raw allocation
- Constraints without a TTL, or a TTL above `MAX_CONSTRAINT_TTL`
- Tools that return empty-success on failure instead of raising
- Missing `NexusError` mapping at a service boundary
- Bare `except:` swallowing a real failure
- Secrets read outside `nexus-config`

## Output

One merged table, most severe first:

| Severity | File:line | Defect | Failure scenario | Fix |
| -------- | --------- | ------ | ---------------- | --- |

Every finding needs a **concrete failure scenario**: specific inputs or state → wrong output. A finding without
one is a style opinion, and gets dropped.

Rank by severity:

- **CRITICAL** — wrong results, determinism break, invariant violation, authority escape
- **HIGH** — reachable bug, or a `CLAUDE.md` rule violated
- **MEDIUM** — real cost in a path that matters
- **LOW** — worth knowing

Then: write a failing regression test for the top finding before fixing it.
