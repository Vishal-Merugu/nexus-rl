---
description: 'Spawn 3 parallel reviewer agents to critique the current plan. Usage: /review-plan [path]'
---

Review the plan at `$ARGUMENTS` (or the plan just produced in this session).

Spawn **three reviewers in parallel**, each with a different mandate. Do not let them see each other's output —
independent critiques catch more than a consensus one.

### Reviewer 1 — Architecture (`nexus-system-architect`)

- Does each piece land in the right package or app? Any simulation math leaking into a service, any FastAPI leaking into a library?
- Is state ownership unambiguous? One writer per piece of state?
- Does anything require the outer tier to act at inner-tier speed, or grant it inner-tier authority?
- Is the degraded path designed, not just the happy path?
- Does this need an ADR that the plan does not propose?

### Reviewer 2 — Correctness & MDP (`mdp-reviewer` if MDP-touching, otherwise `python-code-reviewer`)

- Is this an MDP change the plan failed to identify? Check observation, action, reward, episode length.
- Is `MDP_VERSION` bumped, and are invalidated checkpoints listed?
- Does the plan introduce a determinism leak — any draw without an injected `Generator`?
- Are the invariants that this change could break named and asserted?
- Is there a degenerate optimum the new reward or action space would open up?

### Reviewer 3 — Testability & Evidence (`test-writer-fixer` + `evaluation-analyst` perspective)

- Are the tests specified before the implementation, and do they assert observable behaviour?
- Would any planned test train a model? (Blocker.)
- Are the adversarial cases covered — zero capacity, zero demand, incident at t=0, constraint expiring mid-step?
- If the plan makes a performance or quality claim, does it specify baseline, traffic pattern, and seed count?

## Output

Merge the three into one table, most severe first:

| Severity | Reviewer | Finding | Suggested fix |
| -------- | -------- | ------- | ------------- |

Severity: **CRITICAL** (plan produces wrong results or silently invalidates recorded numbers) / **HIGH** (violates
a `CLAUDE.md` rule, or a real bug is likely) / **MEDIUM** (maintainability or cost) / **LOW** (worth knowing).

Then:

- **Fix all CRITICAL and HIGH findings in the plan** before implementation starts.
- Re-run `/review-plan` if the fixes materially changed the design.
- If two reviewers disagree, surface the disagreement rather than picking one — it usually marks a real design question.
