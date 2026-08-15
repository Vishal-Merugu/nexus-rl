---
description: 'Generate a handoff prompt before context runs out. Usage: /continue-session'
---

Generate a handoff so the next session resumes without re-deriving anything.

Write it to `tasks/handoff-<YYYY-MM-DD-HHMM>.md` **and** print it.

```markdown
# Handoff — <date>

## Goal

<what we are trying to achieve, one paragraph — the goal, not the history>

## State

- Branch: <name>, <n> commits ahead of <base>
- Working tree: clean / dirty (<files>)
- Build stage: <A–G from README>
- Quality gates: lint <status>, types <status>, tests <status>

## Done this session

- <change> → <file:line>

## In flight

- <what is half-finished, exactly where it stopped, and what the next concrete edit is>

## Active runs

| run-id | config | status | MDP_VERSION | note |
| ------ | ------ | ------ | ----------- | ---- |

## Decisions made

- <decision> — <why> — <ADR if written, or "needs ADR">

## Dead ends — do not retry

- <what was tried> — <why it failed>

## Next steps

1. <concrete first action for the next session>
2. ...

## Open questions for the user

- <anything blocking that needs a human answer>
```

### Rules

- **Dead ends are the highest-value section.** Without them the next session cheerfully repeats them.
- Give `file:line` references, not descriptions of where things are.
- The first next-step must be concrete enough to act on immediately — "add the fairness term to `reward.py:47`", not "continue reward work".
- If an MDP change is in flight, say so loudly, with the version state.
- If a training run is still going, record its run-id and where its artifacts land.
- Record anything learned that belongs in `tasks/lessons.md` — and write it there too, not only in the handoff.
