---
description: 'Add tests until the behaviours of a file are covered. Usage: /improve-test-coverage <file>'
---

Improve test coverage for: `$ARGUMENTS`

Spawn `test-writer-fixer`.

### Step 1: Measure

```bash
uv run pytest --cov=<import_name> --cov-report=term-missing packages/<pkg>/tests/
```

Note uncovered **branches**, not lines. A 100%-line module with no boundary inputs is less tested than a
70%-line one with them.

### Step 2: List untested behaviours, not untested lines

For each uncovered branch, write the claim it represents:

> "when a link has zero capacity, allocation returns zero rather than dividing by zero"

If a branch has no meaningful claim behind it, it is probably dead code — say so instead of testing it.

### Step 3: Cover the adversarial cases first

Higher value than any percentage increase:

- Zero capacity, zero demand, empty flow set, single node
- Oversubscription — demand far above total capacity
- An incident at t=0, and an incident that outlasts the episode
- A constraint expiring mid-step
- Maximum values: every flow on one path, all classes identical
- Long episodes, for float accumulation

### Step 4: Write them

- Name the claim, not the function: `test_zero_capacity_link_allocates_nothing`
- Explicit seed, explicit fixture graph, no reliance on defaults
- Assert observable output only
- One reason to fail per test
- **Never** call `.learn()` or train anything

### Step 5: Verify the tests can fail

Break the implementation deliberately and confirm each new test fails. A test that passes against broken code is
worse than no test — it is a false assurance that will be trusted later.

### Step 6: Report

| Behaviour now covered | Test | Would catch |
| --------------------- | ---- | ----------- |

Plus: behaviours still untested and why (deliberate, or out of scope). Do not report a coverage percentage as
the headline — report which behaviours are guaranteed.
