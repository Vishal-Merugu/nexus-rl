---
description: 'Diagnose and fix a failing test. Usage: /fix-test-failure <test-file or test-name>'
---

Fix the failing test: `$ARGUMENTS`

Spawn `test-writer-fixer`.

### Step 1: Reproduce

```bash
uv run pytest <target> -q -x
```

Capture the actual failure output. Never work from a description of a failure.

### Step 2: The verdict — do this before touching anything

Decide, explicitly, and say which:

- **The code is wrong** → fix the code. The test did its job.
- **The test is wrong** → the test asserted an implementation detail, or relied on a default that legitimately changed. Fix the test and note what made it fragile.
- **The contract changed intentionally** → this is the dangerous one. If the behaviour genuinely should change, that is a deliberate decision that needs saying out loud. For observation/action/reward it means an `MDP_VERSION` bump and an `mdp-reviewer` pass.

Silently relaxing an assertion to get green is the worst available outcome. If you find yourself widening a
tolerance, stop and justify the new tolerance in a comment.

### Step 3: Check the usual suspects first

| Symptom                                  | Check                                                             |
| ---------------------------------------- | ----------------------------------------------------------------- |
| Passes alone, fails in the suite         | Shared mutable state or a leaked global RNG                       |
| Fails intermittently                     | Determinism leak. Treat as CRITICAL — run the determinism tests   |
| Fails only on CI                         | Ordering, platform float differences, missing fixture             |
| Float assertion off by a tiny amount     | Accumulation. Widen with justification, or fix the accumulation    |
| Shape/dtype error deep in a stack trace  | Observation or action encoding — check `MDP_VERSION` moved         |
| Passed before an unrelated change        | The "unrelated" change was not unrelated. Bisect                   |

### Step 4: Fix minimally

Apply the smallest change to the existing code path. Do not restructure the surrounding code while fixing — it
hides whether the fix worked.

### Step 5: Verify

- [ ] The target test passes
- [ ] The whole package's tests pass — `make test PKG=<pkg>`
- [ ] The fix would have failed against the old code (confirm it addresses the cause, not the symptom)
- [ ] If the cause could recur elsewhere, add a test for that place too

### Step 6: Record

If this failure revealed a pattern worth remembering, add it to `tasks/lessons.md`. A test failure that taught
something and was not written down will be relearned.
