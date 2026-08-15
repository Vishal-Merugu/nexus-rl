---
description: 'Run all quality gates after implementation. Usage: /post-implement-check'
---

## Post-Implementation Quality Check

Run all quality gates to verify the implementation is ready for commit.

### Step 1: Lint

Run `make lint`. Report remaining errors with `file:line` references.

### Step 2: Types

Run `make types` (mypy --strict, whole workspace). Do not scope it to changed files — cross-package type errors
are exactly what this catches.

### Step 3: Identify Changed Files

`git diff --name-only HEAD` (and `git diff --name-only` for unstaged). Note which packages they belong to.

### Step 4: Run Tests

For each changed package:

```bash
make test PKG=<package-name>
```

For dashboard changes: `pnpm --dir apps/dashboard test`.

### Step 5: Determinism Spot-Check

If anything under `packages/nexus-sim` or `packages/nexus-env` changed, run the determinism test explicitly:

```bash
uv run pytest packages/nexus-sim/tests/test_determinism.py -q
```

A determinism failure is CRITICAL regardless of what else passes.

### Step 6: MDP Version Check

If `packages/nexus-env` observation, action, or reward code changed, confirm `MDP_VERSION` was bumped. If not,
stop and spawn `mdp-reviewer`.

### Step 7: Summary

| Check       | Status    | Details                |
| ----------- | --------- | ---------------------- |
| Lint        | PASS/FAIL | N errors remaining     |
| Types       | PASS/FAIL | N type errors          |
| Tests       | PASS/FAIL | X/Y test files passed  |
| Determinism | PASS/FAIL/N-A |                    |
| MDP version | OK/MISSING/N-A |                   |

**If all pass:** "Ready for commit/PR."
**If any fail:** list failures with `file:line` and suggest fixes, then fix and re-run until clean.
