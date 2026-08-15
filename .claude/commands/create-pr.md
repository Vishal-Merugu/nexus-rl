---
description: 'Prepare and create a pull request for the current branch. Usage: /create-pr'
---

## Create Pull Request

### Step 1: Gate — do not skip

Spawn `pre-commit-auditor`. **Proceed only on READY.** If BLOCKED, fix the violations and re-run the auditor.

### Step 2: Branch check

- On the default branch? Create a branch first: `features/XX`, `fixes/XX`, or `experiments/XX`.
- Rebase on the base branch and resolve conflicts.

### Step 3: Commits

Conventional Commits. This repo's types:

| Type        | For                                          |
| ----------- | -------------------------------------------- |
| `feat:`     | New capability                               |
| `fix:`      | Bug fix                                      |
| `sim:`      | Simulator model or fidelity change           |
| `rl:`       | MDP, training, or evaluation change          |
| `agent:`    | Agent layer / MCP tools                      |
| `perf:`     | Performance                                  |
| `refactor:` `chore:` `test:` `docs:` | Maintenance         |

### Step 4: PR description

```markdown
## What

<one paragraph>

## Why

<the problem, not the solution>

## MDP impact

- MDP change: yes/no
- `MDP_VERSION`: <old> → <new>
- Invalidated checkpoints: <run-ids, or none>

## Agent authority impact

- New agent capability: yes/no
- If yes: why it stays inside constraints (ADR-0002)

## Verification

- [ ] `make lint`
- [ ] `make types`
- [ ] `make test`
- [ ] Determinism test passes (if sim/env touched)
- [ ] Benchmarked `step()` cost (if hot path touched): <before> → <after>

## Results

<if this PR makes a performance claim: run-id, baseline, traffic pattern, seed count. Otherwise: N/A>
```

### Step 5: Push and open

```bash
git push -u origin <branch>
gh pr create --title "<conventional commit title>" --body "<description>"
```

### Step 6: Report

Give the PR URL and a one-line summary of what a reviewer should look at first.

**Never** commit files under `experiments/runs/`, `.env`, or `*.zip` checkpoints. The auditor checks this; check
it yourself too.
