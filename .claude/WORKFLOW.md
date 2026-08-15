# Claude Code Workflow Reference

## How It Works

Your Claude setup has three layers of automation:

1. **Hooks** — shell scripts that run automatically on events. You never invoke these.
2. **Hookify Rules** — pattern-matching rules that block or warn when Claude tries forbidden patterns. You never invoke these.
3. **Skills & Agents** — commands you CAN invoke manually, but Claude is also configured to invoke them automatically at the right time.

**You shouldn't need to remember any skill names.** `CLAUDE.md` contains workflow automation rules that trigger
the right skill at the right moment. Just describe what you want to build.

---

## The Automated Lifecycle

```
You: "Build feature X"
  |
  v
/create-plan          <-- Claude enters plan mode, designs with agent teams
  |
  v
/review-plan          <-- 3 parallel reviewer agents critique the plan
  |
  v
Iterate on plan       <-- Fix CRITICAL/HIGH findings
  |
  v
Implement (TDD)       <-- Tests first, then code
  |
  v
/post-implement-check <-- AUTO: ruff + mypy + pytest + vitest
  |
  v
pre-commit-auditor    <-- AUTO: final quality gate (no Any, no global RNG, MDP version)
  |
  v
You: "Create PR"
  |
  v
/create-pr            <-- Rebase, prepare description, push
```

### What triggers automatically

| Trigger                              | What Runs                  | Why                                                       |
| ------------------------------------ | -------------------------- | --------------------------------------------------------- |
| After any implementation             | `/post-implement-check`    | Catches lint/type/test failures before you see them       |
| Observation / action / reward change | `mdp-reviewer` agent       | Verifies `MDP_VERSION` bump, flags dead checkpoints       |
| Reward curve pasted, training stuck  | `/diagnose-reward`         | Structured triage instead of guessing at hyperparameters  |
| Before commit/PR                     | `pre-commit-auditor` agent | No `Any`, no global RNG, no committed run artifacts       |
| Context running low                  | `/continue-session`        | Generates handoff prompt for seamless continuation        |

---

## Skills You CAN Invoke Manually

### Everyday

| Command                      | When to Use                                       |
| ---------------------------- | ------------------------------------------------- |
| `/create-plan <description>` | Start any non-trivial feature or investigation    |
| `/review-plan [path]`        | Spawn 3 reviewer agents to critique current plan  |
| `/post-implement-check`      | Force a quality check at any point                |
| `/continue-session`          | Generate handoff prompt before context dies       |

### RL / Simulation

| Command                          | When to Use                                              |
| -------------------------------- | -------------------------------------------------------- |
| `/diagnose-reward [run-id]`      | Training plateaued, collapsed, or is suspiciously flat   |
| `/train-and-evaluate <config>`   | Launch a run and produce the baseline comparison         |
| `/add-traffic-pattern <name>`    | Add a seeded demand generator with its evaluation wiring |

### Agent Layer

| Command                 | When to Use                                             |
| ----------------------- | ------------------------------------------------------- |
| `/add-mcp-tool <name>`  | Expose new state or a new constrained write to the LLM  |

### Testing

| Command                         | When to Use                     |
| ------------------------------- | ------------------------------- |
| `/improve-test-coverage <file>` | Add tests until branches covered |
| `/fix-test-failure <file>`      | Diagnose and fix a failing test |
| `/find-code-issues`             | 3-agent adversarial bug hunt    |

### PR Workflow

| Command      | When to Use                           |
| ------------ | ------------------------------------- |
| `/create-pr` | Push and create PR for current branch |

---

## Hookify Rules (Automatic — Block/Warn on Bad Patterns)

Pattern-matching guards that intercept edits and commands BEFORE they execute.

| Rule                       | Event                    | Action    | What It Catches                                    |
| -------------------------- | ------------------------ | --------- | -------------------------------------------------- |
| `block-any-type`           | File edit (`.py`)        | **BLOCK** | `: Any`, `-> Any`, `List[Any]`, `cast(`            |
| `block-global-rng`         | File edit (`.py`)        | **BLOCK** | `import random`, `np.random.<fn>`, `np.random.seed` |
| `block-bare-except`        | File edit (`.py`)        | **BLOCK** | `except:` and `except Exception:` without re-raise |
| `block-training-in-tests`  | File edit (`test_*.py`)  | **BLOCK** | `.learn(`, `train(` inside a test file             |
| `block-commit-run-artifacts` | Bash command           | **BLOCK** | `git add` touching `experiments/runs/`             |
| `warn-star-import`         | File edit (`.py`)        | **WARN**  | `from x import *`, re-export barrels               |

### Where rules live

- Files: `.claude/hookify.{name}.local.md` in project root
- Format: YAML frontmatter (name, event, action, conditions) + markdown message body

---

## Agents (Specialized Sub-Processes)

Claude spawns these automatically when relevant.

| Agent                       | Specialty                              | When Used                                  |
| --------------------------- | -------------------------------------- | ------------------------------------------ |
| `nexus-system-architect`    | Two-tier system design, service boundaries | Architecture planning                  |
| `simulator-architect`       | Flow-level simulation math, invariants | Simulator changes, fidelity questions      |
| `rl-training-debugger`      | Reward shaping, instability, plateaus  | Training problems (auto on stuck runs)     |
| `mdp-reviewer`              | Observation/action/reward review       | Any MDP change (auto-triggered)            |
| `mcp-tool-designer`         | Tool schemas, agent authority limits   | Agent-layer work                           |
| `evaluation-analyst`        | Baselines, seeds, statistical honesty  | Results, writeups, comparison tables       |
| `python-code-reviewer`      | Typed Python review                    | Code quality analysis                      |
| `pre-commit-auditor`        | Quality gate checklist                 | Before commits (auto-triggered)            |
| `realtime-frontend-expert`  | React + WebSocket dashboard            | Dashboard work                             |
| `devops-automation-expert`  | Docker, K8s, GitHub Actions            | Infrastructure tasks                       |
| `bug-hunter-analyst`        | Root cause analysis                    | Bug reports                                |

---

## Knowledge System (Two Layers)

### 1. Shared Lessons (versioned, team-wide)

**File:** `tasks/lessons.md` — committed to git.

Patterns learned from corrections: determinism rules, type safety, MDP discipline, experiment hygiene.

### 2. Local Memory (personal, per-machine)

**Location:** `~/.claude/projects/.../memory/` — NOT in git.

Personal preferences: working style, individual workflow quirks.

### What goes where?

| Type of Knowledge       | Where              | Example                                     |
| ----------------------- | ------------------ | ------------------------------------------- |
| Project rules           | `tasks/lessons.md` | "Never touch a global RNG"                  |
| Architecture decisions  | `docs/adr/`        | "Agent writes constraints, not allocations" |
| Domain vocabulary       | `CONTEXT.md`       | "Inner tier vs. outer tier"                 |
| Personal preferences    | Local memory       | "User prefers terse responses"              |

---

## Quick Reference

**Starting a feature:** Just describe it. Claude plans, implements with TDD, runs quality gates.

**Training is stuck:** Paste the reward curve or the run-id. Claude runs `/diagnose-reward` automatically.

**Changed the reward function:** Claude auto-spawns `mdp-reviewer` and will tell you which checkpoints just died.

**Context running out:** Claude auto-generates a handoff prompt. Or run `/continue-session`.
