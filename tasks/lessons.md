# Lessons Learned

Patterns Claude must follow, learned from corrections during development. This file is **versioned in git**.
Claude updates it after user corrections.

**Format:** Each lesson has a rule, why it exists, and when to apply it.

---

## Determinism

### Never touch a global RNG

Every stochastic draw takes an explicitly passed `numpy.random.Generator`. No `import random`, no
`np.random.uniform(...)`, no module-level seeding.

**Why:** A simulator whose trajectory depends on hidden global state cannot be debugged. "Same seed, same
config, different reward" burns hours and looks like an RL problem when it is a plumbing problem.

**Apply:** Every file under `packages/nexus-sim` and `packages/nexus-mdp`. Enforced by hookify rule
`block-global-rng`.

### Seed and record every run

A run's `config.yaml` records the seed, the git SHA, and `MDP_VERSION`. A result without those is not
reproducible and therefore not a result.

**Apply:** Anything writing to `experiments/runs/`.

---

## Type Safety

### Never use `Any`

Use protocols, generics, `object` with narrowing, or a real type. `# type: ignore` requires an inline reason
comment.

**Apply:** Every Python file. Enforced by hookify rule `block-any-type`.

### Never use `cast()` to silence mypy

If mypy disagrees, one of you is wrong and it is usually not mypy. Narrow properly.

**Why:** A cast that hides a wrong type surfaces as a shape error 40 minutes into a training run.

### Numpy arrays carry their dtype in the annotation

`NDArray[np.float32]`, not `np.ndarray`. Observation and action arrays are `float32` — SB3 assumes it.

**Why:** A silent `float64` observation costs memory and throws dtype errors deep inside SB3's buffers.

---

## Simulation

### `step()` returns new state, never mutates

`NetworkState` is frozen. Mutating it in place invalidates snapshots already handed to the dashboard and the
agent layer, producing "the explanation doesn't match what I saw" bugs that are near-impossible to trace.

### Assert physical invariants, don't assume them

Utilization ∈ [0, 1]. Queue ≥ 0. Sum of allocations ≤ capacity. Assert at the boundary of every simulation
function.

**Why:** A negative queue propagates into the reward as a bonus, and the policy learns to cause it. RL will
find every invariant you did not enforce.

### Keep `step()` under ~1 ms

Training does millions of steps. A 10× slower step turns a 20-minute run into an overnight one and kills
iteration speed. Vectorize with numpy; profile before adding fidelity.

---

## MDP

### Reward components are logged individually

Never log only the scalar. `info["reward_components"]` carries every named term.

**Why:** A flat reward curve is uninterpretable. A flat *total* with a rising throughput term and a falling
fairness term tells you exactly what happened.

### Changing observation/action/reward means bumping `MDP_VERSION`

And saying so in the PR, and flagging which checkpoints it invalidates.

**Why:** Silent MDP drift makes every previously-recorded number wrong, quietly.

### Test the env contract, never train in a test

Assert shapes, bounds, determinism under a fixed seed, reward sign under obviously-good and obviously-bad
actions. A test that calls `model.learn()` is a slow, flaky test that proves nothing.

**Apply:** Enforced by hookify rule `block-training-in-tests`.

---

## Agent Layer

### The agent writes constraints, never allocations

There is no code path from the LLM to an allocation number. If a feature seems to need one, it needs an ADR
instead.

**Why:** Both a safety property and the project's central design claim. See ADR-0002.

### Every constraint has a TTL

Bounded by `MAX_CONSTRAINT_TTL`. The controller expires them itself.

**Why:** An operator override that never expires becomes undocumented permanent policy nobody remembers
setting.

### A failed tool call is a failed tool call

Never return an empty-success payload from an MCP tool. The model will narrate around it and invent numbers
that look exactly like real ones.

**Apply:** Every tool in `apps/agent-layer/src/nexus_agent/tools/`.

---

## Process

### Always verify before declaring done

`make lint`, `make types`, and the relevant tests. Every time.

### Claims about performance need a baseline, a pattern, and ≥3 seeds

"The policy improved latency" is not a claim. "PPO run `a1b2c3` cut p95 latency 23% vs. the `static` baseline
on `diurnal`, mean of 5 seeds" is.

**Apply:** See `.claude/skills/rl-experiment-hygiene/SKILL.md`.

### Never commit run artifacts

`experiments/runs/` is gitignored except `.gitkeep`. Checkpoints are large, binary, and invalidated by the
next MDP change.

---

## Communication

### Lead with the bottom line. No novels.

Open with what is actually changing, in one sentence. Then the choice. Stop.

### Say "simulated" out loud

In READMEs, writeups, and interview talking points, state the simulation boundary plainly. Overclaiming
real-network validation would undermine an otherwise strong project — and the honest framing is the more
impressive one.

---

## Service State (added after the first full audit)

### Read paths must never mutate

`GET /health` and `GET /constraints` both called a `prune()` that deleted state. A Kubernetes liveness
probe therefore wiped every operator constraint every 15 seconds, and a parameterless constraint read
destroyed anything issued after timestep 0.

**Why:** garbage collection needs *real, advancing* time. A reader asking about an earlier moment is not
evidence that later state is dead. Collect only on the control path, where the timestep is the simulation's
own clock.

**Apply:** any endpoint whose verb is GET. If it changes anything, it is not a read.

### Filter before the wire, not with a method

`ConstraintSet.active()` filters, but the serialised payload carried the unfiltered tuple — so clients
received expired constraints and the agent layer would report lapsed rules as in force.

**Why:** a method is not part of the JSON. Whatever the response model holds is what the client believes.

### A background loop may never die

The control loop caught `(httpx.HTTPError, ValueError)`, but `ActionDecodingError` is a `NexusError`. One
malformed action killed the task permanently; the service kept answering requests and `/health` still said
`ok` while the simulation sat frozen at timestep 0.

**Why:** this is the exact "healthy while doing nothing" failure the loop exists to prevent. Wrap the loop
body, catch the project's own error base, and expose liveness (`control_loop_alive`) on health.

**Apply:** every `asyncio.create_task` that is supposed to run forever.

### `reset()` without a seed must advance the run, not repeat it

The env re-seeded with the master seed on every bare reset, so every training episode was byte-identical.
A policy would train on one trajectory forever, look converged, and generalise to nothing.

**Why:** SB3 calls bare `reset()` between episodes. Episode `n` now draws from `seed + n` — varied, and the
whole sequence still replays exactly.

**Apply:** any gymnasium env. Assert that consecutive episodes differ *and* that the sequence is reproducible.

### Refuse impossible configuration at construction

`flow_count=100` on a 30-pair topology silently built 30 flows, so the observation came out shorter than
the declared space — surfacing as a shape error deep inside the training loop, far from the cause.

**Why:** silent truncation moves the error far from its origin. Fail where the mistake was made.

### A guard that returns early guards nothing

The generalisation check returned early whenever the training pattern was absent from
`evaluation_patterns`, which let an *empty* evaluation set pass. Check the invariant unconditionally.

### Constraints only shape what the controller may do — never what it does

The fallback allocator replaced the static baseline with a full-open action whenever any constraint
existed, removing all throttling and relabelling the result as `controller` rather than `baseline:static`.
A routing constraint must add routing and nothing else, and provenance must never be overwritten.

### Test service *sequences*, not just components

269 unit tests passed while every one of the above was live. They covered components in isolation and never
exercised a sequence of requests against a running service — which is where read-path mutation, bad
defaults, and dying tasks all hide.

**Apply:** for each service, test: read-after-write, read-at-a-different-time, probe-then-read,
restart/reset, and the background task under every way its dependency can fail.

---

## Boundaries (added after the second, deeper audit)

### A declared bound that never runs is decoration

The MCP `*Input` models were used only to *generate* JSON schema; handlers received the raw dict. So
`avoid_links` accepted 500 links against a declared maximum of 32, and `limit` accepted -50 against a
declared range of 1-200.

**Why:** for a tool a language model can call, bounds are an authority question, not hygiene. Validate the
arguments against the model before dispatch, and turn the failure into a tool error.

**Apply:** every `ToolSpec`. `validate_arguments` runs on every call.

### A contract field nobody reads is a lie

`class_floor_mbps` was accepted by `Constraint`, blessed by its "constrains something" validator, and
advertised in ADR-0002, SECURITY.md and CONTEXT.md — while no code read it. A floor-only constraint was
accepted, confirmed to the operator, expired on schedule, and did nothing.

**Why:** the empty-success failure at the constraint level, which is the exact thing ADR-0002 exists to
prevent. Either implement the field or delete it; never let the type promise what the system does not do.

**Apply:** when adding a field to a boundary contract, add its consumer in the same change.

### `json.dumps` emits invalid JSON by default

NaN and inf become bare `NaN`/`Infinity` literals, which `jq` and JS `JSON.parse` reject. A run that went
NaN produced a metrics file nothing could read — exactly when `/diagnose-reward` needs to read it.

**Why:** `allow_nan=True` is the default and it is wrong for anything written to disk.

**Apply:** `allow_nan=False`, map non-finite to `null`, flag the record.

### Never decide identity by substring

`is_local_llm` matched `"localhost" in url`, so `evil-localhost.example.com` and `127.0.0.1.attacker.net`
both passed as local and bypassed the remote-credential guard.

**Apply:** parse and compare the hostname. Same for any host, origin, or scheme check.

### Report a condition only if it is real

An incident targeting a link the topology does not have appeared in `active_incident_ids` while degrading
nothing — the dashboard showed "Incident active" and the agent would have told an operator the network was
degraded when it was not.

**Apply:** derive status from effect, not from intent.

### Guard the degenerate config at construction

`episode_length=0`, `flow_count=-3`, `tick_seconds=0` were all accepted and failed later, somewhere else.
A zero tick spun the control loop at 518 timesteps in 0.15s.

### An error message that names the wrong cause costs more than no message

A non-finite action tripped the allocation validator with "shape mismatch — check MDP_VERSION". The real
cause is a diverged policy. Point at the actual cause, or the reader spends an hour in the wrong file.

---

## Architecture (added after the structure review)

### Two services must never derive shared state independently from config

The controller rebuilt the simulator's flow set from its own `NEXUS_SEED`/`NEXUS_TOPOLOGY` to learn the
action space. Flow ids are positional (`f0`..`fn`), so a one-digit config difference produced ids that
lined up perfectly while the routes behind them belonged to entirely different flows — wrong paths, no
error anywhere.

**Why:** config-derived agreement is agreement by coincidence. Put the shared thing in the message.

**Apply:** if service B needs to know something about service A's world, A tells it. `FlowState` carries
`candidate_paths` for exactly this reason.

### Split the contract from the framework that consumes it

`nexus-env` held both the MDP codec and the gymnasium `Env`, so the controller — which needs only the codec
— pulled gymnasium into a service that never constructs an `Env`. Splitting out `nexus-mdp` made "the MDP
lives in one place" a property of the package graph rather than a convention.

**Apply:** when a library has two consumer groups with different dependency needs, that is usually two
libraries.

### Resolve artifact paths against the project root, never the CWD

`RUNS_ROOT = Path("experiments/runs")` scattered runs across whatever directory the command was invoked
from. A run nobody can find is a run that did not happen.

### Tooling notes

- Hatchling already ships every file under `packages = ["src/<pkg>"]`, including `py.typed`. Adding
  `force-include` grafts a stray top-level `src/` into the wheel.
- `ruff format` reflows `# noqa` comments across lines, which silently breaks string-replacement edits
  targeting them. Re-read after formatting.
