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
- **Never bare `uv sync`.** It resolves the root project only and *uninstalls* the rest of the workspace —
  `stable-baselines3` disappeared, which made `mypy` report an unrelated `unused-ignore` in `train.py` and
  looked exactly like a real regression in code nothing had touched. `make install`
  (`uv sync --all-packages --all-extras`) is the only correct sync.
- When a gate fails in a file your change never touched, check the environment before the code. Stash and
  re-run against a clean HEAD: if it fails there too, it is not your diff.

---

## Gates that have never run (added after the first real toolchain install)

### A gate that has never executed is not a gate

`uv`, `pnpm` and `docker` were absent while the repo was written, so `make` had never run once. The tree
looked finished — 362 passing tests, lint clean — and hid: a uv workspace that could not resolve at all
(root project and `packages/nexus-rl` shared a name), 82 `mypy --strict` errors, an MCP server that
raised on import against its own resolved dependency, and a `lint` script pointing at an eslint config
that did not exist.

**Why:** "written but never executed" is indistinguishable from "broken" until something runs it. The
green tests were real but covered only the paths that did not need the missing tools.

**Apply:** install the toolchain first, run every documented command once, and treat the first run of any
gate as a source of findings rather than a formality.

### Pin the major of a dependency whose API you are built on

`mcp>=1.2` resolved to 2.0.0, which removed the lowlevel `@server.list_tools()` decorators the agent
layer is built on. `build_server()` raised `AttributeError` on import; the service could not start.

**Why:** an unbounded lower bound is a promise that every future major stays compatible. It also went
unnoticed because no test constructed the server — 362 tests passed while the entry point was dead.

**Apply:** if a module is written against a library's specific API shape, bound the major. And test the
*assembly* of every service, not only its parts.

### mypy's `disallow_any_explicit` is unusable with pydantic

pydantic's own `BaseModel` signature contains `Any`, and mypy attributes it to the **subclass definition
line**. Every contract model in `nexus-types` therefore failed a rule none of its source violated — 28 of
the 82 errors.

**Why:** the setting reads library code, not just ours. The project rule ("never use `Any`") belongs
where it can see only our source: ruff `ANN401` plus the `block-any-type` hookify rule.

### An annotation that contradicts the endpoint makes the guard dead code

`OrchestratorClient._request` was typed `dict[str, object]`, but `/history`, `/incidents` and
`/decisions` return JSON arrays. The callers' own `isinstance(payload, list)` checks were statically
unreachable — mypy had been reporting a real contradiction, and the code only worked because nothing
type-checked it.

**Apply:** when a narrowing guard reads as unreachable, suspect the annotation before the guard.

### Validate into a typed model, then keep the type

`validate_arguments` produced a validated pydantic model and immediately `model_dump()`ed it back to a
raw dict, so every handler re-coerced fields out of `object` — ten `type: ignore`s to compile something
validation had already established. Making `ToolSpec` generic in its input model, behind a non-generic
protocol so the registry stays homogeneous, removed all ten.

**Why:** discarding types at the exact moment you have earned them is how `Any` re-enters a codebase that
bans it.

---

## Results (added after the first end-to-end training run)

### A tool that cannot produce the project's central claim is the bug

`nexus-eval --run-id <id>` validated the checkpoint's MDP version, logged the run's metadata, and then
compared the three baselines against one another. The policy was never rolled out. `rollout_policy`
existed, was tested, and was called by nothing.

**Why:** the whole point of the evaluator is "does the learned controller beat the floor". Every
individual piece worked; the wiring between them did not exist, and no test asserted the end-to-end
claim.

**Apply:** for each headline claim the project makes, there should be one test that fails if the claim
can no longer be produced.

### Evaluate on the run's own configuration, never the tool's defaults

The evaluator scored policies on its default `mesh-6` / `diurnal` + `flash_crowd` regardless of what the
run recorded. A policy trained on a different topology or flow count would have been compared against a
different problem and the number reported as if it meant something.

**Apply:** read `topology`, `flow_count` and `evaluation_patterns` from the run artifact, and **raise**
when they are missing rather than defaulting.

---

## A1 / O-RAN (added in Stage H)

### An interface needs a decision, not just an acknowledgement

The v1 constraint endpoint could accept or raise. It could not say "two of those three links exist" or
"that floor exceeds total capacity, here is what is achievable". Partial acceptance is a *distinct*
state: collapsing it into success lets "avoid these six links, four of which are fictional" report as a
complete success.

**Apply:** every A1 decision carries a status, an operator-facing reason, the enforced and rejected
elements, and an achievable alternative where one exists.

### Translate the decision, not the request

A partially-accepted policy must enforce only what the xApp agreed to. Translating the original request
would apply guidance the operator was explicitly told had been refused — worse than either accepting or
rejecting outright.

### A refusal is a 200, not a 4xx

The negotiation is the protocol. Encoding a refusal as a transport error invites the agent layer to treat
it as a failed call and retry, instead of relaying what the xApp actually said.

### An interface with no client is not built

Stage H shipped `nexus-a1`, the xApp's `/a1/*` routes, 35 tests and an ADR — and the agent layer went on
POSTing `Constraint` documents to `/constraints`. Every advertised capability worked; nothing reached it.
Partial acceptance, refusal-with-alternative, the achievable alternative — all unreachable from an
operator, which is the only place they have any value.

**Why:** this is "a contract field nobody reads is a lie" one level up. The consumer side is the easy half
to test in isolation and the easy half to mistake for the whole feature.

**Apply:** a boundary is finished when something on the *other* side calls it. Land the producer in the
same change, and write one test that fails if the old path comes back — here, "no write tool touches
`/constraints`".

### Test the two tiers against each other, not against a mock

Every agent-layer test handed the tools a hand-written controller response. That proves the rApp behaves
correctly given a decision; it cannot prove the two sides agree on URL shape, field names, or enum
spellings — a mock agrees with whatever it was written to agree with.

**Apply:** for a contract between services, drive the real client over ASGI into the real app
(`apps/agent-layer/tests/test_a1_round_trip.py`) and assert on what is *enforced*, not what was returned.
Then break the mapping on purpose and confirm a test fails; one that survives a deliberate bug was never
testing the claim.

---

## Guardrails (added in Stages K–L)

### A validator is not a guardrail

`AllocationAction` rejects shares outside [0, 1] and non-finite values in its own model validator. So
anything holding an `AllocationAction` is holding an action already proven fine — a safety check placed
there inspects only actions that could not be wrong. Worse, the rejection happened as a *parse error* in
the simulator's control loop, which caught it, logged one line, and stepped with no action at all: every
flow at full demand, an unmanaged network that looked managed.

**Why:** a type boundary rejects bad values by refusing to exist. A behavioural guardrail has to *see* the
bad value to name it. The two are not substitutes.

**Apply:** the supervisor parses the controller's response leniently into `ProposedAction`, checks it, and
emits a validated `AllocationAction`. Inspect raw, emit trusted.

### Correct where correction is unambiguous; fall back where it is not

Clamping 1.9 to 1.0 preserves the policy's evident intent. There is no equivalent repair for a NaN — 0.0
starves the flow, 1.0 floods it, and both are inventions. Guessing there turns the guardrail into a
second, unaccountable policy.

**Apply:** an omitted flow is held at its current rate, never zeroed. Absent is an omission; zero is a
decision.

### Hysteresis, or the guardrail becomes an oscillator

The fallback engages because a metric breached, and then *fixes that metric*. Releasing as soon as things
look healthy hands the policy back the network it just broke. Release needs a longer run of healthy steps
than engagement needed breaching ones.

**Apply:** any automatic revert-and-restore. Engage fast on unrepairable failures, slowly on noisy ones,
and always release slowly.

### A fallback action must never wear the policy's name

Relabelling a baseline allocation as `controller` makes the dashboard, the decision log and the agent
layer's explanations all attribute the baseline's behaviour to the policy. It carries
`supervisor:fallback`.

### Shadow mode measures readiness, never superiority

The candidate's actions were never applied, so the network never responded to them and there is no
counterfactual latency to compare. What it can honestly say: would it have been safe, is it responsive, is
it meaningfully different. Which policy is *better* is the offline evaluation suite's verdict, where both
were really applied.

**Apply:** enforce "never applied" structurally — no code path from the shadow proposal to the returned
action — and test it with a value nothing else would produce.

### Randomizing the environment must not cost determinism

Domain randomization varies the network per episode so the policy learns a distribution. Every draw still
comes from an explicitly passed `Generator`, so `(seed, domain)` fully determines the trajectory. And it
is opt-in: `IDENTITY_DOMAIN` by default, because a mode that could silently be on would invalidate every
result recorded before it existed.

**Apply:** also randomize the *conditions*, never the *problem* — the flow set is built from the nominal
topology so two episodes stay comparable.

### `uv sync` alone prunes the workspace

Repeated from Tooling notes because it cost time twice: bare `uv sync` uninstalls the rest of the
workspace and makes `mypy` report errors in files nothing touched. Always `make install`.

---

## Slicing and multi-agent (added in Stage J)

### A per-slice measure that gets averaged is an aggregate again

Mean latency across slices lets a large, fast best-effort slice mask a breached premium one — the exact
blindness slicing exists to remove. `attainment_summary` reports each slice and the count met, and
deliberately reports no mean.

**Apply:** when adding differentiated guarantees, check that no downstream consumer collapses them.

### An SLA is a conjunction; a reward needs a gradient

Both, and they are not in tension. `is_met` is all-terms-hold, because an SLA that is "mostly met" is
broken. `headroom` is the *minimum* term's distance from its limit, because a met/unmet step function
tells the policy nothing about whether it is getting warmer — which is how a policy settles just past a
limit it could cheaply have cleared.

### Weight the priority, and never to zero

An unweighted mean over slices lets the policy satisfy two cheap best-effort slices and abandon premium —
strictly worse, scores better. Weighting by `priority + 1` fixes it; weighting by `priority` alone would
make best-effort worth zero, which is permission to starve it.

### "Decentralized" is a claim the code must support

The easy fake is to hand every agent the global observation with other agents' entries zeroed: the vector
is globally shaped, the tests pass, the reward improves, and the CTDE result means nothing. There is no
symptom to notice later.

**Apply:** the load-bearing test is the negative one — an agent's local observation must not change when
another agent's flows change. `packages/nexus-mdp/tests/test_multi_agent.py`.

### Reward-only changes still bump MDP_VERSION

Adding `sla_attainment_reward` left the observation and action space untouched. It is still a bump: a
version-1 policy optimised a different objective, so its numbers answer a different question. The bump
records *what* changed and *which runs* it invalidates, in `nexus_mdp/version.py`.

### Write to the repo, not next to it

A `Write` with a path missing the repo root created `/Users/vishal/Documents/Repos/nexus-mdp/` — a stray
tree one directory up, outside git, invisible to every gate.

**Apply:** absolute paths for new files start with the project root. Check `git status` after creating
files in a package that shares a name with a directory prefix.
