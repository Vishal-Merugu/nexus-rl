# 0007 — The MDP contract is a separate package from the gymnasium wrapper

Date: 2026-08-15
Status: Accepted

## Context

Two groups of code need the MDP:

1. **Training** needs a gymnasium `Env` — `reset`, `step`, `observation_space`, `action_space`, and the
   Stable-Baselines3 conventions around them.
2. **The controller service** needs only the *codec*: `encode_observation`, `decode_discrete_action`,
   `decode_continuous_action`, and `MDP_VERSION`. It never constructs an `Env`, never calls `reset`, and
   never runs an episode. It receives a state over HTTP, encodes it, asks a policy for an action, decodes
   it, and replies.

Originally both lived in `nexus-env`. Because that package depends on `gymnasium`, the controller — a
production service in its own container — pulled gymnasium in transitively to use four pure functions.
The agent layer did too, through the same chain.

That is not just image weight. It puts a training framework in the dependency closure of a service whose
correctness has nothing to do with training, and it makes "the MDP is defined in exactly one place" a
convention maintained by discipline rather than a property of the package graph. Nothing structurally
prevented a second copy of the observation encoding from appearing in the controller, and the rule most
worth enforcing here is precisely that there is only ever one.

## Decision

Split the MDP contract into its own package.

**`nexus-mdp`** — pure numpy. `observation.py`, `action_space.py`, `reward.py`, `version.py`. Depends on
`nexus-types`, `nexus-sim` (for physical constants) and `nexus-errors`. **No gymnasium, no SB3, no torch.**
This is the contract.

**`nexus-env`** — the gymnasium `Env` that wraps `nexus-sim` against that contract. Depends on
`nexus-mdp` and `gymnasium`. Training only.

The rule that follows: **nothing under `apps/` imports `nexus-env`.** The controller imports `nexus-mdp`.

## Consequences

- **"The MDP lives in one place" is now enforced by the package graph.** A service cannot accidentally
  depend on the gym wrapper, and a second encoder in the controller would require a new dependency edge
  that is visible in a diff.
- **Service images shed gymnasium and its transitive dependencies.** The controller image carries numpy and
  pydantic where it used to carry a training stack.
- **`MDP_VERSION` moved** from `nexus_env/version.py` to `nexus_mdp/version.py`. Any doc still naming the
  old path is stale.
- **One more package to maintain** — twelve rather than eleven. The boundary is real (different consumers,
  different dependency needs), so this is the good kind of package split rather than premature layering.
- **The split makes the review surface for an MDP change smaller and sharper.** `mdp-reviewer` runs against
  one package containing four modules, none of which imports a framework.
- **`nexus-mdp` depends on `nexus-sim` for `MAX_QUEUE_MB`, `MAX_LINK_LATENCY_MS` and `EPSILON`.** These are
  physical constants of the simulated world that the normalisation divides by, so the direction is correct
  — but it does mean the contract package is not free-standing. If a second simulator ever lands (v2
  Track C, ns-3), those constants move to a shared home rather than being duplicated.

## Alternatives considered

- **Keep one package and make gymnasium an optional extra.** Rejected: an optional dependency that is
  imported at module scope in `environment.py` still breaks the import for anyone without it, and guarding
  it behind a lazy import scatters `try/except ImportError` through the package for a boundary that is
  better expressed as two packages.
- **Put the codec in `nexus-types`.** Rejected: `nexus-types` is pydantic contracts crossing service
  boundaries and deliberately carries no numpy. The encoding is logic, not a wire type.
- **Let the controller depend on `nexus-env` and accept the weight.** Rejected: this is the status quo the
  split exists to end. The weight was the visible symptom; the real cost was that single-definition of the
  MDP rested on nobody making a reasonable-looking mistake.

## Related

- ADR-0004 — discrete DQN before continuous PPO; both action spaces live in `nexus-mdp`.
- ADR-0006 — the controller decodes from received state, using this package's decoder.
- `tasks/lessons.md` — "Split the contract from the framework that consumes it".
