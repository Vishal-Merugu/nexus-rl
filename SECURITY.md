# Security Policy

NEXUS-RL is a simulation and research project. It manages no production traffic and holds no user data. The
security surface that does exist is concentrated in one place: **what the LLM agent layer is allowed to do.**

## Reporting a vulnerability

Open a GitHub issue with the `security` label, or contact the maintainer directly for anything that should not
be public. There is no bounty programme.

## Threat model

In scope:

- **Over-broad agent authority.** An LLM that can issue raw control actions can destabilise the network it is
  supposed to manage — including via prompt injection through operator input.
- **Unbounded operator overrides.** A constraint with no expiry silently becomes permanent policy.
- **Secret leakage.** LLM API keys reaching logs, run artifacts, container images, or the dashboard.
- **Unauthenticated control surfaces.** Service APIs that mutate simulation or controller state.

Out of scope:

- Attacks against the simulated network itself (that is a *feature* — see `Incident` in `CONTEXT.md`).
- Denial of service against local development containers.
- Anything about real network hardware. There is none.

## Controls

### Agent authority is structurally constrained

The agent layer has no code path that produces an allocation. Its write tools accept only `Constraint`
objects — objective reweightings, action masks, per-class floors. This is enforced at the type level
(`nexus_types.Constraint`) and at the controller's ingress, which rejects anything else.

A prompt injection that convinces the model to "set node 3 to 100% bandwidth" produces, at worst, a constraint
the controller optimises within — never a direct allocation. See
[ADR-0002](docs/adr/0002-constrained-agent-authority.md).

### Every constraint expires

`ttl_timesteps` is mandatory and bounded (`MAX_CONSTRAINT_TTL`). The controller expires constraints itself
rather than waiting to be told to. An agent that goes silent leaves no residue.

### Provenance is recorded

Every constraint carries `issued_by` and the operator utterance that produced it. Anything the agent layer
changed is attributable after the fact — which is also what makes Use Case 2 (explanation) honest.

### Secrets

- Read only through `nexus-config`; never `os.environ` at a call site.
- `.env` is gitignored; `.env.example` carries names and never values.
- Run artifacts under `experiments/runs/` are gitignored, and the config snapshot written into each run
  redacts any key whose name matches `*_KEY`, `*_SECRET`, `*_TOKEN`.
- Container images never bake in credentials — they are injected at runtime.

### Service surfaces

Local development binds services to localhost. The Kubernetes manifests expose only the dashboard; simulator,
controller and agent layer are `ClusterIP` services. Mutating endpoints (`/step`, `/constraints`,
`/incidents`) require the shared `NEXUS_API_KEY` header when `NEXUS_ENV != "development"`.

## Known limitations

- The controller trusts the simulator's state. In a real deployment that channel would need integrity
  protection; here both are inside one trust boundary.
- There is no multi-tenancy. One simulation, one operator, one set of constraints.
- Rate limiting on the agent layer is per-process and in-memory. Adequate for a demo, not for exposure.
