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

The agent layer has no code path that produces an allocation. Its write tools emit only `A1Policy`
documents — objective reweightings, routing restrictions, per-class floors — over the A1 interface. This is
enforced at the type level (`nexus_a1.policy.A1Policy`) and at the xApp's ingress, which validates every
policy against the live network before deciding what, if anything, it will enforce.

A prompt injection that convinces the model to "set node 3 to 100% bandwidth" produces, at worst, a policy
the xApp is free to refuse — never a direct allocation. See
[ADR-0002](docs/adr/0002-constrained-agent-authority.md) and
[ADR-0008](docs/adr/0008-oran-aligned-a1-interface.md).

There is a second, narrower rule with its own test: no write tool may call `POST /constraints` directly.
That endpoint bypasses A1 validation and returns no decision, so reaching it from the agent layer would
restore exactly the unchecked write A1 replaced.

### Every policy expires

`ttl_timesteps` is mandatory and bounded (`MAX_CONSTRAINT_TTL`) on every A1 policy — a deliberate deviation
from the standard, which requires no policy lifetime. The xApp expires policies itself rather than waiting
to be told to. An agent that goes silent leaves no residue.

### Provenance is recorded

Every policy carries `issued_by` (`rapp:agent-layer`) and the operator utterance that produced it, and the
xApp records the decision that admitted or refused it. Anything the agent layer changed is attributable
after the fact — which is also what makes Use Case 2 (explanation) honest.

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
