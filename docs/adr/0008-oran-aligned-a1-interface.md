# 0008 — The two tiers speak an A1-shaped policy interface

Date: 2026-08-15
Status: Accepted

## Context

v1 arrived at a two-tier design from first principles: a slow, language-capable supervisor that issues
*constraints*, and a fast numeric controller that optimizes within them and retains sole authority to
allocate (ADR-0002). The boundary was chosen because letting a language model emit raw allocations is
unsafe, not because any standard said so.

That structure is, almost exactly, the one the O-RAN Alliance standardized:

| NEXUS-RL (v1)                     | O-RAN                                        | Interface |
| --------------------------------- | -------------------------------------------- | --------- |
| RL controller service             | xApp on a Near-RT RIC (10 ms – 1 s loop)     | A1 (consumes policy), E2 (to network elements) |
| LLM agent layer                   | rApp on a Non-RT RIC within SMO (> 1 s loop) | A1 (issues policy), R1 |
| "Constrained authority"           | A1 policy guidance                            | A1 |
| Network simulator                 | E2 nodes (stand-ins for O-CU / O-DU)          | E2 |
| Dashboard                         | SMO-side operator console                     | O1-style |

The convergence is the interesting part. The Non-RT RIC issues policy and does *not* issue control
actions; the Near-RT RIC decides how to satisfy that policy. That is ADR-0002 restated in the industry's
own vocabulary.

Two things follow. First, the project can describe itself in terms a telecom engineer already knows,
which is worth more than a bespoke vocabulary that has to be explained from scratch. Second — and this is
the part that changes the code — a real interface has properties v1's constraint POST did not: a
**version**, an advertised set of **policy types**, and a **decision** in response, including partial
acceptance and refusal with a reason.

v1's constraint endpoint accepted or raised. It had no way to say "I can do two of those three links" or
"that floor exceeds the network's total capacity, here is what is achievable". Those answers are the
substance of an autonomous system an operator can actually trust, and their absence was the real gap —
the O-RAN framing is what made it visible.

## Decision

Model the boundary as an **A1-shaped policy interface**, in a new `nexus-a1` package.

An `A1Policy` carries a `policy_type`, a `scope`, a `statement`, a `schema_version`, provenance, and a
bounded TTL. The xApp (controller) answers every offer with an `A1PolicyDecision`:

- `ACCEPTED` — enforced in full
- `PARTIALLY_ACCEPTED` — some scope elements enforced, the rest named in `rejected_elements`
- `REJECTED` — nothing enforced, with an `achievable_alternative` where one exists

The controller exposes `/a1/policytypes`, `PUT|DELETE /a1/policies/{id}`, `GET /a1/policies`, and
`GET /a1/policies/{id}/status`. A refusal is a **200 with a REJECTED decision**, not an HTTP error: the
negotiation is the protocol, not an exception to it, and the agent layer must relay the reason rather
than treat it as a transport failure.

**A1 is the interface; `Constraint` remains the enforcement.** An accepted policy translates into the v1
constraint the control path already knows how to expire, precedence-resolve and apply. Nothing in `/act`
changed. That is what makes this track additive rather than a rewrite.

Three deliberate deviations from the standard, all in the direction of being stricter:

1. **TTL is mandatory and bounded.** A1 has no required policy lifetime. An operator override that never
   expires becomes undocumented permanent policy nobody remembers setting — in a system whose selling
   point is autonomy, that is the failure that discredits it.
2. **Feasibility is validated against the network as last observed**, not merely against the schema. A
   policy offered before any state has arrived is refused rather than accepted blind.
3. **A policy id already in force is refused, not treated as an update.** Deletion removes every match,
   so a reused id would make a later withdrawal ambiguous.

## Consequences

- **The system can now say "no", and say why.** Use Case 8 (policy rejection and negotiation) works:
  ask for a floor above total capacity and the xApp refuses with the capacity figure and an achievable
  alternative. This is the single largest behavioural gain of the track.
- **Partial acceptance is a distinct state.** "Avoid these six links, four of which do not exist" reports
  as `PARTIALLY_ACCEPTED` with both lists, where v1 would have applied two and reported success.
- **Translation follows the decision, not the request.** A partially-accepted steering policy enforces
  only the links the xApp agreed to — otherwise the operator is told a link was ignored while the
  controller quietly acts on it.
- **The interface is versioned and negotiated.** A policy written against a newer schema is refused
  rather than partially interpreted. The rApp and xApp are separately deployed and either can be older.
- **This is architectural alignment, not conformance.** Nothing here is tested against an O-RAN
  conformance suite, the transport is plain HTTP rather than the standard's own bindings, and E2 remains
  the existing `/act` call rather than a real E2AP implementation. The honest claim is "structured
  according to O-RAN's architecture", never "O-RAN compliant" — and the README says so.
- **One more package.** `nexus-a1` depends only on `nexus-types` and `nexus-errors`, so it stays a wire
  contract and does not drag the simulator into the agent layer.
- **The v1 constraint path is now internal.** The agent layer issues A1 policies exclusively; nothing in
  `write_tools.py` writes a `Constraint`, and a test asserts no write tool touches `/constraints`.
  `POST /constraints` still exists on the controller for direct operator use, but it is no longer on any
  path the LLM can reach. Retiring that endpoint entirely is a follow-up, not part of this ADR.
- **The rApp's write surface grew by one tool.** The xApp advertises three policy types, and until the rApp
  could issue a `CLASS_FLOOR` it advertised a capability no client could use — the interface-level form of
  the "contract field nobody reads" failure. `set_class_floor` closes that, and it is also the only policy
  type the network can make *impossible*, which is what makes Use Case 8 demonstrable rather than
  theoretical. `revoke_constraint` became `withdraw_policy` for the same reason: withdrawal has to key on
  the policy id the operator was given, or the xApp's policy store and its constraint store drift apart.

## Alternatives considered

- **Deploy the ORAN-SC reference Near-RT RIC and host a real xApp on it.** Rejected for now: it is a
  substantial microservice platform with its own Kubernetes footprint, and the architectural literacy is
  what an interview probes. Building A1-shaped interfaces and documenting the mapping delivers most of
  the value at a fraction of the effort — this is Decision 6 in the v2 spec, and it stays open.
- **Rename the existing constraint types to O-RAN vocabulary and stop there.** Rejected: that is bolting
  on terminology. The parts of A1 that are genuinely valuable — versioning, advertised policy types,
  partial acceptance, refusal with a reason — are behaviour, not naming.
- **Replace `Constraint` with `A1Policy` throughout the controller.** Rejected: the constraint machinery
  is the tested enforcement path (expiry, last-writer-wins objectives, accumulating restrictions, class
  floors). Keeping A1 as the interface and the constraint as the internal representation means the O-RAN
  work adds a boundary without destabilising the control loop.
- **Return HTTP 4xx for a rejected policy.** Rejected: a refusal is a normal, expected outcome that
  carries an operator-facing explanation. Encoding it as a transport error invites the agent layer to
  handle it as a failed call and retry, rather than relaying what the xApp actually said.

## Related

- ADR-0002 — constrained agent authority; A1 is the standardised expression of the same rule.
- ADR-0006 — the controller decodes from received state, which is why A1 feasibility is checked against
  the last observed network rather than a configured model.
- `CONTEXT.md` — O-RAN vocabulary (xApp, rApp, Near-RT/Non-RT RIC, A1).
