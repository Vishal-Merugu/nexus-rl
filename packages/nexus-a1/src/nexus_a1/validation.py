"""Validating an A1 policy against the network the xApp is actually controlling.

Schema validity is not feasibility. A policy can be perfectly well-formed and still be impossible: it can
name links that do not exist, ask for a floor larger than the network's total capacity, or forbid every
path a flow has. The xApp is the only component that knows the current network, so it is the only one
that can say so.

Every check here answers in the same shape — accept, partially accept, or reject, always with a reason an
operator can act on. Nothing silently narrows a policy and reports success.
"""

from __future__ import annotations

from nexus_a1.decision import A1PolicyDecision, A1PolicyStatus
from nexus_a1.policy import A1Policy, A1PolicyType
from nexus_a1.version import MIN_SUPPORTED_SCHEMA_VERSION
from nexus_types.constants import MAX_CONSTRAINT_TTL
from nexus_types.network_state import NetworkState

SUPPORTED_POLICY_TYPES: frozenset[A1PolicyType] = frozenset(A1PolicyType)
"""What this xApp advertises it can enforce. An unadvertised type is refused, not partially honoured."""


def validate_policy(
    policy: A1Policy,
    state: NetworkState,
    current_timestep: int,
    supported_types: frozenset[A1PolicyType] = SUPPORTED_POLICY_TYPES,
) -> A1PolicyDecision:
    """Decide whether the xApp can enforce a policy against the network it currently sees.

    Ordered cheapest-first: version, then type, then timing, then feasibility against the live topology.
    A policy that fails an early check is refused without pretending to reason about the later ones.
    """
    version_decision = _check_schema_version(policy)
    if version_decision is not None:
        return version_decision

    if policy.policy_type not in supported_types:
        return _reject(
            policy,
            f"This controller does not implement {policy.policy_type.value} policies. It advertises: "
            f"{', '.join(sorted(t.value for t in supported_types))}.",
        )

    timing_decision = _check_timing(policy, current_timestep)
    if timing_decision is not None:
        return timing_decision

    if policy.policy_type is A1PolicyType.TRAFFIC_STEERING:
        return _validate_traffic_steering(policy, state)
    if policy.policy_type is A1PolicyType.CLASS_FLOOR:
        return _validate_class_floor(policy, state)
    return _validate_qos_objective(policy)


def _check_schema_version(policy: A1Policy) -> A1PolicyDecision | None:
    from nexus_a1.version import A1_SCHEMA_VERSION

    if policy.schema_version > A1_SCHEMA_VERSION:
        return _reject(
            policy,
            f"Policy is written against A1 schema version {policy.schema_version}; this controller "
            f"understands up to {A1_SCHEMA_VERSION}. Interpreting it would mean guessing at fields this "
            f"version does not define.",
        )
    if policy.schema_version < MIN_SUPPORTED_SCHEMA_VERSION:
        return _reject(
            policy,
            f"Policy uses retired A1 schema version {policy.schema_version}; the oldest supported is "
            f"{MIN_SUPPORTED_SCHEMA_VERSION}.",
        )
    return None


def _check_timing(policy: A1Policy, current_timestep: int) -> A1PolicyDecision | None:
    if policy.ttl_timesteps > MAX_CONSTRAINT_TTL:
        return _reject(
            policy,
            f"TTL {policy.ttl_timesteps} exceeds the maximum of {MAX_CONSTRAINT_TTL} timesteps. Every "
            f"operator override must expire.",
        )
    # One timestep of slack: the issuer reads the current timestep, then posts, and the simulation may
    # have advanced once in between. More than that is a policy that would activate at an unpredictable
    # moment.
    if policy.issued_at_timestep > current_timestep + 1:
        return _reject(
            policy,
            f"Policy is stamped at timestep {policy.issued_at_timestep} but the network is at "
            f"{current_timestep}. It would take effect at an unpredictable point; re-issue it against "
            f"the current timestep.",
        )
    return None


def _validate_traffic_steering(policy: A1Policy, state: NetworkState) -> A1PolicyDecision:
    """Steering can be partly honoured: some named links may not exist, and some flows may have no
    alternative path.

    A flow whose every candidate path touches a forbidden link is *reported*, not silently black-holed —
    the decoder keeps routing it, so the operator must be told the restriction will not hold there.
    """
    known = {link.link_id for link in state.links}
    requested = policy.statement.avoid_link_ids
    enforced = tuple(link_id for link_id in requested if link_id in known)
    unknown = tuple(link_id for link_id in requested if link_id not in known)

    if not enforced:
        return _reject(
            policy,
            f"None of the named links exist in this topology: {', '.join(unknown)}.",
            rejected_elements=unknown,
            achievable_alternative=(
                f"Links currently in the topology: {', '.join(sorted(known))}."
                if known
                else "The topology reports no links."
            ),
        )

    forbidden = frozenset(enforced)
    stranded = tuple(
        flow.flow_id
        for flow in state.flows
        if flow.candidate_paths
        and all(any(link_id in forbidden for link_id in path) for path in flow.candidate_paths)
    )

    if unknown or stranded:
        notes: list[str] = []
        if unknown:
            notes.append(f"unknown links ignored: {', '.join(unknown)}")
        if stranded:
            notes.append(
                f"flows with no alternative path, which will keep using a named link rather than be "
                f"dropped: {', '.join(stranded)}"
            )
        return A1PolicyDecision(
            policy_id=policy.policy_id,
            status=A1PolicyStatus.PARTIALLY_ACCEPTED,
            reason=f"Steering applied to {', '.join(enforced)}; {'; '.join(notes)}.",
            enforced_elements=enforced,
            rejected_elements=unknown + stranded,
            achievable_alternative=(
                "A constraint may never black-hole traffic, so stranded flows keep their path. Add "
                "capacity or free one of the named links to steer them."
                if stranded
                else None
            ),
        )

    return _accept(policy, f"Steering new traffic away from {', '.join(enforced)}.", enforced)


def _validate_class_floor(policy: A1Policy, state: NetworkState) -> A1PolicyDecision:
    """A floor larger than the network can carry is infeasible, and saying so is the point.

    Total nominal capacity is a generous ceiling — no single class can receive all of it — so a floor
    above it is unambiguously impossible rather than merely unlikely.
    """
    floors = policy.statement.class_floor_mbps or {}
    total_capacity = sum(link.effective_capacity_mbps for link in state.links)

    infeasible = {traffic_class: floor for traffic_class, floor in floors.items() if floor > total_capacity}
    feasible = {traffic_class: floor for traffic_class, floor in floors.items() if floor <= total_capacity}

    if not feasible:
        largest = max(infeasible.values(), default=0.0)
        return _reject(
            policy,
            f"A floor of {largest:.0f} Mb/s exceeds the network's total capacity of "
            f"{total_capacity:.0f} Mb/s, so it cannot be guaranteed under any allocation.",
            rejected_elements=tuple(sorted(c.value for c in infeasible)),
            achievable_alternative=(
                f"The largest floor this topology could honour for a single class is below "
                f"{total_capacity:.0f} Mb/s."
            ),
        )

    if infeasible:
        return A1PolicyDecision(
            policy_id=policy.policy_id,
            status=A1PolicyStatus.PARTIALLY_ACCEPTED,
            reason=(
                f"Floors applied for {', '.join(sorted(c.value for c in feasible))}. Floors for "
                f"{', '.join(sorted(c.value for c in infeasible))} exceed the network's total capacity "
                f"of {total_capacity:.0f} Mb/s and were not applied."
            ),
            enforced_elements=tuple(sorted(c.value for c in feasible)),
            rejected_elements=tuple(sorted(c.value for c in infeasible)),
            achievable_alternative=f"Reduce those floors below {total_capacity:.0f} Mb/s.",
        )

    return _accept(
        policy,
        "Class floors applied: "
        + ", ".join(f"{c.value} ≥ {floor:.0f} Mb/s" for c, floor in sorted(feasible.items())),
        tuple(sorted(c.value for c in feasible)),
    )


def _validate_qos_objective(policy: A1Policy) -> A1PolicyDecision:
    """Objective reweighting is always feasible — it changes what is optimized, not what is possible.

    Worth stating explicitly rather than leaving as a fallthrough: this is the one policy type that
    cannot be infeasible, because it never promises an outcome.
    """
    objective = policy.statement.qos_objective
    assert objective is not None  # guaranteed by A1Policy._statement_matches_type
    return _accept(
        policy,
        "Objective reweighted to "
        f"throughput={objective.throughput:g}, latency={objective.latency:g}, "
        f"drops={objective.drops:g}, fairness={objective.fairness:g}. This changes what the controller "
        "optimizes for; it does not retrain the policy and does not guarantee a specific outcome.",
    )


def _accept(policy: A1Policy, reason: str, enforced: tuple[str, ...] = ()) -> A1PolicyDecision:
    return A1PolicyDecision(
        policy_id=policy.policy_id,
        status=A1PolicyStatus.ACCEPTED,
        reason=f"{reason} Expires at timestep {policy.expires_at_timestep()}.",
        enforced_elements=enforced,
    )


def _reject(
    policy: A1Policy,
    reason: str,
    rejected_elements: tuple[str, ...] = (),
    achievable_alternative: str | None = None,
) -> A1PolicyDecision:
    return A1PolicyDecision(
        policy_id=policy.policy_id,
        status=A1PolicyStatus.REJECTED,
        reason=reason,
        rejected_elements=rejected_elements,
        achievable_alternative=achievable_alternative,
    )
