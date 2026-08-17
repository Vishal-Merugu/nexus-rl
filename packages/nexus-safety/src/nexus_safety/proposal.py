"""An action as the policy actually produced it, before any type-level validation.

This type exists because of a subtlety that decides whether the safety supervisor is real or decorative.

``AllocationAction`` validates itself: its ``_shares_within_bounds`` validator rejects shares outside
[0, 1], and non-finite values fail the same comparison. So **by the time you are holding an
``AllocationAction``, every interesting failure has already happened somewhere else** — as a parse error
in whoever deserialized it. A guardrail that only ever sees validated data guards nothing; it inspects
actions that were already proven fine.

Worse, that parse error is where the pre-supervisor system degraded badly. The simulator's control loop
catches it, logs it, and steps with *no action at all* — every flow at full demand. A diverged policy
therefore produced an unmanaged network rather than a deliberate fall back to a known-good allocator, and
the only record was one line about malformed JSON.

So the supervisor parses the controller's response **leniently**, into this type, and does its own
checking. The validated ``AllocationAction`` is what it emits, not what it inspects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nexus_errors.control_error import ActionDecodingError


@dataclass(frozen=True, slots=True)
class ProposedAction:
    """What the controller said, parsed but not judged."""

    share_by_flow_id: dict[str, float]
    path_by_flow_id: dict[str, tuple[str, ...]] = field(default_factory=dict)
    issued_by: str = "controller"


def parse_proposed_action(payload: object) -> ProposedAction:
    """Read a controller's ``/act`` response without enforcing any bound.

    Only *structural* failures raise — a response that is not an object, or shares that are not numbers at
    all. Those are not policy misbehaviour the supervisor can correct; they mean the thing on the other
    end is not speaking the action contract, and there is nothing to guard.

    A share of ``1.9`` or ``NaN`` parses cleanly here **on purpose**. Judging it is
    :func:`nexus_safety.bounds.check_action`'s job, and it can only do that job if the value survives
    long enough to be judged.
    """
    if not isinstance(payload, dict):
        raise ActionDecodingError(
            f"Controller returned a {type(payload).__name__}, not an action object.",
            {"got": type(payload).__name__},
        )

    raw_shares = payload.get("share_by_flow_id")
    if not isinstance(raw_shares, dict):
        raise ActionDecodingError(
            "Controller response has no 'share_by_flow_id' object, so there is no action to check.",
            {"got": type(raw_shares).__name__},
        )

    shares: dict[str, float] = {}
    for flow_id, value in raw_shares.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            # bool is an int in Python; `true` would silently become a share of 1.0.
            raise ActionDecodingError(
                f"Share for {flow_id!r} is a {type(value).__name__}, not a number.",
                {"flow_id": str(flow_id), "value": repr(value)},
            )
        shares[str(flow_id)] = float(value)

    return ProposedAction(
        share_by_flow_id=shares,
        path_by_flow_id=_parse_paths(payload.get("path_by_flow_id")),
        issued_by=_parse_issuer(payload.get("issued_by")),
    )


def _parse_paths(raw: object) -> dict[str, tuple[str, ...]]:
    """Routes, keeping only well-formed entries.

    A malformed route is dropped rather than raising: the flow keeps its default path, which is a safe
    outcome, and refusing the whole action over one bad route would escalate a routing typo into a
    fallback.
    """
    if not isinstance(raw, dict):
        return {}
    paths: dict[str, tuple[str, ...]] = {}
    for flow_id, path in raw.items():
        if isinstance(path, list | tuple) and all(isinstance(link, str) for link in path):
            paths[str(flow_id)] = tuple(str(link) for link in path)
    return paths


def _parse_issuer(raw: object) -> str:
    return raw if isinstance(raw, str) and raw else "controller"
