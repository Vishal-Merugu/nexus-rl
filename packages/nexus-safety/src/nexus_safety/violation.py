"""What a guardrail caught, and what it did about it.

A violation is a *finding*, not an error. The supervisor's job is to keep the network running while
refusing to let a policy do something harmful, so the normal outcome of a violation is a corrected action
and a log line — not an exception that stops the control loop.

Every violation names the flow or link it concerns. "The action was unsafe" is not actionable at 2am;
"f4's share was 1.9, clamped to 1.0" is.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ViolationKind(StrEnum):
    """The bound that was crossed.

    Each of these is something a learned policy can genuinely produce. Non-finite shares come out of a
    diverged network; out-of-range shares out of an action-space mismatch; unknown flows and invalid paths
    out of a policy meeting a network it was not trained on.
    """

    NON_FINITE_SHARE = "non_finite_share"
    """NaN or infinity. Almost always a diverged policy rather than a decoding bug — and it must never
    reach the simulator, where it would propagate into every downstream metric as a silent NaN."""

    SHARE_OUT_OF_RANGE = "share_out_of_range"
    """Outside [0, 1]. Usually an action-space/MDP_VERSION mismatch."""

    UNKNOWN_FLOW = "unknown_flow"
    """The action allocates to a flow the network does not contain."""

    MISSING_FLOW = "missing_flow"
    """A flow in the network received no allocation at all. Distinct from a zero share: zero is a
    decision, absent is an omission, and they must not be conflated."""

    INVALID_PATH = "invalid_path"
    """A route that is not among the flow's candidate paths. Enforced because the simulator would
    otherwise carry traffic over links the topology does not connect."""

    STARVATION = "starvation"
    """A flow held at or near zero for longer than the configured window. The one bound that is about
    *behaviour over time* rather than a single action — a policy that permanently starves a flow to
    optimise an aggregate is the classic degenerate optimum."""


class Violation(BaseModel):
    """One bound crossed by one action."""

    model_config = ConfigDict(frozen=True)

    kind: ViolationKind
    subject: str = Field(description="The flow or link the finding concerns, e.g. 'f4'.")
    detail: str = Field(description="Operator-facing explanation, naming the offending value.")
    corrected_to: float | None = Field(
        default=None,
        description="The value the supervisor substituted, when it could correct rather than fall back.",
    )
