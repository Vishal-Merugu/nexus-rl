"""What the xApp says back when handed a policy.

A1 is a negotiation, not a command channel. The Near-RT tier retains control authority, which means it
must be able to answer "no" and "only partly" — and say *which* part and *why*, in terms an operator can
act on.

This is the type behind Use Case 8. Silently accepting an infeasible policy and then failing to honour it
is the worst available outcome: the operator believes a guarantee is in force, and nothing in the system
contradicts them.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class A1PolicyStatus(StrEnum):
    """The outcome of offering a policy to the xApp."""

    ACCEPTED = "ACCEPTED"
    """Enforced in full."""

    PARTIALLY_ACCEPTED = "PARTIALLY_ACCEPTED"
    """Some scope elements are enforced and others are not. ``rejected_elements`` names the rest.

    Deliberately distinct from ACCEPTED: collapsing the two would let "avoid these six links, four of
    which do not exist" report as complete success.
    """

    REJECTED = "REJECTED"
    """Nothing is enforced. The policy has no effect on the network."""


class A1PolicyDecision(BaseModel):
    """The xApp's response to an offered policy.

    ``reason`` is written for an operator, not a log grep: it says what the system will and will not do,
    and where possible what would work instead. The agent layer relays it verbatim rather than
    paraphrasing, because a paraphrase of a refusal is where overclaiming creeps back in.
    """

    model_config = ConfigDict(frozen=True)

    policy_id: str
    status: A1PolicyStatus
    reason: str = Field(
        description="Operator-facing explanation. Always populated, including on ACCEPTED, so the agent "
        "can state what was actually applied rather than inferring it."
    )
    enforced_elements: tuple[str, ...] = Field(
        default=(), description="Scope elements the xApp will act on, e.g. link ids or class names."
    )
    rejected_elements: tuple[str, ...] = Field(
        default=(), description="Scope elements the xApp will not act on."
    )
    achievable_alternative: str | None = Field(
        default=None,
        description="What the xApp could honour instead, when the request is infeasible as stated. This "
        "is what turns a refusal into a negotiation.",
    )

    @property
    def is_enforced(self) -> bool:
        """Whether any part of this policy affects the network."""
        return self.status is not A1PolicyStatus.REJECTED
