"""The action contract — what the controller emits and the simulator applies.

Only the controller produces these. There is no code path from the agent layer to an ``AllocationAction``;
the outer tier writes :mod:`nexus_types.constraint` objects instead. See ADR-0002.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus_errors.control_error import ActionDecodingError


class AllocationAction(BaseModel):
    """A per-flow capacity allocation for one timestep.

    ``share_by_flow_id`` gives each flow's share of its bottleneck link capacity, in [0, 1]. Shares are
    per-flow rather than absolute rates so that an action stays meaningful when link capacity changes under an
    incident — the policy expresses intent, and the simulator resolves it against whatever capacity exists.
    """

    model_config = ConfigDict(frozen=True)

    share_by_flow_id: dict[str, float] = Field(
        description="flow_id → share of bottleneck capacity, each in [0, 1]."
    )
    path_by_flow_id: dict[str, tuple[str, ...]] = Field(
        default_factory=dict,
        description="flow_id → chosen path as ordered link_ids. Empty means keep the flow's default path.",
    )
    issued_by: str = Field(
        default="controller",
        description="Provenance: 'controller', or 'baseline:<name>' when no policy is loaded.",
    )

    @model_validator(mode="after")
    def _shares_within_bounds(self) -> AllocationAction:
        for flow_id, share in self.share_by_flow_id.items():
            if not 0.0 <= share <= 1.0:
                raise ActionDecodingError(
                    "Allocation share outside [0, 1]. This is usually a shape mismatch between the policy's "
                    "action space and the environment — check MDP_VERSION.",
                    {"flow_id": flow_id, "share": share},
                )
        return self
