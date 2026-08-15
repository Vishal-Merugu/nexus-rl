"""Policy hosting, with an honest fallback.

The controller starts fine with no checkpoint: it serves the ``static`` baseline and says so on ``/health``.
That is what makes Stages A, C, D and E reachable before any training has happened — and it is why the
health check reports *which* allocator is live. A dashboard showing unremarkable allocations is a very
different situation depending on whether a policy is loaded, and guessing wastes an afternoon.

Stable-Baselines3 and torch are imported lazily, so an image without them still runs the fallback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nexus_baselines.allocator import Allocator, StaticAllocator
from nexus_errors.control_error import PolicyLoadError
from nexus_mdp.action_space import (
    FlowRoutingOptions,
    decode_continuous_action,
    decode_discrete_action,
    reroute_paths,
)
from nexus_mdp.observation import encode_observation
from nexus_mdp.version import MDP_VERSION
from nexus_telemetry.client import get_telemetry
from nexus_types.action import AllocationAction
from nexus_types.network_state import NetworkState

_telemetry = get_telemetry("nexus.controller.policy")


class PolicyHost:
    """Serves allocations, from a trained policy when one is loaded and from a baseline otherwise."""

    def __init__(self, action_mode: str = "discrete") -> None:
        self._action_mode = action_mode
        self._model: object | None = None
        self._fallback: Allocator | None = None
        self._policy_label = f"baseline:{StaticAllocator.name}"

    @property
    def policy_label(self) -> str:
        """What is actually serving allocations. Reported on /health — check this before debugging."""
        return self._policy_label

    @property
    def has_trained_policy(self) -> bool:
        return self._model is not None

    def load(self, checkpoint: Path, algorithm: str = "dqn") -> None:
        """Load a checkpoint, or leave the fallback in place and say why.

        A failed load is logged and survived rather than fatal: a controller that refuses to start because
        of a stale checkpoint takes the whole demo with it, when running unmanaged would have been fine.
        """
        try:
            from stable_baselines3 import DQN, PPO
        except ImportError as error:
            raise PolicyLoadError(
                "Serving a trained policy needs the optional 'policy' extra: "
                "uv sync --all-packages --extra policy",
                {"missing": str(error)},
            ) from error

        if not checkpoint.exists():
            raise PolicyLoadError(f"No checkpoint at {checkpoint}.", {"path": str(checkpoint)})

        loader = DQN if algorithm == "dqn" else PPO
        self._model = loader.load(checkpoint)
        self._policy_label = f"{algorithm}:{checkpoint.stem}@mdp{MDP_VERSION}"
        _telemetry.info(
            "Policy loaded",
            {"checkpoint": str(checkpoint), "algorithm": algorithm, "mdp_version": MDP_VERSION},
        )

    def try_load(self, checkpoint: Path | None, algorithm: str = "dqn") -> None:
        """Load if configured; fall back loudly on failure."""
        if checkpoint is None:
            _telemetry.info(
                "No checkpoint configured — serving the static baseline. This is a supported mode, not "
                "an error. /health reports the active allocator.",
                {"policy": self._policy_label},
            )
            return
        try:
            self.load(checkpoint, algorithm)
        except PolicyLoadError as error:
            _telemetry.capture_exception(
                error, "Checkpoint failed to load — continuing on the static baseline"
            )

    @staticmethod
    def _routing_from(state: NetworkState) -> tuple[FlowRoutingOptions, ...]:
        """Read the action space out of the state the simulator sent.

        The controller used to rebuild the flow set from its own ``NEXUS_SEED``/``NEXUS_TOPOLOGY``. Flow ids
        are positional (``f0``..``fn``), so a config mismatch produced ids that lined up perfectly while the
        routes behind them belonged to different flows — wrong paths, no error anywhere. Decoding from the
        received state makes that divergence impossible rather than merely unlikely.
        """
        return tuple(
            FlowRoutingOptions(flow_id=flow.flow_id, candidate_paths=flow.candidate_paths)
            for flow in state.flows
        )

    def _ensure_fallback(self, state: NetworkState) -> Allocator:
        """Derive the static baseline's equal rate from the network as observed, once."""
        if self._fallback is None:
            mean_capacity = (
                sum(link.capacity_mbps for link in state.links) / len(state.links) if state.links else 0.0
            )
            self._fallback = StaticAllocator(equal_rate_mbps=mean_capacity / max(1, len(state.flows)))
        return self._fallback

    def act(
        self,
        state: NetworkState,
        forbidden_link_ids: frozenset[str] = frozenset(),
    ) -> AllocationAction:
        """Produce an allocation for one timestep.

        Operator constraints restrict *routing* here; the policy still chooses rates freely within what
        remains. That asymmetry is constrained authority in code: the agent layer shapes the space the
        controller optimizes in, and never the allocation itself.
        """
        if self._model is None:
            # The fallback keeps its own rates and provenance; constraints only add routing on top.
            # Substituting a decoded OPEN action here (the previous behaviour) silently discarded the
            # baseline's throttling the moment any constraint existed, and reported the result as coming
            # from the controller rather than from a baseline.
            action = self._ensure_fallback(state).allocate(state)
            if not forbidden_link_ids:
                return action
            return action.model_copy(
                update={
                    "path_by_flow_id": reroute_paths(
                        state=state,
                        routing=self._routing_from(state),
                        forbidden_link_ids=forbidden_link_ids,
                    )
                }
            )

        observation = encode_observation(state)
        return self._predict(observation, state, forbidden_link_ids)

    def _predict(
        self,
        observation: NDArray[np.float32],
        state: NetworkState,
        forbidden_link_ids: frozenset[str],
    ) -> AllocationAction:
        model = self._model
        if model is None:  # narrowing for the type checker; act() guarantees this
            raise PolicyLoadError("No policy loaded.", {})

        try:
            raw_action, _ = model.predict(observation, deterministic=True)  # type: ignore[attr-defined]  # SB3 models are untyped
        except (ValueError, RuntimeError) as error:
            _telemetry.capture_exception(
                error,
                "Policy inference failed — falling back for this timestep. A shape error here usually "
                "means MDP_VERSION moved since the checkpoint was trained.",
            )
            return self._ensure_fallback(state).allocate(state)

        if self._action_mode == "discrete":
            return decode_discrete_action(
                int(np.asarray(raw_action).item()), state, self._routing_from(state), forbidden_link_ids
            )
        return decode_continuous_action(np.asarray(raw_action, dtype=np.float32).ravel(), state)


def build_default_host() -> PolicyHost:
    """A host serving the static baseline. What the controller runs with no checkpoint.

    Takes no topology or flow arguments: everything the host needs to decode an action arrives in the
    state it is asked to act on.
    """
    return PolicyHost()


def describe_fallback_reason(checkpoint: Path | None) -> str:
    """Human-readable explanation for /health, so an unexpected baseline is self-diagnosing."""
    if checkpoint is None:
        return "NEXUS_POLICY_PATH is unset — running the static baseline by configuration."
    return (
        f"Checkpoint {checkpoint} did not load; see logs. The usual cause is that MDP_VERSION changed "
        f"since it was trained (now {MDP_VERSION})."
    )
