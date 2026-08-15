"""Loading a trained checkpoint back into something a rollout can run.

This is the piece that lets ``nexus-eval --run-id <id>`` put a *policy* row next to the baseline rows.
Without it the evaluator validated the checkpoint's MDP version, logged its metadata, and then compared
three baselines against each other — so the project's central claim, that the learned controller beats
the non-learned floor, could not be produced by the project's own tool.

Decoding mirrors ``nexus_controller.policy_host`` exactly: the same action space, the same routing read
out of the received state. A policy measured by different decoding logic than it is served with is not
measuring the deployed system.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from nexus_errors.config_error import ConfigurationError
from nexus_mdp.action_space import (
    FlowRoutingOptions,
    decode_continuous_action,
    decode_discrete_action,
)
from nexus_rl.rollout import PolicyPredictor
from nexus_types.action import AllocationAction
from nexus_types.network_state import NetworkState

ActionMode = Literal["discrete", "continuous"]

_ModelPredict = Callable[[NDArray[np.float32]], tuple[NDArray[np.float32], object]]


def routing_from(state: NetworkState) -> tuple[FlowRoutingOptions, ...]:
    """Read the action space out of the state, never out of configuration.

    Flow ids are positional (``f0``..``fn``), so a config-derived flow set lines up perfectly while the
    routes behind the ids belong to different flows — wrong paths, no error anywhere.
    """
    return tuple(
        FlowRoutingOptions(flow_id=flow.flow_id, candidate_paths=flow.candidate_paths) for flow in state.flows
    )


def load_policy_predictor(checkpoint: Path, algorithm: str, action_mode: ActionMode) -> PolicyPredictor:
    """Load an SB3 checkpoint and wrap it as a :data:`PolicyPredictor`.

    ``deterministic=True``: evaluation measures the policy, not its exploration noise. Sampling here
    would make the same checkpoint score differently on every run and turn seed variance into an
    unattributable mess.
    """
    if not checkpoint.exists():
        raise ConfigurationError(
            f"No checkpoint at {checkpoint}. Train the run before evaluating it.",
            {"checkpoint": str(checkpoint)},
        )

    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as error:
        raise ConfigurationError(
            "Evaluating a trained policy needs the optional 'train' extra. Install it with: "
            "uv sync --all-packages --extra train. Baseline-only evaluation works without it.",
            {"missing": str(error)},
        ) from error

    loader = DQN if algorithm == "dqn" else PPO
    model = loader.load(checkpoint, device="cpu")

    def predict(observation: NDArray[np.float32], state: NetworkState) -> AllocationAction:
        action, _ = model.predict(observation, deterministic=True)
        if action_mode == "discrete":
            return decode_discrete_action(int(np.asarray(action).item()), state, routing_from(state))
        return decode_continuous_action(np.asarray(action, dtype=np.float32).ravel(), state)

    return predict
