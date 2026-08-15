"""Training entry point — ``nexus-train``.

Stable-Baselines3 and torch are imported lazily. Only this path needs them, which keeps the simulator,
controller and agent-layer images from carrying a torch install they never use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nexus_env.environment import NexusNetworkEnv
from nexus_errors.config_error import ConfigurationError
from nexus_mdp.version import MDP_VERSION
from nexus_rl.artifacts import current_git_sha, new_run_id, run_dir
from nexus_rl.run_config import RunConfig, load_run_config, write_run_config
from nexus_telemetry.client import get_telemetry

_telemetry = get_telemetry("nexus.train")


def build_env(config: RunConfig, seed_offset: int = 0) -> NexusNetworkEnv:
    """Construct the training environment from a run config."""
    return NexusNetworkEnv(
        topology_name=config.topology,
        traffic_pattern_name=config.traffic_pattern,
        seed=config.seed + seed_offset,
        action_mode=config.action_mode,
        flow_count=config.flow_count,
        episode_length=config.episode_length,
    )


def train(config: RunConfig, run_id: str, runs_root: Path) -> Path:
    """Train a policy and write its artifacts.

    Returns the run directory. The checkpoint lands at ``<run_dir>/policy.zip``.
    """
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as error:
        raise ConfigurationError(
            "Training needs the optional 'train' extra. Install it with: "
            "uv sync --all-packages --extra train",
            {"missing": str(error)},
        ) from error

    directory = run_dir(run_id, runs_root)
    write_run_config(config, directory, git_sha=current_git_sha(), mdp_version=MDP_VERSION)

    env = build_env(config)
    algorithm = DQN if config.algorithm == "dqn" else PPO
    model = algorithm(
        "MlpPolicy",
        env,
        seed=config.seed,
        verbose=0,
        tensorboard_log=None,
        # The hyperparameter set is dynamic by construction: it differs per algorithm and arrives from
        # YAML, so no static signature can describe this splat. Stable-Baselines3 validates it at
        # construction and raises on an unknown or ill-typed keyword, which is where the check belongs.
        **config.hyperparameters,  # type: ignore[arg-type]  # dynamic per-algorithm config splat
    )

    _telemetry.info(
        "Training started",
        {
            "run_id": run_id,
            "algorithm": config.algorithm,
            "total_timesteps": config.total_timesteps,
            "mdp_version": MDP_VERSION,
            "traffic_pattern": config.traffic_pattern,
            "held_out_patterns": list(config.held_out_patterns),
        },
    )

    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
    model.save(directory / "policy")

    _telemetry.info("Training finished", {"run_id": run_id, "run_dir": str(directory)})
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a NEXUS-RL policy. Hyperparameters come from the config file, never the CLI."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to a run config YAML.")
    parser.add_argument(
        "--run-id", default=None, help="Override the generated run id. Useful for smoke runs."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override the budget for a smoke run only. Any other override belongs in the config.",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("experiments/runs"))
    args = parser.parse_args()

    config = load_run_config(args.config)
    if args.total_timesteps is not None:
        config = RunConfig(**{**config.to_dict(), "total_timesteps": args.total_timesteps})

    run_id = args.run_id or new_run_id(config.name)
    directory = train(config, run_id, args.runs_root)

    _telemetry.info(
        "Run complete. Evaluate it before believing anything about it.",
        {"run_id": run_id, "next": f"uv run nexus-eval --run-id {run_id}", "run_dir": str(directory)},
    )


if __name__ == "__main__":  # supports `python -m nexus_rl.train` as well as the console script
    main()
