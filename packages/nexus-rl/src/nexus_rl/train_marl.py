"""``nexus-train-marl`` — train MAPPO over network slices from a config file.

Hyperparameters live in the config, never on the command line: a run whose settings existed only in
somebody's shell history is not reproducible, and an irreproducible run is not a result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from nexus_errors.config_error import ConfigurationError
from nexus_mdp.version import MDP_VERSION
from nexus_rl.artifacts import RUNS_ROOT, current_git_sha, new_run_id, run_dir
from nexus_rl.mappo import MappoConfig, train_mappo
from nexus_rl.marl_env import MultiAgentNetworkEnv
from nexus_telemetry.client import get_telemetry

_telemetry = get_telemetry("nexus.train.marl")

_ENV_KEYS = {
    "topology",
    "traffic_pattern",
    "flow_count",
    "episode_length",
    "max_flows_per_agent",
    "max_links_per_agent",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO over network slices.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    args = parser.parse_args()

    if not args.config.exists():
        raise ConfigurationError(f"No config at {args.config}.", {"path": str(args.config)})
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Config at {args.config} is not a mapping.", {"path": str(args.config)})

    name = str(raw.get("name", "mappo"))
    seed = int(raw.get("seed", 0))
    env_kwargs = {key: raw[key] for key in _ENV_KEYS if key in raw}
    mappo_fields = set(MappoConfig.__dataclass_fields__)
    unknown = set(raw) - _ENV_KEYS - mappo_fields - {"name"}
    if unknown:
        # A silently ignored key is how a run ends up not doing what its config says.
        raise ConfigurationError(f"Unknown config keys: {sorted(unknown)}.", {"path": str(args.config)})

    env = MultiAgentNetworkEnv(seed=seed, **env_kwargs)
    config = MappoConfig(**{key: raw[key] for key in mappo_fields if key in raw})

    run_id = new_run_id(name)
    directory = run_dir(run_id, args.runs_root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(
            {**raw, "git_sha": current_git_sha(), "mdp_version": MDP_VERSION, "algorithm": "mappo"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    _telemetry.info("MAPPO training started", {"run_id": run_id, "agents": env.agent_count})
    progress = train_mappo(env, config, checkpoint_path=directory / "policy.pt")
    _telemetry.info(
        "MAPPO training complete",
        {
            "run_id": run_id,
            "episodes": len(progress.episode_rewards),
            "mean_episode_reward": progress.mean_episode_reward,
            "final_components": progress.component_means[-1] if progress.component_means else {},
        },
    )
    summary = (
        f"{run_id}: {len(progress.episode_rewards)} episodes, mean reward {progress.mean_episode_reward:.2f}"
    )
    print(summary)  # noqa: T201  — the CLI's output is the deliverable


if __name__ == "__main__":
    main()
