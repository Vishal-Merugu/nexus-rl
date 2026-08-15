"""Run configuration.

Hyperparameters live in config files, never on the command line. A run whose settings existed only in
somebody's shell history is not reproducible, and an irreproducible run is not a result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from nexus_errors.config_error import ConfigurationError

Algorithm = Literal["dqn", "ppo"]

type HyperparameterValue = float | int | bool | str
"""What a hyperparameter can actually be.

Not ``float``: ``buffer_size``, ``batch_size``, ``n_steps`` and ``n_epochs`` are ints in every config and
in Stable-Baselines3's own signatures, and ``device`` is a string. Declaring them all ``float`` implied a
coercion that never happens — dataclasses do not convert — so the annotation simply disagreed with both
the YAML and the library."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything needed to reproduce a training run.

    Together with the recorded git SHA and ``MDP_VERSION``, this fully determines the run — that is the
    property the whole experiment-hygiene discipline rests on.
    """

    name: str
    algorithm: Algorithm
    seed: int
    total_timesteps: int

    topology: str = "mesh-6"
    traffic_pattern: str = "diurnal"
    flow_count: int = 8
    episode_length: int = 512
    action_mode: Literal["discrete", "continuous"] = "discrete"

    hyperparameters: dict[str, HyperparameterValue] = field(default_factory=dict)
    """Algorithm hyperparameters passed straight to Stable-Baselines3."""

    evaluation_patterns: tuple[str, ...] = ("diurnal", "flash_crowd")
    """Patterns to evaluate on. At least one must be absent from training — otherwise the run cannot
    demonstrate generalisation, only memorisation."""

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ConfigurationError("total_timesteps must be positive.", {"name": self.name})
        if self.episode_length <= 0:
            raise ConfigurationError(
                "episode_length must be positive.",
                {"name": self.name, "episode_length": self.episode_length},
            )
        if self.flow_count <= 0:
            raise ConfigurationError(
                "flow_count must be positive; a network with no flows has nothing to allocate.",
                {"name": self.name, "flow_count": self.flow_count},
            )
        if self.algorithm == "dqn" and self.action_mode == "continuous":
            raise ConfigurationError(
                "DQN cannot handle a continuous action space. Use PPO, or switch action_mode to "
                "'discrete'. See ADR-0004.",
                {"name": self.name},
            )
        # Checked unconditionally. The previous version returned early whenever the training pattern was
        # absent from the list, which let an *empty* evaluation set — evaluating on nothing at all — pass.
        if len(self.evaluation_patterns) < 2:
            raise ConfigurationError(
                "Evaluate on at least two patterns, one of them not trained on. A policy tested only "
                "where it trained has demonstrated memorisation, not competence.",
                {"name": self.name, "evaluation_patterns": list(self.evaluation_patterns)},
            )
        if not self.held_out_patterns:
            raise ConfigurationError(
                "Every evaluation pattern is the training pattern, so generalisation cannot be measured. "
                "Hold at least one pattern out.",
                {
                    "name": self.name,
                    "traffic_pattern": self.traffic_pattern,
                    "evaluation_patterns": list(self.evaluation_patterns),
                },
            )

    @property
    def held_out_patterns(self) -> tuple[str, ...]:
        """Evaluation patterns the policy was not trained on. Where generalisation is actually measured."""
        return tuple(p for p in self.evaluation_patterns if p != self.traffic_pattern)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_run_config(path: Path) -> RunConfig:
    """Load a run config from YAML."""
    if not path.exists():
        raise ConfigurationError(f"No config at {path}.", {"path": str(path)})

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Config at {path} is not a mapping.", {"path": str(path)})

    known = set(RunConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        # A silently ignored key is how a run ends up not doing what its config says.
        raise ConfigurationError(
            f"Unknown config keys: {sorted(unknown)}.", {"path": str(path), "known": sorted(known)}
        )

    if "evaluation_patterns" in raw:
        raw["evaluation_patterns"] = tuple(raw["evaluation_patterns"])
    return RunConfig(**raw)


def write_run_config(config: RunConfig, run_dir: Path, git_sha: str, mdp_version: int) -> Path:
    """Snapshot a config into a run directory.

    The snapshot carries the git SHA and ``MDP_VERSION`` alongside the settings. Without those, a checkpoint
    cannot be checked for validity later and its numbers cannot be trusted.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.yaml"
    payload: dict[str, object] = {
        **config.to_dict(),
        "git_sha": git_sha,
        "mdp_version": mdp_version,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path
