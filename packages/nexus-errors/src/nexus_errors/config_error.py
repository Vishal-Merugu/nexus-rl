"""Errors raised while loading configuration or run artifacts."""

from __future__ import annotations

from nexus_errors.base_error import NexusError


class ConfigurationError(NexusError):
    """Configuration is missing, malformed, or internally inconsistent."""

    code = "configuration_error"


class MdpVersionMismatchError(ConfigurationError):
    """An artifact was produced under a different MDP version than the one now loaded.

    A checkpoint or a recorded metric only means something relative to the observation/action/reward contract
    it was produced under. Comparing across versions silently produces wrong numbers, so it is refused.
    """

    code = "mdp_version_mismatch"

    def __init__(self, expected: int, found: int, artifact: str) -> None:
        super().__init__(
            f"Artifact {artifact} was produced under MDP_VERSION {found}, but the current environment is "
            f"version {expected}. Retrain, or check out the commit recorded in the run config.",
            {"expected": expected, "found": found, "artifact": artifact},
        )
