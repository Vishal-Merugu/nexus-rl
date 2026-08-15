"""Application settings — the single reader of environment variables.

Nothing else in the repo calls ``os.environ``. Adding a variable read anywhere else is a review blocker: it
makes configuration untraceable, and it is how secrets end up somewhere they should not be.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from nexus_errors.config_error import ConfigurationError

_REDACTED = "***"
_SECRET_SUFFIXES = ("_key", "_secret", "_token")


class Settings(BaseSettings):
    """Runtime configuration, read once from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="NEXUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    env: str = Field(default="development", description="'development', 'staging', or 'production'.")
    log_level: str = Field(default="INFO")

    seed: int = Field(default=1337, description="Master seed for the simulator service.")
    topology: str = Field(default="mesh-6", description="Named topology from nexus_sim.topology.")

    simulator_url: str = Field(default="http://localhost:8000")
    controller_url: str = Field(default="http://localhost:8001")

    policy_path: Path | None = Field(
        default=None,
        description="Checkpoint for the controller. None falls back to the `static` baseline allocator, "
        "which the controller reports on /health.",
    )

    # --- Agent-layer LLM ---
    # Provider-agnostic on purpose: everything here speaks the OpenAI-compatible Chat Completions API,
    # which Ollama, llama.cpp, LM Studio, vLLM, OpenRouter, Groq, Together, OpenAI and Anthropic all
    # expose. The default is a local Ollama with no key, so the stack runs with no account anywhere.
    # See ADR-0005.
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible endpoint. Local Ollama by default.",
    )
    llm_model: str = Field(
        default="qwen2.5:7b",
        description="Model id as the endpoint names it. Needs tool-calling support for the agent layer.",
    )
    llm_api_key: str = Field(
        default="",
        description="Bearer token for the endpoint. Empty is correct for local Ollama and llama.cpp.",
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        description="Generous by default — a 7B model on CPU is slow, and a timeout mid-explanation is "
        "worse than a wait.",
    )

    api_key: str = Field(
        default="",
        description="Shared key guarding mutating endpoints. Required outside development. Unrelated to "
        "llm_api_key.",
    )

    @field_validator("log_level")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        upper = value.upper()
        if upper not in allowed:
            raise ConfigurationError(f"Unknown log level {value!r}.", {"allowed": sorted(allowed)})
        return upper

    @property
    def is_development(self) -> bool:
        return self.env == "development"

    @property
    def is_local_llm(self) -> bool:
        """Whether the configured endpoint is running on this machine.

        Used to skip the API-key check and to soften timeouts — a local model needs no credential and is
        legitimately slow.
        """
        # Compare the parsed hostname, never a substring: "evil-localhost.example.com" and
        # "127.0.0.1.attacker.net" both contain a loopback string while resolving anywhere at all, which
        # bypassed the remote-endpoint credential guard.
        hostname = urlparse(self.llm_base_url).hostname
        return hostname in {"localhost", "127.0.0.1", "::1", "host.docker.internal"}

    def require_api_key(self) -> str:
        """Return the API key, failing loudly if it is missing outside development.

        Mutating endpoints call this. A deployment that forgot the key should fail at the first request with a
        clear message, not serve an unauthenticated control surface.
        """
        if self.is_development:
            return self.api_key
        if not self.api_key:
            raise ConfigurationError(
                "NEXUS_API_KEY is required when NEXUS_ENV is not 'development' — mutating endpoints would "
                "otherwise be unauthenticated.",
                {"env": self.env},
            )
        return self.api_key

    def redacted_dump(self) -> dict[str, object]:
        """Serialize for a run artifact or a log line, with secrets stripped.

        Run configs are written to disk and read by whoever inspects the experiment later. Anything whose name
        looks like a credential is replaced rather than recorded.
        """
        dumped = self.model_dump(mode="json")
        return {
            key: (_REDACTED if value and key.endswith(_SECRET_SUFFIXES) else value)
            for key, value in dumped.items()
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once.

    Cached deliberately: settings that can change mid-process make two services disagree about the same
    configuration, and the resulting bug looks like anything but a config problem.
    """
    return Settings()
