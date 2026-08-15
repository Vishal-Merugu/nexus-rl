from __future__ import annotations

import pytest

from nexus_config.settings import Settings
from nexus_errors.config_error import ConfigurationError


def test_defaults_allow_the_stack_to_start_with_no_env() -> None:
    settings = Settings(_env_file=None)

    assert settings.env == "development"
    assert settings.seed == 1337
    assert settings.policy_path is None  # controller falls back to the static baseline


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        Settings(_env_file=None, log_level="CHATTY")


def test_log_level_is_normalised_to_upper_case() -> None:
    assert Settings(_env_file=None, log_level="debug").log_level == "DEBUG"


def test_api_key_is_optional_in_development() -> None:
    assert Settings(_env_file=None, env="development").require_api_key() == ""


def test_missing_api_key_outside_development_fails_loudly() -> None:
    settings = Settings(_env_file=None, env="production", api_key="")

    with pytest.raises(ConfigurationError, match="NEXUS_API_KEY is required"):
        settings.require_api_key()


def test_secrets_are_redacted_in_the_run_artifact_dump() -> None:
    dumped = Settings(_env_file=None, api_key="super-secret-value").redacted_dump()

    assert dumped["api_key"] == "***"
    assert dumped["seed"] == 1337


def test_empty_secret_is_left_empty_rather_than_redacted() -> None:
    assert Settings(_env_file=None, api_key="").redacted_dump()["api_key"] == ""


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:11434/v1", True),
        ("http://host.docker.internal:11434/v1", True),
        # Substring matching classified both of these as local, bypassing the remote-credential guard
        # for a host an attacker controls.
        ("https://evil-localhost.example.com/v1", False),
        ("https://127.0.0.1.attacker.net/v1", False),
        ("https://api.groq.com/openai/v1", False),
    ],
)
def test_locality_is_decided_by_hostname_not_substring(url: str, expected: bool) -> None:
    assert Settings(_env_file=None, llm_base_url=url).is_local_llm is expected
