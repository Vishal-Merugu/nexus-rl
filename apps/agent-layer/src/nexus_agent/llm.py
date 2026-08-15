"""LLM client — provider-agnostic by construction.

Everything here talks to an **OpenAI-compatible Chat Completions endpoint**, because that interface is the
one thing every option speaks: Ollama, llama.cpp, LM Studio, vLLM, OpenRouter, Groq, Together, DeepInfra,
OpenAI itself, and Anthropic through its compatibility endpoint. Switching providers is a change to two
environment variables, not a code change.

The default is a **local Ollama with no API key**, so the stack runs end to end with no account anywhere.
See ADR-0005.

Deliberately hand-rolled over the ``openai`` SDK: the surface used here is one POST with a ``tools`` array,
and a thin client keeps the agent-layer image small and the failure modes visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import httpx

from nexus_errors.base_error import error_to_string
from nexus_errors.config_error import ConfigurationError
from nexus_errors.control_error import AgentError
from nexus_telemetry.client import get_telemetry

_telemetry = get_telemetry("nexus.agent.llm")

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation the model asked for."""

    call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class LlmReply:
    """One assistant turn: prose, tool calls, or both."""

    content: str
    tool_calls: tuple[ToolCall, ...]

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LlmClient:
    """Chat client for any OpenAI-compatible endpoint.

    Args:
        base_url: Endpoint root, including ``/v1``. Local Ollama is ``http://localhost:11434/v1``.
        model: Model id as that endpoint names it.
        api_key: Bearer token. Empty is correct for local servers, which is the default case.
        temperature: 0.0 by default — this agent reports network facts, and creativity is a defect here.
        timeout_seconds: Generous, because a 7B model on CPU is genuinely slow.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> LlmReply:
        """One chat completion, optionally offering tools.

        Tool calls are returned rather than executed — the caller runs them, so tool failures surface as
        tool errors instead of being swallowed inside the client.
        """
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as error:
            raise AgentError(
                _explain_status(error, self._base_url, self._model),
                {"status": error.response.status_code, "base_url": self._base_url},
            ) from error
        except httpx.ConnectError as error:
            raise AgentError(
                f"Could not reach the LLM endpoint at {self._base_url}. If you are using Ollama, check it "
                f"is running (`ollama serve`) and the model is pulled (`ollama pull {self._model}`).",
                {"base_url": self._base_url},
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise AgentError(
                f"LLM request failed: {error_to_string(error)}", {"base_url": self._base_url}
            ) from error

        return _parse_reply(body)


def _explain_status(error: httpx.HTTPStatusError, base_url: str, model: str) -> str:
    """Turn a status code into something actionable.

    Each of these is a real 20 minutes lost the first time it happens with a self-hosted endpoint.
    """
    status = error.response.status_code
    if status == 401:
        return (
            f"The endpoint at {base_url} rejected the credential. Set NEXUS_LLM_API_KEY, or point "
            f"NEXUS_LLM_BASE_URL at a local server that needs none."
        )
    if status == 404:
        return (
            f"Model {model!r} was not found at {base_url}. For Ollama, pull it first: "
            f"`ollama pull {model}`. Check NEXUS_LLM_MODEL matches the id the endpoint uses."
        )
    if status == 429:
        return (
            f"Rate limited by {base_url}. Free tiers throttle aggressively — wait, or switch "
            f"NEXUS_LLM_BASE_URL to a local Ollama."
        )
    return f"LLM endpoint returned {status}: {error.response.text[:200]}"


def _parse_reply(body: dict[str, object]) -> LlmReply:
    """Parse a Chat Completions response.

    Tolerant of the small divergences between implementations — Ollama and llama.cpp are close to the spec
    but not identical, and a strict parser would reject working responses.
    """
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AgentError("LLM response contained no choices.", {"body_keys": sorted(body)})

    # Every shape below is checked rather than assumed. Self-hosted servers deviate from the spec in small
    # ways, and an AttributeError escaping here would surface as an unhandled 500 instead of a tool error
    # the agent can report honestly.
    first = choices[0]
    if not isinstance(first, dict):
        raise AgentError("LLM choice was not an object.", {"got": type(first).__name__})

    message = first.get("message")
    if not isinstance(message, dict):
        # Includes the null case. Coercing a missing message to an empty reply would hand the operator
        # silence dressed up as an answer — the empty-success failure this project rejects everywhere else.
        raise AgentError("LLM response contained no message object.", {"got": type(message).__name__})

    content = message.get("content") or ""

    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise AgentError("LLM tool_calls was not a list.", {"got": type(raw_calls).__name__})

    tool_calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            raise AgentError("LLM tool call was not an object.", {"got": type(raw).__name__})
        function = raw.get("function")
        if not isinstance(function, dict):
            raise AgentError("LLM tool call had no function object.", {"got": type(function).__name__})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            # The spec says a JSON string; some local servers send an object. Accept both.
            import json

            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as error:
                raise AgentError(
                    "The model produced tool arguments that are not valid JSON. Smaller local models do "
                    "this — try a model with stronger tool-calling support.",
                    {"raw": str(function.get("arguments"))[:200]},
                ) from error
        tool_calls.append(
            ToolCall(
                call_id=str(raw.get("id") or f"call_{index}"),
                name=str(function.get("name", "")),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )

    return LlmReply(content=str(content), tool_calls=tuple(tool_calls))


def build_client_from_settings(
    base_url: str,
    model: str,
    api_key: str,
    temperature: float,
    timeout_seconds: float,
    is_local: bool,
) -> LlmClient:
    """Construct the client, failing early on a remote endpoint with no credential."""
    if not is_local and not api_key:
        raise ConfigurationError(
            f"NEXUS_LLM_BASE_URL points at a remote endpoint ({base_url}) but NEXUS_LLM_API_KEY is empty. "
            f"Set a key, or use a local endpoint such as http://localhost:11434/v1.",
            {"base_url": base_url},
        )

    _telemetry.info(
        "LLM configured",
        {"base_url": base_url, "model": model, "local": is_local, "keyed": bool(api_key)},
    )
    return LlmClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )


def tool_specs_to_openai_schema(
    specs: tuple[tuple[str, str, dict[str, object]], ...],
) -> list[dict[str, object]]:
    """Convert ``(name, description, input_schema)`` triples into the OpenAI ``tools`` array shape.

    Kept here rather than in the MCP server so the tool registry stays transport-neutral: the same specs
    feed both the MCP surface and a direct chat-completions loop.
    """
    return [
        {
            "type": "function",
            "function": {"name": name, "description": description.strip(), "parameters": schema},
        }
        for name, description, schema in specs
    ]
