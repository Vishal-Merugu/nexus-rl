"""LLM client tests.

The client is provider-agnostic, so these assert on the OpenAI-compatible contract and on the divergences
real local servers actually exhibit — not on any one vendor's behaviour.
"""

from __future__ import annotations

import httpx
import pytest

from nexus_agent.llm import (
    LlmClient,
    build_client_from_settings,
    tool_specs_to_openai_schema,
)
from nexus_errors.config_error import ConfigurationError
from nexus_errors.control_error import AgentError


def _client(handler: object, **overrides: object) -> LlmClient:
    kwargs: dict[str, object] = {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
        "client": httpx.AsyncClient(transport=httpx.MockTransport(handler)),  # type: ignore[arg-type]
    }
    kwargs.update(overrides)
    return LlmClient(**kwargs)  # type: ignore[arg-type]


def _reply(content: str = "n2->n3 is at 94% utilization.") -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return handler


@pytest.mark.anyio
async def test_a_plain_reply_comes_back_as_content() -> None:
    reply = await _client(_reply()).complete([{"role": "user", "content": "status?"}])

    assert "94%" in reply.content
    assert reply.wants_tools is False


@pytest.mark.anyio
async def test_tool_calls_are_returned_not_executed() -> None:
    # The caller runs tools, so a tool failure surfaces as a tool error rather than being swallowed here.
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "get_flow_detail",
                                        "arguments": '{"flow_id": "f3"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    reply = await _client(handler).complete([{"role": "user", "content": "why is f3 slow?"}])

    assert reply.wants_tools
    assert reply.tool_calls[0].name == "get_flow_detail"
    assert reply.tool_calls[0].arguments == {"flow_id": "f3"}


@pytest.mark.anyio
async def test_tool_arguments_sent_as_an_object_are_accepted() -> None:
    # The spec says a JSON string; some local servers send an object. Both are real, so both work.
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "c1", "function": {"name": "get_network_state", "arguments": {}}}
                            ]
                        }
                    }
                ]
            },
        )

    reply = await _client(handler).complete([])

    assert reply.tool_calls[0].arguments == {}


@pytest.mark.anyio
async def test_malformed_tool_arguments_name_the_likely_cause() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"tool_calls": [{"function": {"name": "x", "arguments": "{not json"}}]}}
                ]
            },
        )

    with pytest.raises(AgentError, match="tool-calling support"):
        await _client(handler).complete([])


@pytest.mark.anyio
async def test_a_missing_ollama_model_says_how_to_pull_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(AgentError, match="ollama pull"):
        await _client(handler).complete([])


@pytest.mark.anyio
async def test_an_unreachable_endpoint_says_to_check_ollama_is_running() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(AgentError, match="ollama serve"):
        await _client(handler).complete([])


@pytest.mark.anyio
async def test_rate_limiting_suggests_falling_back_to_local() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(429, json={"error": "slow down"})

    with pytest.raises(AgentError, match="local Ollama"):
        await _client(handler).complete([])


@pytest.mark.anyio
async def test_a_rejected_credential_names_the_variable_to_set() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, json={"error": "unauthorized"})

    with pytest.raises(AgentError, match="NEXUS_LLM_API_KEY"):
        await _client(handler).complete([])


@pytest.mark.anyio
async def test_an_empty_choices_array_is_an_error_not_an_empty_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(AgentError, match="no choices"):
        await _client(handler).complete([])


@pytest.mark.anyio
async def test_no_authorization_header_is_sent_when_no_key_is_configured() -> None:
    # Local servers reject an empty bearer token, so the header must be absent rather than blank.
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    await _client(handler, api_key="").complete([])

    assert seen["auth"] is None


def test_a_remote_endpoint_without_a_key_fails_early_with_the_fix() -> None:
    with pytest.raises(ConfigurationError, match="localhost:11434"):
        build_client_from_settings(
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            api_key="",
            temperature=0.0,
            timeout_seconds=60.0,
            is_local=False,
        )


def test_a_local_endpoint_needs_no_key() -> None:
    client = build_client_from_settings(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b",
        api_key="",
        temperature=0.0,
        timeout_seconds=60.0,
        is_local=True,
    )

    assert client.model == "qwen2.5:7b"


def test_tool_specs_convert_to_the_openai_tools_shape() -> None:
    schema = tool_specs_to_openai_schema(
        (("get_network_state", "  Current state.  ", {"type": "object", "properties": {}}),)
    )

    assert schema[0]["type"] == "function"
    assert schema[0]["function"]["name"] == "get_network_state"  # type: ignore[index]
    assert schema[0]["function"]["description"] == "Current state."  # type: ignore[index]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("message is null", {"choices": [{"message": None}]}),
        ("message is a string", {"choices": [{"message": "hello"}]}),
        ("choice is not an object", {"choices": ["hello"]}),
        ("tool_calls is a dict", {"choices": [{"message": {"tool_calls": {"a": 1}}}]}),
        ("tool call is a string", {"choices": [{"message": {"tool_calls": ["nope"]}}]}),
        ("tool call has no function", {"choices": [{"message": {"tool_calls": [{"id": "1"}]}}]}),
    ],
)
async def test_hostile_response_shapes_raise_agent_errors_not_attribute_errors(
    label: str, body: dict[str, object]
) -> None:
    # Self-hosted servers deviate from the spec in small ways. A leaked AttributeError would surface as an
    # unhandled 500 rather than something the agent can report honestly to the operator.
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=body)

    with pytest.raises(AgentError):
        await _client(handler).complete([])
