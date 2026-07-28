"""Unit tests for chatbot/openai_backend.py (fully mocked – no live API calls)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from chatbot.agent import LLMResponse, Message
from chatbot.openai_backend import OpenAIBackend, _messages_to_oai


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_response(content: str):
    """Build a mock openai.ChatCompletion response that returns plain text."""
    choice = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content=content, tool_calls=None),
    )
    return SimpleNamespace(choices=[choice])


def _make_tool_response(name: str, args: dict):
    """Build a mock openai.ChatCompletion response that requests a tool call."""
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=json.dumps(args))
    )
    choice = SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(content=None, tool_calls=[tool_call]),
    )
    return SimpleNamespace(choices=[choice])


# ---------------------------------------------------------------------------
# _messages_to_oai (format conversion)
# ---------------------------------------------------------------------------

class TestMessagesToOai:
    def test_system_message(self):
        msgs = [Message(role="system", content="You are an assistant.")]
        result = _messages_to_oai(msgs)
        assert result == [{"role": "system", "content": "You are an assistant."}]

    def test_user_message(self):
        msgs = [Message(role="user", content="Hello")]
        result = _messages_to_oai(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_plain_message(self):
        msgs = [Message(role="assistant", content="Hi there")]
        result = _messages_to_oai(msgs)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_tool_call_and_result(self):
        msgs = [
            Message(
                role="assistant",
                content="[tool call: calculate({\"expression\": \"2+2\"})]",
                tool_name="calculate",
                tool_args={"expression": "2+2"},
            ),
            Message(
                role="tool",
                content="Tool 'calculate' returned: 4",
                tool_name="calculate",
                tool_result={"result": 4},
            ),
        ]
        result = _messages_to_oai(msgs)

        assert result[0]["role"] == "assistant"
        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "call_0"
        assert tc["function"]["name"] == "calculate"
        assert json.loads(tc["function"]["arguments"]) == {"expression": "2+2"}

        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_0"

    def test_multiple_tool_calls_get_sequential_ids(self):
        msgs = [
            Message(role="assistant", content="[call]", tool_name="datetime", tool_args={}),
            Message(role="tool", content="obs1", tool_name="datetime", tool_result={"result": "now"}),
            Message(role="assistant", content="[call]", tool_name="calculate", tool_args={"expression": "1+1"}),
            Message(role="tool", content="obs2", tool_name="calculate", tool_result={"result": 2}),
        ]
        result = _messages_to_oai(msgs)
        assert result[0]["tool_calls"][0]["id"] == "call_0"
        assert result[1]["tool_call_id"] == "call_0"
        assert result[2]["tool_calls"][0]["id"] == "call_1"
        assert result[3]["tool_call_id"] == "call_1"


# ---------------------------------------------------------------------------
# OpenAIBackend.complete
# ---------------------------------------------------------------------------

@pytest.fixture()
def backend():
    """Return an OpenAIBackend with the openai client fully mocked out.

    We bypass the constructor entirely to avoid any real openai.OpenAI
    instantiation (which would require a valid API key or an installed package
    at fixture-setup time).
    """
    b = OpenAIBackend.__new__(OpenAIBackend)
    b.model = "gpt-4o-mini"
    b.temperature = 0
    b.max_tokens = 1024
    b._client = MagicMock()
    yield b


class TestOpenAIBackendComplete:
    def test_text_response(self, backend):
        backend._client.chat.completions.create.return_value = _make_text_response("Hello!")
        messages = [Message(role="user", content="Hi")]
        result = backend.complete(messages, [])
        assert isinstance(result, LLMResponse)
        assert result.text == "Hello!"
        assert result.tool_call is None

    def test_tool_call_response(self, backend):
        backend._client.chat.completions.create.return_value = _make_tool_response(
            "calculate", {"expression": "3*4"}
        )
        messages = [Message(role="user", content="What is 3*4?")]
        tools = [{"type": "function", "function": {"name": "calculate", "description": "...", "parameters": {}}}]
        result = backend.complete(messages, tools)
        assert result.tool_call is not None
        assert result.tool_call["name"] == "calculate"
        assert result.tool_call["args"] == {"expression": "3*4"}
        assert result.text is None

    def test_empty_content_returns_empty_string(self, backend):
        backend._client.chat.completions.create.return_value = _make_text_response("")
        messages = [Message(role="user", content="?")]
        result = backend.complete(messages, [])
        assert result.text == ""

    def test_tools_forwarded_when_provided(self, backend):
        backend._client.chat.completions.create.return_value = _make_text_response("ok")
        tools = [{"type": "function", "function": {"name": "echo", "description": "echo", "parameters": {}}}]
        messages = [Message(role="user", content="test")]
        backend.complete(messages, tools)
        call_kwargs = backend._client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"

    def test_no_tools_omits_tools_key(self, backend):
        backend._client.chat.completions.create.return_value = _make_text_response("ok")
        messages = [Message(role="user", content="test")]
        backend.complete(messages, [])
        call_kwargs = backend._client.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs

    def test_model_forwarded(self, backend):
        backend._client.chat.completions.create.return_value = _make_text_response("ok")
        backend.complete([Message(role="user", content="x")], [])
        call_kwargs = backend._client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Import error when openai is not installed
# ---------------------------------------------------------------------------

class TestOpenAIBackendImportError:
    def test_raises_import_error_without_openai(self):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="pip install openai"):
                OpenAIBackend(api_key="key")
