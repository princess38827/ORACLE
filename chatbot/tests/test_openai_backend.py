"""Unit tests for chatbot/backends/openai_backend.py."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from chatbot.agent import Message
from chatbot.backends.openai_backend import OpenAIBackend, _to_oai_message
from chatbot.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_response(content: str):
    """Build a mock openai.ChatCompletion response that returns plain text."""
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_tool_call_response(tool_name: str, tool_args: dict):
    """Build a mock openai.ChatCompletion response that requests a tool call."""
    func = SimpleNamespace(name=tool_name, arguments=json.dumps(tool_args))
    tc = SimpleNamespace(function=func)
    message = SimpleNamespace(content=None, tool_calls=[tc])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_backend() -> tuple[OpenAIBackend, MagicMock]:
    """Return an OpenAIBackend with a patched openai.OpenAI client."""
    with patch("chatbot.backends.openai_backend.OpenAIBackend.__init__", return_value=None):
        backend = OpenAIBackend.__new__(OpenAIBackend)

    mock_client = MagicMock()
    backend._client = mock_client
    backend.model = "gpt-4o-mini"
    backend.temperature = 0.7
    backend.max_tokens = 1024
    return backend, mock_client


TOOLS = ToolRegistry.default().schema()


# ---------------------------------------------------------------------------
# _to_oai_message
# ---------------------------------------------------------------------------

class TestToOaiMessage:
    def test_user_message(self):
        msg = Message(role="user", content="hello")
        result = _to_oai_message(msg)
        assert result == {"role": "user", "content": "hello"}

    def test_assistant_message(self):
        msg = Message(role="assistant", content="Hi there!")
        result = _to_oai_message(msg)
        assert result == {"role": "assistant", "content": "Hi there!"}

    def test_system_message(self):
        msg = Message(role="system", content="You are helpful.")
        result = _to_oai_message(msg)
        assert result == {"role": "system", "content": "You are helpful."}

    def test_tool_message_uses_tool_name_as_call_id(self):
        msg = Message(role="tool", content="42", tool_name="calculate")
        result = _to_oai_message(msg)
        assert result["role"] == "tool"
        assert result["content"] == "42"
        assert result["tool_call_id"] == "calculate"

    def test_tool_message_without_name_uses_unknown(self):
        msg = Message(role="tool", content="ok", tool_name=None)
        result = _to_oai_message(msg)
        assert result["tool_call_id"] == "unknown"


# ---------------------------------------------------------------------------
# OpenAIBackend.complete — text path
# ---------------------------------------------------------------------------

class TestOpenAIBackendTextResponse:
    def setup_method(self):
        self.backend, self.mock_client = _make_backend()

    def test_returns_llm_response_with_text(self):
        self.mock_client.chat.completions.create.return_value = _make_text_response("Hello!")
        msgs = [Message(role="user", content="Hi")]
        resp = self.backend.complete(msgs, TOOLS)
        assert resp.text == "Hello!"
        assert resp.tool_call is None

    def test_api_called_with_correct_model(self):
        self.mock_client.chat.completions.create.return_value = _make_text_response("ok")
        self.backend.complete([Message(role="user", content="hi")], TOOLS)
        call_kwargs = self.mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_api_called_with_tools(self):
        self.mock_client.chat.completions.create.return_value = _make_text_response("ok")
        self.backend.complete([Message(role="user", content="hi")], TOOLS)
        call_kwargs = self.mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == TOOLS
        assert call_kwargs["tool_choice"] == "auto"

    def test_no_tools_omits_tool_fields(self):
        self.mock_client.chat.completions.create.return_value = _make_text_response("ok")
        self.backend.complete([Message(role="user", content="hi")], [])
        call_kwargs = self.mock_client.chat.completions.create.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    def test_empty_content_returns_empty_string(self):
        self.mock_client.chat.completions.create.return_value = _make_text_response("")
        resp = self.backend.complete([Message(role="user", content="hi")], [])
        assert resp.text == ""
        assert resp.tool_call is None


# ---------------------------------------------------------------------------
# OpenAIBackend.complete — tool-call path
# ---------------------------------------------------------------------------

class TestOpenAIBackendToolCallResponse:
    def setup_method(self):
        self.backend, self.mock_client = _make_backend()

    def test_returns_llm_response_with_tool_call(self):
        self.mock_client.chat.completions.create.return_value = _make_tool_call_response(
            "calculate", {"expression": "2 + 3"}
        )
        resp = self.backend.complete([Message(role="user", content="2+3")], TOOLS)
        assert resp.tool_call is not None
        assert resp.text is None
        assert resp.tool_call["name"] == "calculate"
        assert resp.tool_call["args"] == {"expression": "2 + 3"}

    def test_tool_call_with_empty_args(self):
        self.mock_client.chat.completions.create.return_value = _make_tool_call_response(
            "datetime", {}
        )
        resp = self.backend.complete([Message(role="user", content="time?")], TOOLS)
        assert resp.tool_call["name"] == "datetime"
        assert resp.tool_call["args"] == {}

    def test_tool_call_with_invalid_json_args_falls_back_to_empty_dict(self):
        func = SimpleNamespace(name="calculate", arguments="NOT JSON")
        tc = SimpleNamespace(function=func)
        message = SimpleNamespace(content=None, tool_calls=[tc])
        choice = SimpleNamespace(message=message)
        self.mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])

        resp = self.backend.complete([Message(role="user", content="?")], TOOLS)
        assert resp.tool_call["name"] == "calculate"
        assert resp.tool_call["args"] == {}


# ---------------------------------------------------------------------------
# OpenAIBackend — message conversion
# ---------------------------------------------------------------------------

class TestOpenAIBackendMessageConversion:
    def setup_method(self):
        self.backend, self.mock_client = _make_backend()
        self.mock_client.chat.completions.create.return_value = _make_text_response("ok")

    def test_all_message_roles_are_forwarded(self):
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
            Message(role="tool", content="42", tool_name="calculate"),
        ]
        self.backend.complete(messages, [])
        sent = self.mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert len(sent) == 4
        assert sent[0]["role"] == "system"
        assert sent[1]["role"] == "user"
        assert sent[2]["role"] == "assistant"
        assert sent[3]["role"] == "tool"


# ---------------------------------------------------------------------------
# OpenAIBackend — ImportError when openai is missing
# ---------------------------------------------------------------------------

class TestOpenAIBackendImportError:
    def test_raises_import_error_when_openai_missing(self):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="pip install openai"):
                OpenAIBackend()
