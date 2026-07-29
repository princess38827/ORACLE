"""
openai_backend.py — OpenAI chat-completions backend for the ORACLE chatbot.

Usage
-----
    from chatbot import ChatbotAgent, OpenAIBackend

    agent = ChatbotAgent(
        backend=OpenAIBackend(model="gpt-4o", api_key="sk-..."),
    )
    print(agent.chat("What is 12 * 8?"))
    agent.shutdown()

The ``openai`` package must be installed::

    pip install openai

The backend maps ORACLE's internal ``Message`` list to the OpenAI
``messages`` format, passes the tool schema produced by ``ToolRegistry``
(which already uses the OpenAI function-calling shape) unchanged, and
translates the response back into an ``LLMResponse``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from chatbot.agent import LLMBackend, LLMResponse, Message


class OpenAIBackend(LLMBackend):
    """
    LLM backend that delegates to the OpenAI chat-completions API.

    Parameters
    ----------
    model:
        OpenAI model name (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``).
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when not supplied.
    temperature:
        Sampling temperature passed to the API (default ``0.7``).
    max_tokens:
        Maximum number of completion tokens (default ``1024``).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use OpenAIBackend. "
                "Install it with: pip install openai"
            ) from exc

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = openai.OpenAI(api_key=resolved_key)

    # ------------------------------------------------------------------
    # LLMBackend interface
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse:
        """
        Call the OpenAI chat-completions endpoint and return an
        ``LLMResponse`` with either a text reply or a tool-call request.
        """
        oai_messages = [_to_oai_message(m) for m in messages]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        # Tool-call path
        if message.tool_calls:
            tc = message.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            return LLMResponse(tool_call={"name": tc.function.name, "args": args})

        # Text path
        return LLMResponse(text=message.content or "")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_oai_message(msg: Message) -> Dict[str, Any]:
    """Convert an ORACLE ``Message`` to the OpenAI messages-list format."""
    if msg.role == "tool":
        # OpenAI expects tool results as role="tool" with a tool_call_id.
        # Since ORACLE's Message doesn't store tool_call_id we use the
        # tool name as a stable stand-in.
        return {
            "role": "tool",
            "tool_call_id": msg.tool_name or "unknown",
            "content": msg.content,
        }
    return {"role": msg.role, "content": msg.content}
