"""
openai_backend.py — OpenAI Chat Completions backend for the ORACLE chatbot.

Usage
-----
    from chatbot import ChatbotAgent
    from chatbot.openai_backend import OpenAIBackend

    agent = ChatbotAgent(backend=OpenAIBackend(model="gpt-4o-mini"))
    print(agent.chat("What is 12 * 8?"))
    agent.shutdown()

The backend maps the internal :class:`Message` format to the OpenAI messages
API, forwards tool schemas as ``tools``, and parses tool-call or text
responses back into :class:`LLMResponse`.

Environment variables
---------------------
``OPENAI_API_KEY``
    API key used when *api_key* is not passed explicitly.
``OPENAI_BASE_URL``
    Optional base URL override (useful for proxies / Azure OpenAI endpoints).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .agent import LLMBackend, LLMResponse, Message

logger = logging.getLogger(__name__)


def _messages_to_oai(messages: List[Message]) -> List[Dict[str, Any]]:
    """
    Convert internal :class:`Message` objects to the OpenAI message format.

    The OpenAI API requires that every ``tool_calls`` entry on an assistant
    message is later echoed back as a ``tool`` message with a matching
    ``tool_call_id``.  Because the internal :class:`Message` dataclass does
    not store an OpenAI-issued ID, we assign deterministic sequential IDs
    (``call_0``, ``call_1``, …) so that the history is always consistent.
    """
    oai_messages: List[Dict[str, Any]] = []
    call_counter = 0

    # First pass: assign a sequential call_id to every assistant tool-call
    # message, then use the same counter to match the following tool message.
    # We iterate once to build the id map, then once to emit the OAI dicts.
    call_ids: List[Optional[str]] = []
    for msg in messages:
        if msg.role == "assistant" and msg.tool_name:
            call_ids.append(f"call_{call_counter}")
            call_counter += 1
        else:
            call_ids.append(None)

    # Track the id of the most-recently-seen assistant tool call so the
    # following tool-result message can reference it.
    pending_call_id: Optional[str] = None

    for msg, call_id in zip(messages, call_ids):
        if msg.role == "system":
            oai_messages.append({"role": "system", "content": msg.content})

        elif msg.role == "user":
            oai_messages.append({"role": "user", "content": msg.content})

        elif msg.role == "assistant":
            if msg.tool_name and msg.tool_args is not None:
                # Tool-call turn
                pending_call_id = call_id
                oai_messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": msg.tool_name,
                                    "arguments": json.dumps(msg.tool_args),
                                },
                            }
                        ],
                    }
                )
            else:
                # Plain assistant reply
                oai_messages.append({"role": "assistant", "content": msg.content})

        elif msg.role == "tool":
            # Tool-result message; must reference the preceding tool_call id.
            result_content = (
                json.dumps(msg.tool_result) if msg.tool_result is not None else msg.content
            )
            oai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": pending_call_id or "call_unknown",
                    "content": result_content,
                }
            )
            pending_call_id = None

    return oai_messages


class OpenAIBackend(LLMBackend):
    """
    LLM backend that uses the OpenAI Chat Completions API.

    Parameters
    ----------
    model:
        OpenAI model identifier (default ``"gpt-4o-mini"``).
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when not provided.
    base_url:
        Optional API base URL (e.g. for Azure OpenAI or local proxies).
        Falls back to the ``OPENAI_BASE_URL`` environment variable.
    temperature:
        Sampling temperature forwarded to the API (default ``0``).
    max_tokens:
        Maximum tokens in the completion (default ``1024``).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import openai  # noqa: F401  (imported lazily to keep the package optional)
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use OpenAIBackend. "
                "Install it with:  pip install openai"
            ) from exc

        import openai as _openai

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL")

        self._client = _openai.OpenAI(
            api_key=resolved_key,
            base_url=resolved_base,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse:
        """
        Call the OpenAI Chat Completions API and return an :class:`LLMResponse`.

        If the model requests a tool call the response will have
        ``tool_call`` populated; otherwise ``text`` is set.
        """
        oai_messages = _messages_to_oai(messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug("OpenAIBackend: sending %d messages to %s", len(oai_messages), self.model)

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish_reason = choice.finish_reason

        if finish_reason == "tool_calls" and choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            logger.debug("OpenAIBackend: tool call '%s' with args %s", tc.function.name, args)
            return LLMResponse(tool_call={"name": tc.function.name, "args": args})

        content = choice.message.content or ""
        logger.debug("OpenAIBackend: text reply (%d chars)", len(content))
        return LLMResponse(text=content)
