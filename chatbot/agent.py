"""
agent.py — Core agentic AI chatbot child for ORACLE.

Architecture
------------
ChatbotAgent orchestrates a ReAct-style (Reason → Act → Observe) loop:

  1. The user sends a message.
  2. The agent examines its conversation history and the current message.
  3. If a tool call is needed the agent invokes the tool and appends the
     observation to the context.
  4. The loop repeats (up to `max_iterations`) until the agent produces a
     final answer.
  5. The final answer is returned to the caller and added to history.

The agent is deliberately LLM-backend-agnostic.  You inject an `LLMBackend`
implementation (see the `RuleBasedBackend` stub below for a zero-dependency
example).  Drop in an OpenAI, Anthropic, or local-model backend by
subclassing `LLMBackend`.

ORACLE integration
------------------
The agent can be bound to an `OracleClient` so the overseer can push tasks
directly to the chatbot and receive results back.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .oracle_client import AgentInfo, OracleClient, OracleTask
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str
    tool_name: Optional[str] = None   # set when role == "tool"
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.tool_args is not None:
            d["tool_args"] = self.tool_args
        if self.tool_result is not None:
            d["tool_result"] = self.tool_result
        return d


# ---------------------------------------------------------------------------
# LLM backend interface
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """
    Abstract base class for language-model backends.

    Subclass this to connect any LLM (OpenAI, Anthropic, local, …).
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
    ) -> "LLMResponse":
        """
        Given the conversation `messages` and available `tools`, return an
        `LLMResponse` that is either a final text reply or a tool-call request.
        """


@dataclass
class LLMResponse:
    """Unified response from any LLM backend."""

    text: Optional[str] = None             # final answer (mutually exclusive with tool_call)
    tool_call: Optional[Dict[str, Any]] = None  # {"name": ..., "args": {...}}


# ---------------------------------------------------------------------------
# Rule-based stub backend (zero dependencies – for demo / tests)
# ---------------------------------------------------------------------------

class RuleBasedBackend(LLMBackend):
    """
    A minimal rule-based backend that handles a handful of patterns without
    calling any external service.  Replace with a real LLM backend in production.

    Supported patterns
    ------------------
    - Arithmetic expressions  → calls the ``calculate`` tool
    - "what time" / "date"    → calls the ``datetime`` tool
    - "summarise" / "summary" → calls the ``summarise`` tool
    - "count words"            → calls the ``word_count`` tool
    - "help" / "tools"        → calls the ``help`` tool
    - Everything else          → returns a generic reply
    """

    def complete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse:
        # If the last message is a tool observation, produce a final text answer from it.
        if messages and messages[-1].role == "tool":
            obs = messages[-1]
            result = obs.tool_result or {}
            value = result.get("result", "")
            error = result.get("error")
            if error:
                return LLMResponse(text=f"The tool encountered an error: {error}")
            return LLMResponse(text=str(value))

        # Get the latest user message
        user_text = ""
        for m in reversed(messages):
            if m.role == "user":
                user_text = m.content
                break

        lower = user_text.lower()

        # Detect arithmetic: at least one digit present with optional operators/parens
        if re.search(r"\d", user_text) and re.search(r"[\d\+\-\*\/\^\(\)]+", user_text):
            # Extract the first contiguous arithmetic sub-expression (starts with digit or paren)
            expr = re.search(r"([\(\d][\d\s\+\-\*\/\^\(\)\.]*[\d\)])", user_text)
            if not expr:
                # Single number
                expr = re.search(r"(\d+(?:\.\d+)?)", user_text)
            if expr:
                return LLMResponse(tool_call={"name": "calculate", "args": {"expression": expr.group(1).strip()}})

        if "time" in lower or "date" in lower or "clock" in lower:
            return LLMResponse(tool_call={"name": "datetime", "args": {}})

        if "summarise" in lower or "summarize" in lower or "summary" in lower or "shorten" in lower:
            # Extract the text after the keyword
            match = re.search(r"(?:summarise|summarize|summary of|shorten)[:\s]+(.+)", user_text, re.DOTALL | re.IGNORECASE)
            text = match.group(1).strip() if match else user_text
            return LLMResponse(tool_call={"name": "summarise", "args": {"text": text, "sentences": 2}})

        if "count" in lower and "word" in lower:
            match = re.search(r"(?:count words?(?:\s+in)?)[:\s]+(.+)", user_text, re.DOTALL | re.IGNORECASE)
            text = match.group(1).strip() if match else user_text
            return LLMResponse(tool_call={"name": "word_count", "args": {"text": text}})

        if "help" in lower or "tool" in lower or "what can you do" in lower:
            return LLMResponse(tool_call={"name": "help", "args": {}})

        # Generic fallback reply
        reply = (
            "I'm the ORACLE chatbot child agent. I can help you with calculations, "
            "the current date/time, text summarisation, word counts, and more. "
            "Ask me anything or type 'help' to see all available tools."
        )
        return LLMResponse(text=reply)


# ---------------------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    Rolling conversation memory with a configurable token budget.

    The budget is approximated by character count (1 token ≈ 4 chars) to keep
    this dependency-free.  Swap in a proper tokeniser if needed.
    """

    CHARS_PER_TOKEN = 4

    def __init__(self, max_tokens: int = 4096) -> None:
        self.max_chars = max_tokens * self.CHARS_PER_TOKEN
        self._messages: List[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()

    def get(self) -> List[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        """Drop oldest non-system messages when we exceed the budget."""
        while self._total_chars() > self.max_chars and len(self._messages) > 1:
            # Keep the system prompt (index 0) if present
            remove_idx = 1 if self._messages[0].role == "system" else 0
            if remove_idx < len(self._messages):
                self._messages.pop(remove_idx)

    def _total_chars(self) -> int:
        return sum(len(m.content) for m in self._messages)


# ---------------------------------------------------------------------------
# ChatbotAgent
# ---------------------------------------------------------------------------

class ChatbotAgent:
    """
    Agentic AI chatbot child for the ORACLE overseer.

    Parameters
    ----------
    backend:
        LLM backend to use.  Defaults to ``RuleBasedBackend`` (no external deps).
    tool_registry:
        Tool registry.  Defaults to ``ToolRegistry.default()``.
    system_prompt:
        System instruction prepended to every conversation.
    max_iterations:
        Maximum tool-call iterations per user turn before forcing a reply.
    memory_tokens:
        Rolling context window size in tokens.
    oracle_url:
        Optional ORACLE overseer URL.  Pass ``None`` for standalone operation.
    agent_name:
        Human-readable name for this agent instance.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an agentic AI assistant – a child agent managed by the ORACLE overseer. "
        "You can use tools to answer questions accurately. "
        "Always use a tool when the user's request can be fulfilled by one. "
        "Reply concisely and helpfully."
    )

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 5,
        memory_tokens: int = 4096,
        oracle_url: Optional[str] = None,
        agent_name: str = "chatbot-child",
    ) -> None:
        self.backend = backend or RuleBasedBackend()
        self.registry = tool_registry or ToolRegistry.default()
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.memory = ConversationMemory(max_tokens=memory_tokens)
        self.agent_id = str(uuid.uuid4())

        # Seed the memory with the system prompt
        self.memory.add(Message(role="system", content=self.system_prompt))

        # ORACLE integration
        self._oracle = OracleClient(
            oracle_url=oracle_url,
            agent_info=AgentInfo(
                agent_id=self.agent_id,
                name=agent_name,
            ),
            task_handler=self._handle_oracle_task,
        )
        self._oracle.start()

    # ------------------------------------------------------------------
    # Public chat API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's reply.

        This is the main entry point for interactive use.
        """
        self.memory.add(Message(role="user", content=user_message))

        reply = self._react_loop()

        self.memory.add(Message(role="assistant", content=reply))
        return reply

    def reset(self) -> None:
        """Clear conversation history (keeps the system prompt)."""
        self.memory.clear()
        self.memory.add(Message(role="system", content=self.system_prompt))

    def shutdown(self) -> None:
        """Gracefully shut down the agent and deregister from ORACLE."""
        self._oracle.stop()

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def _react_loop(self) -> str:
        """
        Reason → Act → Observe loop.

        Runs up to `max_iterations` tool calls before forcing a final answer.
        """
        tool_schemas = self.registry.schema()

        for iteration in range(self.max_iterations):
            response = self.backend.complete(self.memory.get(), tool_schemas)

            if response.text is not None:
                # The LLM produced a final answer – done.
                return response.text

            if response.tool_call:
                tool_name = response.tool_call.get("name", "")
                tool_args = response.tool_call.get("args", {})

                logger.debug(
                    "Agent [iter %d]: calling tool '%s' with args %s",
                    iteration + 1,
                    tool_name,
                    tool_args,
                )

                # Record the tool invocation in memory
                self.memory.add(
                    Message(
                        role="assistant",
                        content=f"[tool call: {tool_name}({json.dumps(tool_args)})]",
                        tool_name=tool_name,
                        tool_args=tool_args,
                    )
                )

                # Execute the tool
                tool_result = self.registry.call(tool_name, tool_args)
                observation = self._format_observation(tool_name, tool_result)

                logger.debug("Agent [iter %d]: tool result: %s", iteration + 1, tool_result)

                # Record the observation in memory
                self.memory.add(
                    Message(
                        role="tool",
                        content=observation,
                        tool_name=tool_name,
                        tool_result=tool_result,
                    )
                )

                # Continue the loop so the LLM can use the observation
                continue

            # Neither text nor tool_call – break to avoid infinite loop
            break

        # Fallback if we exhausted iterations or got an unexpected response
        return "I've reached the maximum number of reasoning steps. Please rephrase your question."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_observation(tool_name: str, result: Dict[str, Any]) -> str:
        if "error" in result and result["error"]:
            return f"Tool '{tool_name}' returned an error: {result['error']}"
        value = result.get("result", "")
        return f"Tool '{tool_name}' returned: {value}"

    def _handle_oracle_task(self, task: OracleTask) -> str:
        """Callback invoked by OracleClient when the overseer pushes a task."""
        logger.info("ChatbotAgent: received task %s from ORACLE: %s", task.task_id, task.instruction)
        reply = self.chat(task.instruction)
        self._oracle.report_result(task.task_id, reply)
        return reply
