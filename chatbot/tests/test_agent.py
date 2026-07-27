"""Unit tests for chatbot/agent.py."""

import pytest

from chatbot.agent import (
    ChatbotAgent,
    ConversationMemory,
    LLMResponse,
    Message,
    RuleBasedBackend,
)
from chatbot.tools import ToolRegistry


# ---------------------------------------------------------------------------
# RuleBasedBackend
# ---------------------------------------------------------------------------

class TestRuleBasedBackend:
    def setup_method(self):
        self.backend = RuleBasedBackend()
        self.tools = ToolRegistry.default().schema()

    def _msg(self, text: str) -> list:
        return [Message(role="user", content=text)]

    def test_arithmetic_triggers_calculate(self):
        resp = self.backend.complete(self._msg("What is 5 * 6?"), self.tools)
        assert resp.tool_call is not None
        assert resp.tool_call["name"] == "calculate"

    def test_time_triggers_datetime(self):
        resp = self.backend.complete(self._msg("What time is it?"), self.tools)
        assert resp.tool_call is not None
        assert resp.tool_call["name"] == "datetime"

    def test_help_triggers_help(self):
        resp = self.backend.complete(self._msg("help"), self.tools)
        assert resp.tool_call is not None
        assert resp.tool_call["name"] == "help"

    def test_unknown_returns_text(self):
        resp = self.backend.complete(self._msg("Tell me a joke"), self.tools)
        assert resp.text is not None
        assert resp.tool_call is None


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------

class TestConversationMemory:
    def test_add_and_get(self):
        mem = ConversationMemory(max_tokens=1000)
        mem.add(Message(role="user", content="hello"))
        assert len(mem.get()) == 1

    def test_clear_resets_messages(self):
        mem = ConversationMemory()
        mem.add(Message(role="user", content="hello"))
        mem.clear()
        assert mem.get() == []

    def test_trimming_preserves_system_prompt(self):
        # Very small budget so trimming is triggered
        mem = ConversationMemory(max_tokens=5)  # ~20 chars
        mem.add(Message(role="system", content="sys"))
        for i in range(20):
            mem.add(Message(role="user", content=f"message number {i}"))
        messages = mem.get()
        # System prompt should be retained
        assert messages[0].role == "system"
        # Total char count should not exceed budget
        total_chars = sum(len(m.content) for m in messages)
        assert total_chars <= mem.max_chars or len(messages) == 1


# ---------------------------------------------------------------------------
# ChatbotAgent
# ---------------------------------------------------------------------------

class TestChatbotAgent:
    def setup_method(self):
        self.agent = ChatbotAgent(oracle_url=None)

    def teardown_method(self):
        self.agent.shutdown()

    def test_chat_returns_string(self):
        reply = self.agent.chat("hello")
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_arithmetic_is_answered(self):
        reply = self.agent.chat("42 * 7")
        assert "294" in reply

    def test_datetime_is_answered(self):
        reply = self.agent.chat("What time is it?")
        assert "UTC" in reply

    def test_help_lists_tools(self):
        reply = self.agent.chat("help")
        assert "calculate" in reply.lower() or "tool" in reply.lower()

    def test_reset_clears_history(self):
        self.agent.chat("remember this: banana")
        self.agent.reset()
        messages = self.agent.memory.get()
        # Only the system prompt should remain
        assert len(messages) == 1
        assert messages[0].role == "system"

    def test_multi_turn_conversation(self):
        reply1 = self.agent.chat("What is 10 + 5?")
        assert "15" in reply1
        reply2 = self.agent.chat("And now multiply that by 2")
        assert isinstance(reply2, str)  # agent responds without crashing
