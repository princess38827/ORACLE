"""Unit tests for chatbot/tools.py."""

import pytest

from chatbot.tools import ToolRegistry, tool_calculate, tool_datetime, tool_echo, tool_summarise, tool_word_count


# ---------------------------------------------------------------------------
# tool_calculate
# ---------------------------------------------------------------------------

class TestToolCalculate:
    def test_basic_addition(self):
        result = tool_calculate({"expression": "2 + 3"})
        assert result["result"] == 5

    def test_multiplication(self):
        result = tool_calculate({"expression": "12 * 8"})
        assert result["result"] == 96

    def test_division(self):
        result = tool_calculate({"expression": "10 / 4"})
        assert abs(result["result"] - 2.5) < 1e-9

    def test_exponentiation(self):
        result = tool_calculate({"expression": "2^10"})
        assert result["result"] == 1024

    def test_parentheses(self):
        result = tool_calculate({"expression": "(2 + 3) * 4"})
        assert result["result"] == 20

    def test_unsafe_expression_returns_error(self):
        result = tool_calculate({"expression": "__import__('os').getcwd()"})
        assert result["result"] is None
        assert "error" in result

    def test_empty_expression_returns_error(self):
        result = tool_calculate({"expression": ""})
        assert result["result"] is None or result.get("error")


# ---------------------------------------------------------------------------
# tool_datetime
# ---------------------------------------------------------------------------

class TestToolDatetime:
    def test_returns_string(self):
        result = tool_datetime({})
        assert isinstance(result["result"], str)
        assert "UTC" in result["result"]

    def test_custom_format(self):
        result = tool_datetime({"format": "%Y"})
        year = int(result["result"])
        assert 2020 < year < 2100


# ---------------------------------------------------------------------------
# tool_summarise
# ---------------------------------------------------------------------------

class TestToolSummarise:
    _TEXT = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump!"

    def test_default_sentences(self):
        result = tool_summarise({"text": self._TEXT})
        assert result["result"].count(".") + result["result"].count("!") >= 1

    def test_one_sentence(self):
        result = tool_summarise({"text": self._TEXT, "sentences": 1})
        assert "fox" in result["result"]
        assert "liquor" not in result["result"]

    def test_empty_text(self):
        result = tool_summarise({"text": ""})
        assert result["result"] == ""


# ---------------------------------------------------------------------------
# tool_word_count
# ---------------------------------------------------------------------------

class TestToolWordCount:
    def test_simple(self):
        result = tool_word_count({"text": "hello world"})
        assert result["result"] == 2

    def test_empty(self):
        result = tool_word_count({"text": ""})
        assert result["result"] == 0

    def test_multiple_spaces(self):
        result = tool_word_count({"text": "  a  b  c  "})
        assert result["result"] == 3


# ---------------------------------------------------------------------------
# tool_echo
# ---------------------------------------------------------------------------

class TestToolEcho:
    def test_echoes_message(self):
        result = tool_echo({"message": "ping"})
        assert result["result"] == "ping"

    def test_missing_message(self):
        result = tool_echo({})
        assert result["result"] == ""


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_default_has_built_in_tools(self):
        registry = ToolRegistry.default()
        for name in ("calculate", "datetime", "summarise", "word_count", "echo", "help"):
            assert name in registry.tools

    def test_call_unknown_returns_error(self):
        registry = ToolRegistry.default()
        result = registry.call("nonexistent_tool", {})
        assert result["result"] is None
        assert "error" in result

    def test_register_custom_tool(self):
        registry = ToolRegistry()
        registry.register("greet", lambda args: {"result": f"Hello, {args.get('name')}!"}, "Greet someone.")
        result = registry.call("greet", {"name": "Alice"})
        assert result["result"] == "Hello, Alice!"

    def test_schema_returns_list(self):
        registry = ToolRegistry.default()
        schema = registry.schema()
        assert isinstance(schema, list)
        assert all("function" in item for item in schema)
