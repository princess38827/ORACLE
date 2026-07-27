"""
tools.py — Built-in tools available to the ORACLE chatbot child agent.

Each tool is a callable that accepts a dict of arguments and returns a dict
containing at least a "result" key.  The ToolRegistry maps tool names to their
implementation so the agent can look them up at runtime.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ToolFn = Callable[[Dict[str, Any]], Dict[str, Any]]


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def tool_calculate(args: Dict[str, Any]) -> Dict[str, Any]:
    """Safely evaluate a simple arithmetic expression."""
    expression: str = str(args.get("expression", "")).strip()
    # Allow only safe characters: digits, operators, parentheses, spaces, dots
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.\%\^]+", expression):
        return {"result": None, "error": "Unsafe expression – only arithmetic operators allowed."}
    try:
        # Replace ^ with ** for exponentiation
        safe_expr = expression.replace("^", "**")
        result = eval(safe_expr, {"__builtins__": {}}, {"math": math})  # noqa: S307
        return {"result": result}
    except Exception as exc:
        return {"result": None, "error": str(exc)}


def tool_datetime(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return the current UTC date/time."""
    now = datetime.now(timezone.utc)
    fmt = args.get("format", "%Y-%m-%d %H:%M:%S UTC")
    return {"result": now.strftime(fmt)}


def tool_summarise(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return the first N sentences of *text* as a summary."""
    text: str = str(args.get("text", ""))
    n: int = int(args.get("sentences", 3))
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(sentences[:n])
    return {"result": summary or text}


def tool_word_count(args: Dict[str, Any]) -> Dict[str, Any]:
    """Count the words in *text*."""
    text: str = str(args.get("text", ""))
    words = text.split()
    return {"result": len(words)}


def tool_echo(args: Dict[str, Any]) -> Dict[str, Any]:
    """Echo back the provided message (useful for testing)."""
    return {"result": args.get("message", "")}


def tool_help(args: Dict[str, Any]) -> Dict[str, Any]:
    """List all registered tools with their descriptions."""
    registry = ToolRegistry.default()
    lines = [f"- **{name}**: {meta['description']}" for name, meta in registry.tools.items()]
    return {"result": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry that maps tool names to implementations and metadata."""

    _default: Optional["ToolRegistry"] = None

    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        fn: ToolFn,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tools[name] = {
            "fn": fn,
            "description": description,
            "parameters": parameters or {},
        }

    def get(self, name: str) -> Optional[ToolFn]:
        entry = self.tools.get(name)
        return entry["fn"] if entry else None

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        fn = self.get(name)
        if fn is None:
            return {"result": None, "error": f"Unknown tool: '{name}'"}
        try:
            return fn(args)
        except Exception as exc:
            return {"result": None, "error": f"Tool '{name}' raised an exception: {exc}"}

    def schema(self) -> List[Dict[str, Any]]:
        """Return an OpenAI-style tool-schema list for the LLM prompt."""
        out = []
        for name, meta in self.tools.items():
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": meta["description"],
                        "parameters": meta["parameters"],
                    },
                }
            )
        return out

    # ------------------------------------------------------------------
    # Default singleton with built-in tools pre-registered
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "ToolRegistry":
        if cls._default is None:
            r = cls()
            r.register(
                "calculate",
                tool_calculate,
                "Evaluate an arithmetic expression (e.g. '2 + 3 * 4').",
                {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression to evaluate.",
                        }
                    },
                    "required": ["expression"],
                },
            )
            r.register(
                "datetime",
                tool_datetime,
                "Return the current UTC date and time.",
                {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": "strftime format string (optional).",
                        }
                    },
                },
            )
            r.register(
                "summarise",
                tool_summarise,
                "Return the first N sentences of the provided text as a summary.",
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to summarise."},
                        "sentences": {
                            "type": "integer",
                            "description": "Number of sentences to return (default 3).",
                        },
                    },
                    "required": ["text"],
                },
            )
            r.register(
                "word_count",
                tool_word_count,
                "Count the number of words in the provided text.",
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to count."}
                    },
                    "required": ["text"],
                },
            )
            r.register(
                "echo",
                tool_echo,
                "Echo back a message (useful for testing connectivity).",
                {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            )
            r.register(
                "help",
                tool_help,
                "List all available tools with their descriptions.",
                {"type": "object", "properties": {}},
            )
            cls._default = r
        return cls._default
