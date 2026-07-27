# Chatbot Child Agent

An agentic AI chatbot that runs as a **child agent** managed by the [ORACLE](../README.md) overseer.

## Architecture

```
ORACLE overseer
     │
     │  register / heartbeat / task dispatch / result
     ▼
ChatbotAgent  (ReAct loop)
     │
     ├── LLMBackend  ← swap in OpenAI, Anthropic, local model, …
     │
     ├── ToolRegistry
     │       ├── calculate   – arithmetic evaluation
     │       ├── datetime    – current UTC time
     │       ├── summarise   – extract leading sentences
     │       ├── word_count  – count words in text
     │       ├── echo        – connectivity test
     │       └── help        – list all tools
     │
     ├── ConversationMemory  (rolling context window)
     │
     └── OracleClient  (HTTP integration with the overseer)
```

The agent uses a **ReAct** (Reason → Act → Observe) loop: the LLM decides
whether to call a tool or produce a final answer, tools are executed locally,
and the observation is fed back into the context for the next iteration.

## Quick start

```python
from chatbot import ChatbotAgent

# Standalone (no ORACLE overseer)
agent = ChatbotAgent()

print(agent.chat("What is 12 * 8?"))
# → Tool 'calculate' returned: 96

print(agent.chat("What time is it?"))
# → Tool 'datetime' returned: 2026-07-27 07:30:00 UTC

print(agent.chat("help"))
# → lists all available tools

agent.shutdown()
```

### Connect to ORACLE

```python
agent = ChatbotAgent(oracle_url="http://localhost:8080", agent_name="chatbot-child-1")
```

The agent will:
1. Register itself with ORACLE on startup (`POST /api/v1/agents/register`).
2. Send heartbeats every 30 s (`POST /api/v1/agents/heartbeat`).
3. Execute any task ORACLE dispatches and send the result back
   (`POST /api/v1/tasks/result`).
4. Deregister on `agent.shutdown()`.

### Use a real LLM backend

```python
from chatbot import ChatbotAgent, LLMBackend, LLMResponse, Message
from typing import Any, Dict, List

class MyOpenAIBackend(LLMBackend):
    def complete(self, messages: List[Message], tools: List[Dict[str, Any]]) -> LLMResponse:
        # Map messages to OpenAI format, call the API, parse tool_calls / content
        ...

agent = ChatbotAgent(backend=MyOpenAIBackend())
```

### Register a custom tool

```python
from chatbot import ChatbotAgent, ToolRegistry

registry = ToolRegistry.default()

def my_tool(args):
    return {"result": f"Hello, {args.get('name', 'world')}!"}

registry.register(
    name="greet",
    fn=my_tool,
    description="Greet someone by name.",
    parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
)

agent = ChatbotAgent(tool_registry=registry)
print(agent.chat("greet Alice"))
```

## Module overview

| File | Purpose |
|------|---------|
| `agent.py` | `ChatbotAgent`, `LLMBackend`, `RuleBasedBackend`, `ConversationMemory` |
| `tools.py` | `ToolRegistry` and built-in tool implementations |
| `oracle_client.py` | `OracleClient` – ORACLE overseer HTTP integration |
| `__init__.py` | Public API re-exports |

## Running tests

```bash
python -m pytest chatbot/tests/ -v
```
