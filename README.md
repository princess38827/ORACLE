# ORACLE
An agent overseer

## Child agents

| Agent | Path | Description |
|-------|------|-------------|
| **chatbot** | [`chatbot/`](chatbot/) | Agentic AI chatbot child – multi-turn conversation with tool-use (ReAct loop) and ORACLE overseer integration |

## Quick start – chatbot child

```python
from chatbot import ChatbotAgent

agent = ChatbotAgent()                          # standalone (no ORACLE URL needed)
print(agent.chat("What is 12 * 8?"))            # → 96
print(agent.chat("What time is it?"))           # → current UTC time
print(agent.chat("help"))                       # → list of built-in tools
agent.shutdown()
```

See [`chatbot/README.md`](chatbot/README.md) for full documentation.
