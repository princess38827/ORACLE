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

## External references

- [AgiBot OS documentation](https://www.agibot.com/DOCS/OS?gad_source=1&gad_campaignid=23894634934&gbraid=0AAAABC0KuBBXErWWkHV_xEWgX2ud2WU5L&gclid=CjwKCAjwyabTBhBFEiwAM3mNUDCZC7Q7M1Sifmu8tEi0lTlTVlZxSMXzSRSK846LaEhcQpOUGkV6wRoCssgQAvD_BwE)
