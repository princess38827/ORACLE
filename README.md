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

## Scripts

### Multi-Agent Portfolio Analysis (`scripts/multi_agent_portfolio_5in1.py`)

A hub-and-spoke multi-agent portfolio analysis workflow using the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).

**Architecture:** Head Portfolio Manager (hub) orchestrates 5 specialist agents in parallel:

| Specialist | Data Sources | Focus |
|------------|-------------|-------|
| Macro | FRED API | GDP, rates, inflation, unemployment |
| Fundamental | yfinance | Revenue, margins, moat, earnings |
| Quantitative | yfinance | Technicals, momentum, rate correlation |
| Sentiment | LLM reasoning | News flow, analyst ratings, narrative |
| Risk | Synthesis | Tail risks, position sizing, scenarios |

**Install dependencies:**

```bash
pip install openai-agents python-dotenv fredapi yfinance
```

**Set environment variables:**

```bash
export OPENAI_API_KEY=sk-...
export FRED_API_KEY=...   # optional, mock data used if unset
```

**Run:**

```bash
python scripts/multi_agent_portfolio_5in1.py --ticker GOOGL \
  --question "How would a rate cut affect GOOGL?"
```

The final investment memo is written to `outputs/investment_memo.md`.

> **Disclaimer:** Educational only. Not investment advice.
