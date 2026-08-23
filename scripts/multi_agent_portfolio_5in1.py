"""
Multi-Agent Orchestration with OpenAI Agents SDK: 5-in-1 Smart Portfolio Analysis
================================================================================
Clean, runnable Python script — converts the notebook into a production-ready workflow.

Architecture: Hub-and-Spoke
- Hub: Head Portfolio Manager Agent (PM)
- Spokes: 5 Specialist Agents
  1. Macro Agent         -> FRED API + WebSearch
  2. Fundamental Agent   -> Yahoo Finance (via MCP) + WebSearch
  3. Quantitative Agent  -> Code Interpreter (stats, correlations, charts)
  4. Sentiment Agent     -> WebSearch for news, analyst sentiment
  5. Risk Agent          -> Scenario planning, position sizing

Pattern: Agents-as-Tool with parallel execution

Requirements:
  pip install openai-agents python-dotenv fredapi yfinance

Env vars:
  OPENAI_API_KEY - required
  FRED_API_KEY   - optional but recommended for macro data

Run:
  python multi_agent_portfolio_5in1.py --ticker GOOGL --question "How would a rate cut affect GOOGL?"

Disclaimer: Educational only. Not investment advice.
"""

import os
import asyncio
import datetime
import json
import argparse
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

# --- SDK Imports ---
try:
    from agents import Agent, Runner, function_tool, ModelSettings, trace
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("OpenAI Agents SDK not found. Install with: pip install openai-agents")
    # Dummy decorator for linting when SDK missing
    def function_tool(*args, **kwargs):
        def wrapper(fn): return fn
        return wrapper

# --- CONFIG ---
DEFAULT_MODEL = "gpt-4.1"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DISCLAIMER = "\n\nDISCLAIMER: I am an AI language model, not a registered investment adviser. Educational only. Consult a qualified financial professional."

# --- Custom Tools ---

@function_tool(description_override="Fetch FRED economic data (GDP, CPI, FEDFUNDS, DGS10, UNRATE). Returns latest values.")
def get_fred_data(series_ids: str) -> str:
    """
    series_ids: comma-separated FRED series, e.g. 'GDP,CPIAUCSL,FEDFUNDS,DGS10,UNRATE'
    """
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        return "FRED_API_KEY not set. Returning mock context: Fed Funds 4.25-4.5%, CPI elevated ~320, GDP growing, Unemployment 4.2%."
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_key)
        results = {}
        for sid in [s.strip() for s in series_ids.split(",")]:
            try:
                series = fred.get_series(sid)
                results[sid] = {"latest": float(series.dropna().iloc[-1]), "last_date": str(series.dropna().index[-1].date())}
            except Exception as e:
                results[sid] = {"error": str(e)}
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error fetching FRED: {e}"

@function_tool(description_override="Get stock fundamentals and price history via yfinance. For local dev without MCP.")
def get_stock_data(ticker: str, period: str = "1y") -> str:
    """ticker: e.g. GOOGL, period: 1mo, 6mo, 1y, 5y"""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period=period)
        summary = {
            "ticker": ticker,
            "name": info.get("longName"),
            "current_price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "margins": {
                "gross": info.get("grossMargins"),
                "operating": info.get("operatingMargins"),
                "profit": info.get("profitMargins"),
            },
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "price_history_last_5": hist['Close'].tail().to_dict() if not hist.empty else {},
        }
        return json.dumps(summary, indent=2, default=str)
    except Exception as e:
        return f"Error fetching stock data for {ticker}: {e}"

@function_tool(description_override="Write the final investment memo to a markdown file and return path.")
def save_memo_tool(content: str, filename: str = "investment_memo.md") -> str:
    path = OUTPUT_DIR / filename
    path.write_text(content + DISCLAIMER, encoding="utf-8")
    return json.dumps({"file": str(path), "status": "saved"})

# --- Agent Prompts ---

PM_BASE_PROMPT = """
You are the Head Portfolio Manager at a top-tier long/short equity fund.
Philosophy: Originality, risk-awareness, challenge consensus, scenario planning.

Your team (5 specialists):
- macro_analysis: GDP, rates, inflation, USD, policy
- fundamental_analysis: revenue, margins, moat, earnings
- quantitative_analysis: correlations to rates, technicals, momentum, vol
- sentiment_analysis: news, analyst ratings, market narrative
- risk_analysis: tail risks, position sizing, best/worst/base cases

WORKFLOW:
1. Classify task: ticker? question type? date is {today}
2. Call run_all_specialists_parallel with tailored inputs for each specialist. BE SPECIFIC.
3. Review outputs critically. Look for contradictions and differentiated insights.
4. Synthesize into a memo with:
   - Executive Summary (2-3 sentences, include differentiated insight)
   - 5 specialist sections (evidence-based)
   - Portfolio Manager Perspective (your synthesis)
   - Recommendation & Price Target with scenarios (base/bear/bull)
5. Call save_memo_tool with final markdown. End memo with **END_OF_MEMO**

RULES:
- Always use parallel tool first for speed.
- Don't be generic. Quantify.
- If data missing, state it and use reasoning.
- Never just follow consensus - challenge it.
"""

SPECIALIST_PROMPTS = {
    "macro": "You are a Macro Economist. Use get_fred_data tool. Focus on rates, inflation, GDP, unemployment, DXY, tariffs. Be evidence-based. Output markdown with data table.",
    "fundamental": "You are a Fundamental Equity Analyst. Use get_stock_data. Analyze revenue, net income, margins, moat, cloud/AI growth. Include quarterly trends if available. Output markdown.",
    "quant": "You are a Quantitative Analyst. Use get_stock_data. Analyze price vs moving averages, momentum, volatility, correlation to FEDFUNDS and DGS10 (if you have data). Discuss rate sensitivity. Be original with stats.",
    "sentiment": "You are a Sentiment & News Analyst. Use your knowledge + reasoning about recent analyst sentiment (Buy/Hold/Sell), news flow, AI narrative, regulatory risks. Summarize consensus vs variant view.",
    "risk": "You are a Risk Manager. Given other analyses, outline tail risks: regulatory, macro, sector rotation, AI narrative failure. Propose position sizing and hedging. Provide 3 scenarios: Bull/Base/Bear with price targets."
}

def build_specialist(name: str, instructions: str) -> Any:
    if not SDK_AVAILABLE:
        return None
    return Agent(
        name=f"{name.title()} Specialist",
        instructions=instructions,
        model=DEFAULT_MODEL,
        tools=[get_fred_data, get_stock_data] if name in ["macro", "fundamental", "quant"] else [get_stock_data],
        model_settings=ModelSettings(temperature=0.2)
    )

def build_head_pm(specialists: Dict[str, Any]):
    if not SDK_AVAILABLE:
        return None

    # Wrap specialists as tools
    def make_tool(agent, tool_name, desc):
        @function_tool(name_override=tool_name, description_override=desc)
        async def _tool(input: str) -> str:
            result = await Runner.run(agent, input, max_turns=15)
            return result.final_output
        return _tool

    tools = []
    for key, agent in specialists.items():
        tools.append(make_tool(agent, f"{key}_analysis", f"Run {key} analysis: {SPECIALIST_PROMPTS[key][:80]}..."))

    @function_tool(name_override="run_all_specialists_parallel", description_override="Run all 5 specialist analyses IN PARALLEL. Pass tailored inputs for each. Returns dict of results.")
    async def run_all_parallel(macro_input: str, fundamental_input: str, quant_input: str, sentiment_input: str, risk_input: str) -> str:
        tasks = {
            "macro": Runner.run(specialists["macro"], macro_input, max_turns=15),
            "fundamental": Runner.run(specialists["fundamental"], fundamental_input, max_turns=15),
            "quant": Runner.run(specialists["quant"], quant_input, max_turns=15),
            "sentiment": Runner.run(specialists["sentiment"], sentiment_input, max_turns=15),
            "risk": Runner.run(specialists["risk"], risk_input, max_turns=15),
        }
        results = {}
        # Run concurrently
        completed = await asyncio.gather(*tasks.values())
        for k, res in zip(tasks.keys(), completed):
            results[k] = res.final_output
        return json.dumps(results, indent=2)

    tools.append(run_all_parallel)
    tools.append(save_memo_tool)

    return Agent(
        name="Head Portfolio Manager",
        instructions=PM_BASE_PROMPT.format(today=datetime.date.today().strftime("%B %d, %Y")) + DISCLAIMER,
        model=DEFAULT_MODEL,
        tools=tools,
        model_settings=ModelSettings(parallel_tool_calls=True, tool_choice="auto", temperature=0)
    )

async def main():
    parser = argparse.ArgumentParser(description="5-in-1 Smart Portfolio Agent")
    parser.add_argument("--ticker", default="GOOGL", help="Ticker symbol")
    parser.add_argument("--question", default="How would a planned interest rate reduction affect holdings and what is realistic price target by year end considering Macro, Fundamental, Quantitative, Sentiment, Risk?", help="Investor question")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set")

    if not SDK_AVAILABLE:
        print("SDK missing - running in mock mode with direct tool calls")
        print(get_stock_data(args.ticker))
        print(get_fred_data("GDP,CPIAUCSL,FEDFUNDS,DGS10,UNRATE,DXY"))
        return

    print(f"Building 5-in-1 Agent System for {args.ticker}...")
    specialists = {k: build_specialist(k, v) for k, v in SPECIALIST_PROMPTS.items()}
    pm = build_head_pm(specialists)

    today_str = datetime.date.today().strftime("%B %d, %Y")
    full_question = f"Today is {today_str}. Ticker: {args.ticker}. Question: {args.question}"

    print("Running workflow with tracing...")
    with trace("5-in-1 Investment Workflow", metadata={"ticker": args.ticker}):
        response = await Runner.run(pm, full_question, max_turns=40)

    print("\n" + "="*80)
    print("FINAL OUTPUT:")
    print(response.final_output[:5000])
    print("\nCheck outputs/ folder for full memo.")

if __name__ == "__main__":
    asyncio.run(main())
