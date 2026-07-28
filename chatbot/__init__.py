"""
chatbot — Agentic AI chatbot child agent for the ORACLE overseer.

Quick-start
-----------
    from chatbot import ChatbotAgent

    agent = ChatbotAgent()
    print(agent.chat("What is 42 * 7?"))
    print(agent.chat("What time is it?"))
    agent.shutdown()
"""

from .agent import ChatbotAgent, LLMBackend, LLMResponse, Message, RuleBasedBackend
from .openai_backend import OpenAIBackend
from .oracle_client import AgentInfo, OracleClient, OracleTask
from .tools import ToolRegistry

__all__ = [
    "ChatbotAgent",
    "LLMBackend",
    "LLMResponse",
    "Message",
    "OpenAIBackend",
    "RuleBasedBackend",
    "AgentInfo",
    "OracleClient",
    "OracleTask",
    "ToolRegistry",
]
