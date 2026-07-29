"""Container entrypoint for the ORACLE chatbot child agent."""

import os
import sys

from chatbot import ChatbotAgent

oracle_url = os.environ.get("ORACLE_URL")  # None = standalone mode; no overseer required
agent = ChatbotAgent(oracle_url=oracle_url)

print("ORACLE chatbot ready. Type a message and press Enter. Ctrl-C to quit.", flush=True)
try:
    for line in sys.stdin:
        print(agent.chat(line.rstrip()), flush=True)
finally:
    agent.shutdown()
