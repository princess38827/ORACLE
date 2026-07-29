FROM python:3.12-slim

WORKDIR /app

# Install dependencies (none required for stdlib-only chatbot, but present for
# any optional LLM backend packages that may be added later)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY chatbot/ ./chatbot/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Optional: set ORACLE_URL to connect to the ORACLE overseer.
ENV PYTHONUNBUFFERED=1

CMD ["python", "-c", "\
from chatbot import ChatbotAgent; \
import os, sys; \
oracle_url = os.environ.get('ORACLE_URL'); \
agent = ChatbotAgent(oracle_url=oracle_url); \
print('ORACLE chatbot ready. Type a message and press Enter. Ctrl-C to quit.', flush=True); \
[print(agent.chat(line.rstrip()), flush=True) for line in sys.stdin]; \
agent.shutdown() \
"]
