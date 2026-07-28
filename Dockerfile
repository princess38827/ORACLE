FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY chatbot/ ./chatbot/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# OPENAI_API_KEY must be provided at runtime via environment variable.
# Optional: set ORACLE_URL to connect to the ORACLE overseer.
ENV PYTHONUNBUFFERED=1

CMD ["python", "-c", "\
from chatbot import ChatbotAgent; \
from chatbot.openai_backend import OpenAIBackend; \
import os, sys; \
model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'); \
oracle_url = os.environ.get('ORACLE_URL'); \
agent = ChatbotAgent(backend=OpenAIBackend(model=model), oracle_url=oracle_url); \
print('ORACLE chatbot ready. Type a message and press Enter. Ctrl-C to quit.', flush=True); \
[print(agent.chat(line.rstrip()), flush=True) for line in sys.stdin]; \
agent.shutdown() \
"]
