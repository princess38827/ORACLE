FROM python:3.12-slim

WORKDIR /app

# Install dependencies (none required for stdlib-only chatbot, but present for
# any optional LLM backend packages that may be added later)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY chatbot/ ./chatbot/
COPY entrypoint.py .

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Optional: set ORACLE_URL to connect to the ORACLE overseer.
ENV PYTHONUNBUFFERED=1

CMD ["python", "entrypoint.py"]
