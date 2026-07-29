"""
backends — Optional LLM backend implementations for the ORACLE chatbot.

Available backends
------------------
- ``OpenAIBackend``: delegates to the OpenAI chat-completions API.
  Requires ``pip install openai``.
"""

from .openai_backend import OpenAIBackend

__all__ = ["OpenAIBackend"]
