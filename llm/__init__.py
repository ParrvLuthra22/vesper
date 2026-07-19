"""
VESPER LLM Routing Layer.

Provides ModelRouter, a provider-agnostic entry point for chat/tool-calling
completions. Providers (Groq, Ollama, ...) are self-registering plugins under
llm/providers/ — see llm/providers/__init__.py for the registration contract.

Not wired into the Brain yet (see P03).
"""

from llm.types import LLMResponse, RouterError, ToolCall
from llm.router import ModelRouter

__all__ = [
    "ModelRouter",
    "LLMResponse",
    "ToolCall",
    "RouterError",
]
