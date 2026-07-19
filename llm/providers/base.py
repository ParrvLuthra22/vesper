"""Abstract base class every LLM provider must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from llm.types import LLMResponse


class LLMProvider(ABC):
    """
    Minimal contract for an LLM backend usable by ModelRouter.

    To add a new provider:
        1. Create `llm/providers/<name>_provider.py`.
        2. Subclass `LLMProvider` and decorate the class with
           `@register_provider("<name>")`.
        3. Point `llm.primary.provider` / `llm.fallback.provider` / a purpose
           override at "<name>" in settings.yaml.

    ModelRouter never imports a concrete provider class by name — it resolves
    providers purely through the registry — so none of the above requires any
    change to router.py.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}

    def _get_config(self, key: str, default: Any = None) -> Any:
        """Dot-path lookup against the full app config passed at construction."""
        value: Any = self._config
        for part in key.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
            if value is None:
                return default
        return value

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        Perform one completion call and return a normalized LLMResponse.

        Must raise `llm.errors.RateLimitError` for HTTP 429 / rate-limit
        responses, and `llm.errors.ProviderError` for any other failure
        (network error, misconfiguration, bad response) so ModelRouter can
        decide whether to retry-then-fallback or fall back immediately.
        """
        raise NotImplementedError

    async def is_available(self) -> bool:
        """Cheap pre-flight check (e.g. API key configured, server reachable)."""
        return True
