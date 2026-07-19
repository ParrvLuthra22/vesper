"""
Provider registry for the LLM routing layer.

Providers self-register via `@register_provider("name")` at import time.
ModelRouter looks providers up by name (from config) through this registry —
it never imports a concrete provider class. Adding a new backend therefore
requires only a new `llm/providers/<name>_provider.py` file plus a config
entry naming it — no changes here or in `llm/router.py`.

Resolution order for a given name:
    1. Already-registered classes (covers providers imported ahead of time,
       e.g. by tests registering a dummy provider directly).
    2. Convention-based import of `llm.providers.<name>_provider`, which is
       expected to self-register via the decorator as a side effect.
"""

from __future__ import annotations

import importlib
from typing import Dict, Type

from llm.providers.base import LLMProvider

_REGISTRY: Dict[str, Type[LLMProvider]] = {}


def register_provider(name: str):
    """Class decorator that registers an LLMProvider subclass under `name`."""

    def _decorator(cls: Type[LLMProvider]) -> Type[LLMProvider]:
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_provider_class(name: str) -> Type[LLMProvider]:
    """Resolve a provider class by name, raising KeyError if none is found."""
    if name not in _REGISTRY:
        try:
            importlib.import_module(f"llm.providers.{name}_provider")
        except ImportError as exc:
            raise KeyError(
                f"Unknown LLM provider '{name}': no registered provider and no "
                f"llm/providers/{name}_provider.py module found ({exc})"
            ) from exc

    if name not in _REGISTRY:
        raise KeyError(
            f"Module llm.providers.{name}_provider did not register a "
            f"provider named '{name}' via @register_provider"
        )

    return _REGISTRY[name]


def registered_providers() -> Dict[str, Type[LLMProvider]]:
    """Snapshot of currently registered provider names (diagnostics/tests)."""
    return dict(_REGISTRY)
