"""Helpers for resolving API keys from config and environment."""

from __future__ import annotations

import os
from typing import Callable, Optional


ConfigGetter = Callable[[str], Optional[str]]


def _clean(value: Optional[str]) -> Optional[str]:
    """Normalize config/env values and drop empty strings."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def get_env_value(*names: str) -> Optional[str]:
    """
    Resolve the first non-empty environment value by trying name variants.

    For each provided name we also try upper/lower variants so callers can pass
    a canonical key while still accepting lowercase overrides.
    """
    for name in names:
        for candidate in (name, name.upper(), name.lower()):
            value = _clean(os.getenv(candidate))
            if value:
                return value
    return None


def get_gemini_api_key(config_getter: Optional[ConfigGetter] = None) -> Optional[str]:
    """
    Resolve Gemini key from common env/config locations.

    Priority:
    1. Environment variables
    2. Optional config getter (dot-path lookups)
    """
    env_key = get_env_value(
        "GEMINI_API_KEY",
        "gemini_api_key",
        "GOOGLE_API_KEY",
        "google_api_key",
    )
    if env_key:
        return env_key

    if config_getter is None:
        return None

    for key in (
        "vision.gemini.api_key",
        "intent.gemini.api_key",
        "web_search.gemini.api_key",
        "reasoning.gemini.api_key",
        "gemini_api_key",
        "general.gemini_api_key",
    ):
        try:
            value = _clean(config_getter(key))
        except Exception:
            value = None
        if value:
            return value

    return None


def get_groq_api_key(config_getter: Optional[ConfigGetter] = None) -> Optional[str]:
    """
    Resolve Groq key from common env/config locations.

    Priority:
    1. Environment variables (GROQ_API_KEY)
    2. Optional config getter (dot-path lookups)
    """
    env_key = get_env_value(
        "GROQ_API_KEY",
        "groq_api_key",
    )
    if env_key:
        return env_key

    if config_getter is None:
        return None

    for key in (
        "llm.groq.api_key",
        "groq_api_key",
        "general.groq_api_key",
    ):
        try:
            value = _clean(config_getter(key))
        except Exception:
            value = None
        if value:
            return value

    return None


def get_openrouter_api_key(config_getter: Optional[ConfigGetter] = None) -> Optional[str]:
    """
    Resolve OpenRouter key from common env/config locations.

    Priority:
    1. Environment variables
    2. Optional config getter (dot-path lookups)
    """
    env_key = get_env_value(
        "OPENROUTER_API_KEY",
        "openrouter_api_key",
    )
    if env_key:
        return env_key

    if config_getter is None:
        return None

    for key in (
        "web_search.openrouter.api_key",
        "intent.openrouter.api_key",
        "openrouter_api_key",
        "general.openrouter_api_key",
    ):
        try:
            value = _clean(config_getter(key))
        except Exception:
            value = None
        if value:
            return value

    return None
