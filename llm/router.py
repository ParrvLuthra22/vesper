"""
ModelRouter — provider-agnostic entry point for LLM chat/tool-calling completions.

Routing behavior:
    1. Resolve the primary and fallback (provider, model) pair for the given
       `purpose`, honoring any per-purpose override in config.
    2. Call the primary provider. On a rate-limit (429) response, retry once
       with exponential backoff, then fall back. On any other provider
       failure (network error, misconfiguration, ...), fall back immediately.
    3. Call the fallback provider once. If that also fails, return a
       `RouterError` (not raise) so the caller can surface a graceful message.

Not wired into the Brain yet — see P03.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Union

from llm.errors import ProviderError, RateLimitError
from llm.providers import get_provider_class
from llm.providers.base import LLMProvider
from llm.types import LLMResponse, RouterError
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelRouter:
    """Routes chat completions to a primary provider with automatic fallback."""

    #: Number of extra attempts against the primary provider after a 429
    #: before giving up on it and moving to the fallback.
    PRIMARY_MAX_RETRIES = 1
    #: Base delay for exponential backoff between primary retries.
    BACKOFF_BASE_SECONDS = 0.5

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Full app config dict (as produced by
                `config.settings.load_config_dict`), containing a top-level
                `llm:` section.
        """
        self._config = config or {}
        self._provider_instances: Dict[str, LLMProvider] = {}

    def _get_config(self, key: str, default: Any = None) -> Any:
        """Dot-path lookup against the full app config."""
        value: Any = self._config
        for part in key.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
            if value is None:
                return default
        return value

    def _resolve_tier(self, purpose: str, tier: str) -> Dict[str, str]:
        """
        Resolve the (provider, model) pair for `tier` ("primary"/"fallback"),
        applying any per-purpose override on top of the tier's base config.
        """
        base = dict(self._get_config(f"llm.{tier}", {}) or {})
        override = self._get_config(f"llm.purposes.{purpose}.{tier}", {}) or {}
        if isinstance(override, dict):
            base.update({k: v for k, v in override.items() if v})
        return base

    def _get_provider_instance(self, name: str) -> LLMProvider:
        """Lazily construct (and cache) one provider instance per name."""
        if name not in self._provider_instances:
            provider_cls = get_provider_class(name)
            self._provider_instances[name] = provider_cls(config=self._config)
        return self._provider_instances[name]

    async def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
        purpose: str = "planning",
    ) -> Union[LLMResponse, RouterError]:
        """
        Perform a chat/tool-calling completion, routing per config.

        Returns an `LLMResponse` on success. If both the primary and
        fallback providers fail, returns a `RouterError` instead of raising —
        callers should check `isinstance(result, RouterError)` and surface
        `result.user_message` gracefully.
        """
        primary_tier = self._resolve_tier(purpose, "primary")
        fallback_tier = self._resolve_tier(purpose, "fallback")

        primary_result = await self._attempt_tier(
            tier=primary_tier,
            tier_label="primary",
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            purpose=purpose,
            max_retries=self.PRIMARY_MAX_RETRIES,
        )
        if isinstance(primary_result, LLMResponse):
            return primary_result

        logger.warning(
            f"[ROUTE] primary failed for purpose={purpose!r} ({primary_result}); "
            "falling back"
        )

        fallback_result = await self._attempt_tier(
            tier=fallback_tier,
            tier_label="fallback",
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            purpose=purpose,
            max_retries=0,
        )
        if isinstance(fallback_result, LLMResponse):
            return fallback_result

        logger.error(
            f"[ROUTE] both primary and fallback failed for purpose={purpose!r}: "
            f"primary={primary_result!r} fallback={fallback_result!r}"
        )
        return RouterError(
            primary_error=primary_result,
            fallback_error=fallback_result,
            purpose=purpose,
        )

    async def _attempt_tier(
        self,
        tier: Dict[str, str],
        tier_label: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: str,
        temperature: float,
        purpose: str,
        max_retries: int,
    ) -> Union[LLMResponse, str]:
        """
        Try one tier (primary or fallback), retrying on rate limits up to
        `max_retries` times with exponential backoff.

        Returns an `LLMResponse` on success, or a short error-summary string
        once all attempts for this tier are exhausted.
        """
        provider_name = tier.get("provider")
        model = tier.get("model")
        if not provider_name or not model:
            return f"{tier_label} not configured (missing provider/model)"

        try:
            provider = self._get_provider_instance(provider_name)
        except Exception as exc:
            logger.error(f"[ROUTE] could not load provider '{provider_name}': {exc}")
            return f"could not load provider '{provider_name}': {exc}"

        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                response = await provider.complete(
                    messages=messages,
                    model=model,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                )
                self._log_success(tier_label, provider_name, model, purpose, response)
                return response
            except RateLimitError as exc:
                last_error = f"rate_limited: {exc}"
                self._log_failure(tier_label, provider_name, model, purpose, last_error, attempt)
                if attempt < max_retries:
                    backoff = self.BACKOFF_BASE_SECONDS * (2**attempt)
                    logger.warning(
                        f"[ROUTE] {tier_label} ({provider_name}/{model}) rate-limited, "
                        f"retrying in {backoff:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(backoff)
                    continue
                break
            except ProviderError as exc:
                last_error = str(exc)
                self._log_failure(tier_label, provider_name, model, purpose, last_error, attempt)
                break
            except Exception as exc:  # pragma: no cover - defensive catch-all
                last_error = f"unexpected error: {exc}"
                logger.error(
                    f"[ROUTE] {tier_label} ({provider_name}/{model}) raised an unexpected "
                    f"error: {exc}",
                    exc_info=True,
                )
                break

        return last_error

    @staticmethod
    def _log_success(
        tier_label: str,
        provider_name: str,
        model: str,
        purpose: str,
        response: LLMResponse,
    ) -> None:
        logger.info(
            f"[ROUTE] tier={tier_label} provider={provider_name} model={model} "
            f"purpose={purpose} status=success latency_ms={response.latency_ms:.1f} "
            f"tokens_in={response.usage.get('prompt_tokens', 0)} "
            f"tokens_out={response.usage.get('completion_tokens', 0)} "
            f"tool_calls={len(response.tool_calls)}"
        )

    @staticmethod
    def _log_failure(
        tier_label: str,
        provider_name: str,
        model: str,
        purpose: str,
        error: str,
        attempt: int,
    ) -> None:
        logger.warning(
            f"[ROUTE] tier={tier_label} provider={provider_name} model={model} "
            f"purpose={purpose} status=failed attempt={attempt + 1} error={error}"
        )
