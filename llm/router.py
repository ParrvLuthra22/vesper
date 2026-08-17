"""
ModelRouter — provider-agnostic entry point for LLM chat/tool-calling completions.

Routing behavior:
    1. Resolve the primary and fallback (provider, model) pair for the given
       `purpose`, honoring any per-purpose override in config.
    2. Pace: if this request's estimated size would push the rolling 60s
       spend past the primary's tokens-per-minute ceiling, wait briefly
       first rather than firing into a guaranteed 429.
    3. Call the primary provider. On a rate-limit (429), wait the window the
       provider asked for (`retry-after`, capped) and retry on the primary —
       a TPM reset is usually a second or two, and riding it out keeps the
       turn on the fast provider. On any other failure, fall back at once.
    4. Only if the primary is still failing, call the fallback once. If that
       also fails, return a `RouterError` (not raise) so the caller can
       surface a graceful message.

The fallback is expected to be a small *local* model (Ollama), which is
slow to cold-load — so crossing that line emits an `on_status` notice, and
the router treats it as a rescue path, never a routine one.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from llm.errors import ProviderError, RateLimitError
from llm.providers import get_provider_class
from llm.providers.base import LLMProvider
from llm.token_meter import TokenBudget, estimate_request_tokens
from llm.types import LLMResponse, RouterError
from utils.logger import get_logger

logger = get_logger(__name__)

#: Providers whose fallback is a local model that must load into RAM before
#: it can answer. Crossing to one of these is worth telling the user about.
LOCAL_PROVIDERS = frozenset({"ollama"})

RATE_LIMIT_WAIT_MESSAGE = "One moment, Sir."
LOCAL_FALLBACK_MESSAGE = "Switching to local, Sir — one moment."


class ModelRouter:
    """Routes chat completions to a primary provider with automatic fallback."""

    #: Number of extra attempts against the primary provider after a 429
    #: before giving up on it and moving to the fallback.
    PRIMARY_MAX_RETRIES = 1
    #: Base delay for exponential backoff between primary retries, used only
    #: when the provider did not tell us how long to wait.
    BACKOFF_BASE_SECONDS = 0.5
    #: Never wait longer than this for a rate limit to clear before giving
    #: up on the primary — past it, the local fallback is the faster path.
    DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS = 4.0
    #: Groq's free tier. Used for client-side pacing only; the provider
    #: remains the authority on whether a call is actually over the line.
    DEFAULT_TOKENS_PER_MINUTE = 8000

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Full app config dict (as produced by
                `config.settings.load_config_dict`), containing a top-level
                `llm:` section.
        """
        self._config = config or {}
        self._provider_instances: Dict[str, LLMProvider] = {}

        self._max_rate_limit_wait = float(
            self._get_config(
                "llm.rate_limit.max_wait_seconds", self.DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS
            )
        )
        #: One budget per provider name — only the primary's ceiling matters
        #: in practice, but keying by provider keeps a local fallback (no
        #: meaningful limit) from being paced against a cloud tier's numbers.
        self._budgets: Dict[str, TokenBudget] = {}

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

    def _get_budget(self, provider_name: str) -> TokenBudget:
        """
        The rolling token budget for one provider.

        A local provider has no meaningful per-minute ceiling, so it gets a
        disabled budget (0 TPM) and is never paced.
        """
        if provider_name not in self._budgets:
            if provider_name in LOCAL_PROVIDERS:
                tokens_per_minute = 0
            else:
                tokens_per_minute = int(
                    self._get_config(
                        f"llm.{provider_name}.tokens_per_minute",
                        self._get_config(
                            "llm.rate_limit.tokens_per_minute", self.DEFAULT_TOKENS_PER_MINUTE
                        ),
                    )
                )
            self._budgets[provider_name] = TokenBudget(
                tokens_per_minute=tokens_per_minute,
                max_wait_seconds=self._max_rate_limit_wait,
            )
        return self._budgets[provider_name]

    @staticmethod
    def _notify(on_status: Optional[Callable[[str], None]], message: str) -> None:
        """Deliver an operational notice without ever letting a caller's
        callback break the completion it is narrating."""
        if on_status is None:
            return
        try:
            on_status(message)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"[ROUTE] on_status callback raised, ignoring: {exc}")

    async def provider_status(self, purpose: str = "planning") -> List[Dict[str, Any]]:
        """
        Cheap health snapshot of the primary/fallback providers configured
        for `purpose` — for a /status-style display, not for routing
        decisions. Each entry's `available` comes from the provider's own
        `is_available()` (API key present / SDK installed / etc.), not a
        live network call.
        """
        statuses: List[Dict[str, Any]] = []
        for tier_label in ("primary", "fallback"):
            tier = self._resolve_tier(purpose, tier_label)
            provider_name = tier.get("provider")
            model = tier.get("model")
            entry: Dict[str, Any] = {
                "tier": tier_label,
                "provider": provider_name or "(not configured)",
                "model": model or "(not configured)",
                "available": False,
            }
            if provider_name and model:
                try:
                    provider = self._get_provider_instance(provider_name)
                    entry["available"] = await provider.is_available()
                except Exception as exc:
                    logger.debug(f"[ROUTE] provider_status could not load '{provider_name}': {exc}")
            statuses.append(entry)
        return statuses

    async def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
        purpose: str = "planning",
        on_token: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Union[LLMResponse, RouterError]:
        """
        Perform a chat/tool-calling completion, routing per config.

        Returns an `LLMResponse` on success. If both the primary and
        fallback providers fail, returns a `RouterError` instead of raising —
        callers should check `isinstance(result, RouterError)` and surface
        `result.user_message` gracefully.

        `on_token`, if given, streams text deltas as they arrive (only
        providers that support it will actually stream; others ignore it
        and return the complete response as usual).

        `on_status`, if given, receives short operational notices — riding
        out a rate limit, or crossing to the local fallback — so an
        unavoidable pause reads as deliberate rather than as a hang.
        """
        primary_tier = self._resolve_tier(purpose, "primary")
        fallback_tier = self._resolve_tier(purpose, "fallback")

        await self._pace(primary_tier, messages, tools, purpose, on_status)

        primary_result = await self._attempt_tier(
            tier=primary_tier,
            tier_label="primary",
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            purpose=purpose,
            max_retries=self.PRIMARY_MAX_RETRIES,
            on_token=on_token,
            on_status=on_status,
        )
        if isinstance(primary_result, LLMResponse):
            return primary_result

        logger.warning(
            f"[ROUTE] primary failed for purpose={purpose!r} ({primary_result}); "
            "falling back"
        )

        # A local fallback has to load the model into RAM before it can
        # answer — several seconds after an idle period. Say so, or it
        # reads as a freeze.
        if fallback_tier.get("provider") in LOCAL_PROVIDERS:
            self._notify(on_status, LOCAL_FALLBACK_MESSAGE)

        fallback_result = await self._attempt_tier(
            tier=fallback_tier,
            tier_label="fallback",
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            purpose=purpose,
            max_retries=0,
            on_token=on_token,
            on_status=on_status,
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

    async def _pace(
        self,
        tier: Dict[str, str],
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        purpose: str,
        on_status: Optional[Callable[[str], None]],
    ) -> None:
        """
        Delay briefly if sending this request now would exceed the primary
        provider's tokens-per-minute ceiling.

        Preemptive: a call that waits 1.2s here would otherwise have been a
        429 followed by a longer wait *plus* a retry, or a drop to the slow
        local fallback. Bounded by `llm.rate_limit.max_wait_seconds`.
        """
        provider_name = tier.get("provider")
        if not provider_name:
            return

        budget = self._get_budget(provider_name)
        if not budget.enabled:
            return

        estimated = estimate_request_tokens(messages, tools)
        delay = budget.delay_for(estimated)
        if delay <= 0:
            return

        logger.info(
            f"[ROUTE] pacing {provider_name} for purpose={purpose!r}: "
            f"est_tokens={estimated} used_60s={budget.used()} "
            f"limit={budget.effective_limit} waiting={delay:.1f}s"
        )
        self._notify(on_status, RATE_LIMIT_WAIT_MESSAGE)
        await asyncio.sleep(delay)

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
        on_token: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
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

        # Providers are extended purely by @register_provider (see
        # test_dummy_third_provider_requires_no_router_changes) — a
        # third-party provider predating on_token has no obligation to
        # accept it, so only pass it through when the provider actually
        # declares the parameter.
        extra_kwargs: Dict[str, Any] = {}
        if on_token is not None and "on_token" in inspect.signature(provider.complete).parameters:
            extra_kwargs["on_token"] = on_token

        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                response = await provider.complete(
                    messages=messages,
                    model=model,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    **extra_kwargs,
                )
                self._log_success(tier_label, provider_name, model, purpose, response)
                self._get_budget(provider_name).record(
                    response.usage.get("total_tokens")
                    or estimate_request_tokens(messages, tools)
                )
                return response
            except RateLimitError as exc:
                last_error = f"rate_limited: {exc}"
                self._log_failure(tier_label, provider_name, model, purpose, last_error, attempt)
                if attempt < max_retries:
                    # Prefer the window the provider actually asked for; a
                    # TPM reset is typically ~1-2s, so riding it out keeps
                    # the turn on the fast provider. Blind backoff only
                    # applies when no retry-after was sent.
                    requested = getattr(exc, "retry_after", None)
                    backoff = (
                        float(requested)
                        if requested
                        else self.BACKOFF_BASE_SECONDS * (2**attempt)
                    )
                    if backoff > self._max_rate_limit_wait:
                        logger.warning(
                            f"[ROUTE] {tier_label} ({provider_name}/{model}) rate-limited for "
                            f"{backoff:.1f}s — longer than the {self._max_rate_limit_wait:.1f}s "
                            "ceiling; falling back instead of waiting"
                        )
                        break
                    logger.warning(
                        f"[ROUTE] {tier_label} ({provider_name}/{model}) rate-limited, "
                        f"waiting {backoff:.1f}s then retrying "
                        f"(attempt {attempt + 1}/{max_retries}, "
                        f"source={'retry-after' if requested else 'backoff'})"
                    )
                    self._notify(on_status, RATE_LIMIT_WAIT_MESSAGE)
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
