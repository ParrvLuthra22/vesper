"""WebSearchAgent - web search + summarization pipeline using Tavily and Gemini."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from urllib import request as urlrequest

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency import
    genai = None  # type: ignore
    GEMINI_AVAILABLE = False

try:
    from tavily import TavilyClient

    TAVILY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency import
    TavilyClient = None  # type: ignore
    TAVILY_AVAILABLE = False

from agents.base_agent import AgentCapability, BaseAgent
from schemas.events import (
    ActionRequestEvent,
    ActionResultEvent,
    HUDSearchResultsEvent,
    IntentRecognizedEvent,
    VoiceOutputEvent,
)
from utils.applescript import run_applescript
from utils.api_keys import get_gemini_api_key, get_openrouter_api_key


TRIGGER_PHRASES = [
    "search for",
    "look up",
    "what is",
    "tell me about",
    "latest news on",
    "who is",
    "how does",
]
VISION_EXCLUSION_PHRASES = [
    "on my screen",
    "read my screen",
    "describe my screen",
    "describe screen",
    "read that",
    "what does it say",
]


class WebSearchAgent(BaseAgent):
    """Agent that searches the web and returns concise factual responses."""

    def __init__(
        self,
        name: Optional[str] = None,
        event_bus=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name or "WebSearchAgent", event_bus=event_bus, config=config)
        self._tavily_client: Optional[TavilyClient] = None
        self._gemini_model = None
        self._openrouter_api_key: Optional[str] = None
        self._openrouter_model: str = "x-ai/grok-3-mini-beta"
        self._openrouter_endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
        self._llm_provider: str = "auto"
        self._last_tavily_call: float = 0.0
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_limit = 20

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="web_search",
                description="Searches web with Tavily and summarizes with Gemini",
                input_events=["IntentRecognizedEvent"],
                output_events=["VoiceOutputEvent", "HUDSearchResultsEvent"],
            )
        ]

    async def _setup(self) -> None:
        self._subscribe(IntentRecognizedEvent, self._handle_intent)
        self._subscribe(ActionRequestEvent, self._handle_action_request)
        self._initialize_clients()

    async def _teardown(self) -> None:
        self._tavily_client = None
        self._gemini_model = None
        self._cache.clear()

    def _initialize_clients(self) -> None:
        self._llm_provider = str(self._get_config("web_search.llm_provider", "auto")).strip().lower()
        tavily_key = (
            os.getenv("TAVILY_API_KEY")
            or self._get_config("web_search.tavily_api_key")
            or self._get_config("system.apis.tavily.api_key")
        )
        self._openrouter_api_key = get_openrouter_api_key(self._get_config)
        self._openrouter_model = str(
            self._get_config("web_search.openrouter.model", "x-ai/grok-3-mini-beta")
        ).strip()
        self._openrouter_endpoint = str(
            self._get_config(
                "web_search.openrouter.endpoint",
                "https://openrouter.ai/api/v1/chat/completions",
            )
        ).strip()

        if tavily_key and TAVILY_AVAILABLE:
            self._tavily_client = TavilyClient(api_key=tavily_key)
        elif not TAVILY_AVAILABLE:
            self._logger.warning("tavily-python SDK unavailable; web results retrieval disabled")
        else:
            self._logger.warning("TAVILY_API_KEY not configured; web results retrieval disabled")

        gemini_key = get_gemini_api_key(self._get_config)
        if gemini_key and GEMINI_AVAILABLE and genai is not None:
            genai.configure(api_key=gemini_key)
            self._gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        elif gemini_key and (not GEMINI_AVAILABLE or genai is None):
            self._logger.warning("Gemini key found but google-generativeai SDK is unavailable")
        else:
            self._logger.info("Gemini not configured for web search LLM tasks")

        if self._openrouter_api_key:
            self._logger.info(f"OpenRouter enabled for web search LLM tasks: model={self._openrouter_model}")
        else:
            self._logger.info("OpenRouter key not configured for web search LLM tasks")

        if (
            self._llm_provider == "openrouter"
            and not self._openrouter_api_key
        ):
            self._logger.warning("web_search.llm_provider=openrouter but OPENROUTER_API_KEY is missing")
        if self._llm_provider == "gemini" and self._gemini_model is None:
            self._logger.warning("web_search.llm_provider=gemini but Gemini is not available")

        if self._llm_provider not in {"auto", "gemini", "openrouter", "local"}:
            self._logger.warning(
                f"Unknown web_search.llm_provider='{self._llm_provider}', falling back to auto"
            )
            self._llm_provider = "auto"

    def _is_configured(self) -> bool:
        return bool(self._tavily_client and (self._gemini_model is not None or self._openrouter_api_key))

    def _should_prefer_openrouter(self) -> bool:
        if self._llm_provider == "openrouter":
            return True
        if self._llm_provider == "gemini":
            return False
        if self._llm_provider == "local":
            return False
        # auto
        return bool(self._openrouter_api_key)

    def _has_any_llm(self) -> bool:
        if self._llm_provider == "local":
            return True
        if self._should_prefer_openrouter():
            return bool(self._openrouter_api_key)
        return self._gemini_model is not None or bool(self._openrouter_api_key)

    async def _generate_text(self, prompt: str) -> str:
        if self._llm_provider == "local":
            return ""
        if self._should_prefer_openrouter() and self._openrouter_api_key:
            text = await self._call_openrouter(prompt)
            if text:
                return text
        if self._gemini_model is not None:
            text = await self._call_gemini(prompt)
            if text:
                return text
        if self._openrouter_api_key:
            return await self._call_openrouter(prompt)
        return ""

    async def _call_gemini(self, prompt: str) -> str:
        if self._gemini_model is None:
            return ""

        def _run() -> str:
            response = self._gemini_model.generate_content(prompt)
            return (getattr(response, "text", "") or "").strip()

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _run)
        except Exception as exc:
            self._logger.warning(f"Gemini call failed in WebSearchAgent: {exc}")
            return ""

    async def _call_openrouter(self, prompt: str) -> str:
        if not self._openrouter_api_key:
            return ""

        payload = {
            "model": self._openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self._openrouter_api_key}",
            "Content-Type": "application/json",
        }

        def _run() -> str:
            req = urlrequest.Request(
                self._openrouter_endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=45) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            choices = raw.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {}) or {}
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                combined: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = str(item.get("text", "")).strip()
                        if text:
                            combined.append(text)
                return "\n".join(combined).strip()
            return ""

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _run)
        except Exception as exc:
            self._logger.warning(f"OpenRouter call failed in WebSearchAgent: {exc}")
            return ""

    @staticmethod
    def _extract_query_heuristic(text: str) -> str:
        cleaned = (text or "").strip()
        lower = cleaned.lower()
        for trigger in TRIGGER_PHRASES:
            idx = lower.find(trigger)
            if idx >= 0:
                query = cleaned[idx + len(trigger) :].strip(" .?!,:")
                if query:
                    return query
        return cleaned

    @staticmethod
    def _local_summary(query: str, results: Dict[str, Any]) -> str:
        snippets = []
        for item in results.get("results", [])[:3]:
            content = str(item.get("content", "")).strip()
            if content:
                snippets.append(content[:220].strip())
        if not snippets:
            return f"I found sources for {query}, but I couldn't build a useful summary."
        joined = " ".join(snippets)
        return f"For {query}: {joined[:520].strip()}"

    @staticmethod
    def _clean_query(text: str) -> str:
        value = (text or "").strip().strip("\"'`")
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if not lines:
            return ""
        return lines[0]

    async def _handle_intent(self, event: IntentRecognizedEvent) -> None:
        text = self._event_text(event)
        text_lower = text.lower() if text else ""
        if not text or self._is_vision_query(text_lower) or not self._is_search_trigger(text_lower):
            return

        if not self._has_any_llm():
            await self._emit_voice("Web search is not configured yet.", event)
            return

        query = await self._extract_query(text)
        if not query:
            await self._emit_voice("I couldn't extract a search query.", event)
            return

        await self._emit_voice("Thats a great idea sir", event)

        if self._tavily_client is None:
            summary = await self._answer_without_tavily(query)
            await self._emit_voice(summary, event)
            await self._emit(
                HUDSearchResultsEvent(
                    query=query,
                    summary=summary,
                    sources=[],
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )
            return

        results = await self._search_tavily(query)
        if not results.get("results"):
            await self._emit_voice(f"I couldn't find results for {query}.", event)
            return

        summary = await self._summarize(query, results)
        await self._emit_voice(summary, event)

        sources = [
            {"title": str(item.get("title", "Untitled")), "url": str(item.get("url", ""))}
            for item in results.get("results", [])[:3]
        ]

        await self._emit(
            HUDSearchResultsEvent(
                query=query,
                summary=summary,
                sources=sources,
                source=self._name,
                correlation_id=event.correlation_id or event.event_id,
            )
        )

        lower = text.lower()
        if ("open" in lower or "show me" in lower) and sources and sources[0].get("url"):
            await self._open_in_safari(sources[0]["url"], event)

    async def _handle_action_request(self, event: ActionRequestEvent) -> None:
        """
        Handle a Planner tool-call routed over the bus (the search_web tool).

        Runs the same Tavily+Gemini/OpenRouter search-and-summarize pipeline
        as _handle_intent, but takes an explicit query parameter and answers
        with an ActionResultEvent instead of only speaking the result.
        """
        if event.target_agent != self.name or event.action != "search_web":
            return

        query = str(event.parameters.get("query", "")).strip()
        if not query:
            await self._emit_action_result(event, success=False, error="Missing 'query' parameter")
            return

        if not self._has_any_llm():
            await self._emit_action_result(
                event, success=False, error="Web search is not configured yet"
            )
            return

        try:
            if self._tavily_client is None:
                summary = await self._answer_without_tavily(query)
            else:
                results = await self._search_tavily(query)
                if not results.get("results"):
                    await self._emit_action_result(
                        event, success=False, error=f"No results found for '{query}'"
                    )
                    return
                summary = await self._summarize(query, results)
        except Exception as exc:
            await self._emit_action_result(event, success=False, error=str(exc))
            return

        await self._emit_action_result(event, success=True, result=summary)

    async def _emit_action_result(
        self,
        event: ActionRequestEvent,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        await self._emit(ActionResultEvent(
            source=self._name,
            action=event.action,
            success=success,
            result=result,
            error=error or "",
            plan_id=event.plan_id,
            step_number=event.step_number,
            correlation_id=event.correlation_id,
        ))

    async def _extract_query(self, text: str) -> str:
        prompt = f"Extract the search query from: '{text}'. Return only the query string."
        generated = await self._generate_text(prompt)
        cleaned = self._clean_query(generated)
        if cleaned:
            return cleaned
        return self._extract_query_heuristic(text)

    async def _search_tavily(self, query: str) -> Dict[str, Any]:
        cached = self._cache.get(query)
        if cached is not None:
            self._cache.move_to_end(query)
            return cached

        now = time.monotonic()
        elapsed = now - self._last_tavily_call
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)

        def _run() -> Dict[str, Any]:
            return self._tavily_client.search(query, max_results=5, search_depth="advanced")

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _run)
        self._last_tavily_call = time.monotonic()

        self._cache[query] = results
        if len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

        return results

    async def _summarize(self, query: str, results: Dict[str, Any]) -> str:
        context = "\n\n".join(
            [
                f"Source: {r.get('url', '')}\n{r.get('content', '')}"
                for r in results.get("results", [])
            ]
        )
        summary_prompt = (
            f"Based on these search results, answer '{query}' in 3 clear sentences. "
            "Be direct and factual.\n\n"
            f"{context}"
        )

        generated = await self._generate_text(summary_prompt)
        if generated.strip():
            return generated.strip()
        return self._local_summary(query, results)

    async def _answer_without_tavily(self, query: str) -> str:
        prompt = (
            f"Answer '{query}' in 3 clear sentences. "
            "Be factual and direct. If uncertain, say what is uncertain."
        )
        generated = await self._generate_text(prompt)
        if generated.strip():
            return generated.strip()
        return f"I can help with {query}, but search sources are unavailable right now."

    async def _open_in_safari(self, url: str, event: IntentRecognizedEvent) -> None:
        script = (
            "tell application \"Safari\"\n"
            "  activate\n"
            f"  open location \"{self._escape_applescript(url)}\"\n"
            "end tell"
        )

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, lambda: run_applescript(script))
        except Exception as exc:
            await self._emit_voice(f"I found results but couldn't open Safari: {exc}", event)

    async def _emit_voice(self, text: str, event: IntentRecognizedEvent) -> None:
        await self._emit(
            VoiceOutputEvent(
                text=text,
                source=self._name,
                correlation_id=event.correlation_id or event.event_id,
            )
        )

    @staticmethod
    def _event_text(event: IntentRecognizedEvent) -> str:
        intent = getattr(event, "intent", "") or ""
        text = getattr(event, "text", "") or ""
        raw = getattr(event, "raw_text", "") or ""
        if text.strip():
            return text.strip()
        if raw.strip():
            return raw.strip()
        return intent.strip()

    @staticmethod
    def _is_search_trigger(text_lower: str) -> bool:
        return any(phrase in text_lower for phrase in TRIGGER_PHRASES)

    @staticmethod
    def _is_vision_query(text_lower: str) -> bool:
        return any(phrase in text_lower for phrase in VISION_EXCLUSION_PHRASES)

    @staticmethod
    def _escape_applescript(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')
