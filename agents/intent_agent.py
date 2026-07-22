"""
Intent Agent Module - LLM-based Natural Language Understanding.

This agent processes user input to determine intent and extract entities.
It uses LLMs (Gemini or OpenAI) for sophisticated multi-command detection
and entity extraction with intelligent fallback logic.

Features:
    - Multi-command sequence detection ("Open VS Code and check CPU")
    - LLM-based intent classification with structured JSON output
    - Entity extraction with normalization
    - Fallback to pattern matching when LLM unavailable
    - Confidence scoring and thresholds
    - Versioned prompt templates

Architecture:
    VoiceInputEvent -> IntentAgent -> IntentRecognizedEvent / MultiIntentEvent
    
Event Flow:
    1. Receive VoiceInputEvent with transcribed text
    2. Send to LLM with structured prompt
    3. Parse JSON response for intents and entities
    4. Emit IntentRecognizedEvent (single) or MultiIntentEvent (multi)
    5. On failure, fallback to pattern matching

Example:
    Input: "Open VS Code and tell me CPU usage"
    Output: MultiIntentEvent with:
        [
            {"intent": "OPEN_APP", "entities": {"app": "Visual Studio Code"}},
            {"intent": "GET_SYSTEM_STATS", "entities": {"metric": "cpu"}}
        ]
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

from agents.base_agent import AgentCapability, BaseAgent
from bus.event_bus import EventBus
from orchestrator.reasoning_engine import ReasoningEngine
from schemas.events import (
    IntentRecognizedEvent,
    IntentUnknownEvent,
    MultiIntentEvent,
    VoiceInputEvent,
    VoiceOutputEvent,
)
from utils.logger import get_logger
from utils.prompts import IntentPrompts, FallbackPatterns, log_prompt_usage
from utils.api_keys import get_gemini_api_key


logger = get_logger(__name__)


# =============================================================================
# Intent Definitions
# =============================================================================

@dataclass
class IntentDefinition:
    """
    Definition of a recognized intent.
    
    Attributes:
        name: Unique intent identifier (UPPERCASE by convention)
        description: Human-readable description for LLM
        examples: Example utterances (used in prompt)
        required_slots: Entities that MUST be extracted
        optional_slots: Entities that MAY be extracted
        handler_agent: Agent responsible for execution
    """
    
    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    handler_agent: str = "system_agent"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompt templates."""
        return asdict(self)


# Complete intent catalog for VESPER
DEFAULT_INTENTS = [
    # Application Control
    IntentDefinition(
        name="OPEN_APP",
        description="Open/launch an application on macOS",
        examples=[
            "open Safari",
            "launch VS Code",
            "start Visual Studio Code",
            "run Terminal",
        ],
        required_slots=["app"],
        handler_agent="system_agent",
    ),
    IntentDefinition(
        name="CLOSE_APP",
        description="Close/quit an application",
        examples=[
            "close Safari",
            "quit Chrome",
            "exit Terminal",
            "kill Spotify",
        ],
        required_slots=["app"],
        handler_agent="system_agent",
    ),
    IntentDefinition(
        name="SWITCH_APP",
        description="Switch to a running application",
        examples=[
            "switch to Chrome",
            "go to Terminal",
            "focus on Slack",
        ],
        required_slots=["app"],
        handler_agent="system_agent",
    ),
    
    # System Stats
    IntentDefinition(
        name="GET_SYSTEM_STATS",
        description="Get system performance metrics",
        examples=[
            "what's my CPU usage",
            "how much RAM am I using",
            "check disk space",
            "tell me battery level",
            "system status",
        ],
        optional_slots=["metric"],  # cpu, memory, disk, battery, all
        handler_agent="system_agent",
    ),
    IntentDefinition(
        name="GET_TIME",
        description="Get current time or date",
        examples=[
            "what time is it",
            "what's the date",
            "tell me the time",
            "what day is it",
        ],
        handler_agent="system_agent",
    ),
    
    # Web/Search
    IntentDefinition(
        name="SEARCH_WEB",
        description="Search the web for information",
        examples=[
            "search for Python tutorials",
            "google machine learning",
            "look up weather in NYC",
        ],
        required_slots=["query"],
        handler_agent="system_agent",
    ),
    IntentDefinition(
        name="OPEN_URL",
        description="Open a specific URL in browser",
        examples=[
            "open github.com",
            "go to google.com",
            "browse stackoverflow.com",
        ],
        required_slots=["url"],
        handler_agent="system_agent",
    ),
    
    # Volume Control
    IntentDefinition(
        name="CONTROL_VOLUME",
        description="Control system volume",
        examples=[
            "set volume to 50",
            "turn up the volume",
            "mute",
            "unmute",
            "volume down",
        ],
        optional_slots=["level", "action"],  # action: up, down, mute, unmute
        handler_agent="system_agent",
    ),
    
    # Brightness
    IntentDefinition(
        name="CONTROL_BRIGHTNESS",
        description="Control screen brightness",
        examples=[
            "set brightness to 80%",
            "dim the screen",
            "brighten the display",
        ],
        optional_slots=["level", "action"],
        handler_agent="system_agent",
    ),
    
    # Reminders
    IntentDefinition(
        name="SET_REMINDER",
        description="Set a reminder for later",
        examples=[
            "remind me to call mom at 5pm",
            "set a reminder for the meeting",
            "remind me about the deadline tomorrow",
        ],
        required_slots=["task"],
        optional_slots=["time", "date"],
        handler_agent="memory_agent",
    ),
    
    # General Questions
    IntentDefinition(
        name="GENERAL_QUESTION",
        description="Answer a general knowledge question",
        examples=[
            "what is the capital of France",
            "how do I make pasta",
            "explain quantum computing",
        ],
        required_slots=["question"],
        handler_agent="intent_agent",
    ),
    
    # Conversation
    IntentDefinition(
        name="GREETING",
        description="Respond to a greeting",
        examples=[
            "hello",
            "hi VESPER",
            "good morning",
            "hey there",
        ],
        handler_agent="intent_agent",
    ),
    IntentDefinition(
        name="GOODBYE",
        description="End the conversation",
        examples=[
            "goodbye",
            "bye",
            "see you later",
            "that's all",
        ],
        handler_agent="intent_agent",
    ),
    IntentDefinition(
        name="HELP",
        description="Show available commands or help",
        examples=[
            "what can you do",
            "help",
            "show me what you can do",
            "list commands",
        ],
        handler_agent="intent_agent",
    ),
    IntentDefinition(
        name="THANKS",
        description="Acknowledge user thanks",
        examples=[
            "thank you",
            "thanks",
            "appreciate it",
        ],
        handler_agent="intent_agent",
    ),
    
    # System Control
    IntentDefinition(
        name="SYSTEM_CONTROL",
        description="Control system power state",
        examples=[
            "lock the screen",
            "put computer to sleep",
            "restart",
            "shutdown",
        ],
        required_slots=["action"],  # lock, sleep, restart, shutdown
        handler_agent="system_agent",
    ),
    
    # Vision Control
    IntentDefinition(
        name="START_VISION",
        description="Start the camera and enable gesture/face recognition",
        examples=[
            "start vision",
            "enable vision",
            "turn on the camera",
            "start camera",
            "enable gesture recognition",
            "start face detection",
            "activate vision",
        ],
        handler_agent="vision_agent",
    ),
    IntentDefinition(
        name="STOP_VISION",
        description="Stop the camera and disable gesture/face recognition",
        examples=[
            "stop vision",
            "disable vision",
            "turn off the camera",
            "stop camera",
            "disable gesture recognition",
            "stop face detection",
            "deactivate vision",
        ],
        handler_agent="vision_agent",
    ),
    IntentDefinition(
        name="ENROLL_FACE",
        description="Enroll a person's face for recognition",
        examples=[
            "enroll my face",
            "save my face as John",
            "remember my face",
            "add face for Alice",
            "register face",
        ],
        optional_slots=["name"],
        handler_agent="vision_agent",
    ),
]


# =============================================================================
# NLU Provider Interface
# =============================================================================

class NLUProvider(ABC):
    """Abstract interface for NLU providers."""
    
    @abstractmethod
    async def initialize(self, intents: List[IntentDefinition]) -> None:
        """Initialize with intent definitions."""
        pass
    
    @abstractmethod
    async def process(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process text and extract intent(s).
        
        Returns:
            {
                "is_multi_command": bool,
                "intents": [{"intent": str, "confidence": float, "entities": dict}],
                "execution_mode": "sequential" | "parallel"
            }
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup resources."""
        pass


# =============================================================================
# Gemini NLU Provider
# =============================================================================

class GeminiNLUProvider(NLUProvider):
    """
    NLU provider using Google's Gemini models.
    
    Features:
        - Structured JSON output
        - Multi-intent detection
        - Entity normalization
        - Confidence scoring
    
    Requires:
        - GEMINI_API_KEY environment variable
        - google-genai package installed
    """
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        max_retries: int = 2,
    ):
        self._model_name = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._intents: List[IntentDefinition] = []
        self._client = None
        self._available = False
        self._last_call_success: Optional[bool] = None
    
    async def initialize(self, intents: List[IntentDefinition]) -> None:
        """Initialize Gemini client."""
        self._intents = intents
        
        api_key = get_gemini_api_key()
        if not api_key:
            logger.warning("Gemini API key not set - Gemini NLU unavailable")
            return
        
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._available = True
            logger.info(f"Gemini NLU initialized: model={self._model_name}")
        except ImportError:
            logger.error("google-genai not installed: pip install google-genai")
    
    @property
    def is_available(self) -> bool:
        """Check if provider is ready."""
        return self._available and self._client is not None

    @property
    def last_call_success(self) -> Optional[bool]:
        """Return whether the last API call succeeded (None if never called)."""
        return self._last_call_success
    
    async def process(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process text using Gemini."""
        if not self.is_available:
            return self._empty_result()
        
        # Build prompts
        intent_dicts = [i.to_dict() for i in self._intents]
        system_prompt = IntentPrompts.get_system_prompt(intent_dicts)
        user_prompt = IntentPrompts.get_user_prompt(text, context)
        
        # Retry loop
        for attempt in range(self._max_retries + 1):
            try:
                result = await self._call_gemini(system_prompt, user_prompt)
                log_prompt_usage(IntentPrompts.CURRENT_VERSION, success=True)
                self._last_call_success = True
                return result
            except Exception as e:
                logger.warning(f"Gemini attempt {attempt + 1} failed: {e}")
                if attempt == self._max_retries:
                    log_prompt_usage(IntentPrompts.CURRENT_VERSION, success=False)
                    self._last_call_success = False
                    return self._empty_result()
                await asyncio.sleep(0.5 * (attempt + 1))  # Backoff
        
        return self._empty_result()
    
    async def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """Make the actual Gemini API call."""
        from google.genai import types
        
        # Combine prompts for single-turn
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        
        # Run in executor for async
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self._model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=self._temperature,
                    response_mime_type="application/json",
                )
            )
        )
        
        # Parse response
        result = json.loads(response.text)
        
        # Validate structure
        if "intents" not in result:
            result["intents"] = []
        if "is_multi_command" not in result:
            result["is_multi_command"] = len(result["intents"]) > 1
        if "execution_mode" not in result:
            result["execution_mode"] = "sequential"
        
        logger.debug(f"Gemini result: {len(result['intents'])} intent(s)")
        
        return result
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for failures."""
        return {
            "is_multi_command": False,
            "intents": [],
            "execution_mode": "sequential",
        }
    
    async def shutdown(self) -> None:
        """Cleanup Gemini client."""
        self._client = None
        self._available = False


# =============================================================================
# OpenAI NLU Provider
# =============================================================================

class OpenAINLUProvider(NLUProvider):
    """
    NLU provider using OpenAI's GPT models.
    
    Similar to Gemini but uses OpenAI's chat completion API.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_retries: int = 2,
    ):
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._intents: List[IntentDefinition] = []
        self._client = None
        self._available = False
    
    async def initialize(self, intents: List[IntentDefinition]) -> None:
        """Initialize OpenAI client."""
        self._intents = intents
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set - OpenAI NLU unavailable")
            return
        
        try:
            import openai
            self._client = openai.AsyncOpenAI(api_key=api_key)
            self._available = True
            logger.info(f"OpenAI NLU initialized: model={self._model}")
        except ImportError:
            logger.error("openai not installed: pip install openai")
    
    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None
    
    async def process(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process text using OpenAI."""
        if not self.is_available:
            return {"is_multi_command": False, "intents": [], "execution_mode": "sequential"}
        
        intent_dicts = [i.to_dict() for i in self._intents]
        system_prompt = IntentPrompts.get_system_prompt(intent_dicts)
        user_prompt = IntentPrompts.get_user_prompt(text, context)
        
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                )
                
                result = json.loads(response.choices[0].message.content)
                log_prompt_usage(IntentPrompts.CURRENT_VERSION, success=True)
                return result
                
            except Exception as e:
                logger.warning(f"OpenAI attempt {attempt + 1} failed: {e}")
                if attempt == self._max_retries:
                    log_prompt_usage(IntentPrompts.CURRENT_VERSION, success=False)
                    break
                await asyncio.sleep(0.5 * (attempt + 1))
        
        return {"is_multi_command": False, "intents": [], "execution_mode": "sequential"}
    
    async def shutdown(self) -> None:
        self._client = None
        self._available = False


# =============================================================================
# Ollama NLU Provider (local open-source models)
# =============================================================================

class OllamaNLUProvider(NLUProvider):
    """NLU provider backed by a local Ollama model."""

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        endpoint: str = "http://127.0.0.1:11434/api/generate",
        temperature: float = 0.2,
        max_retries: int = 1,
    ):
        self._model = model
        self._endpoint = endpoint
        self._temperature = temperature
        self._max_retries = max_retries
        self._intents: List[IntentDefinition] = []
        self._available = False

    async def initialize(self, intents: List[IntentDefinition]) -> None:
        self._intents = intents
        self._available = await self._check_ollama_health()
        if self._available:
            logger.info(f"Ollama NLU initialized: model={self._model}")
        else:
            logger.warning("Ollama not reachable. Install/start Ollama to enable local NLU.")

    @property
    def is_available(self) -> bool:
        return self._available

    async def process(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._available:
            return {"is_multi_command": False, "intents": [], "execution_mode": "sequential"}

        intent_dicts = [i.to_dict() for i in self._intents]
        system_prompt = IntentPrompts.get_system_prompt(intent_dicts)
        user_prompt = IntentPrompts.get_user_prompt(text, context)
        full_prompt = (
            f"{system_prompt}\n\n---\n\n{user_prompt}\n\n"
            "Return strictly valid JSON with keys: is_multi_command, intents, execution_mode."
        )

        for attempt in range(self._max_retries + 1):
            try:
                result = await self._call_ollama(full_prompt)
                if "intents" not in result:
                    result["intents"] = []
                if "is_multi_command" not in result:
                    result["is_multi_command"] = len(result["intents"]) > 1
                if "execution_mode" not in result:
                    result["execution_mode"] = "sequential"
                return result
            except Exception as exc:
                logger.warning(f"Ollama attempt {attempt + 1} failed: {exc}")
                if attempt == self._max_retries:
                    break
                await asyncio.sleep(0.5 * (attempt + 1))

        return {"is_multi_command": False, "intents": [], "execution_mode": "sequential"}

    async def shutdown(self) -> None:
        self._available = False

    async def _check_ollama_health(self) -> bool:
        def _run() -> bool:
            req = urlrequest.Request("http://127.0.0.1:11434/api/tags", method="GET")
            with urlrequest.urlopen(req, timeout=2) as resp:
                return resp.status == 200

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _run)
        except Exception:
            return False

    async def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self._temperature},
        }

        def _run() -> Dict[str, Any]:
            req = urlrequest.Request(
                self._endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=45) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            raw = str(body.get("response", "")).strip()
            if not raw:
                raise RuntimeError("Ollama returned empty response")

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    return json.loads(raw[start : end + 1])
                raise

        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, _run)
        except urlerror.URLError as exc:
            raise RuntimeError(f"Ollama network error: {exc}") from exc


# =============================================================================
# Pattern Matcher Provider (Fallback)
# =============================================================================

class PatternMatcherProvider(NLUProvider):
    """
    Pattern-based fallback NLU.
    
    Used when LLM providers are unavailable.
    Lower confidence but always works.
    """
    
    def __init__(self):
        self._intents: List[IntentDefinition] = []
    
    async def initialize(self, intents: List[IntentDefinition]) -> None:
        self._intents = intents
        logger.info("Pattern matcher NLU initialized (fallback mode)")
    
    @property
    def is_available(self) -> bool:
        return True
    
    async def process(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process using pattern matching."""
        matches = FallbackPatterns.match(text)
        
        return {
            "is_multi_command": len(matches) > 1,
            "intents": matches,
            "execution_mode": "sequential",
        }
    
    async def shutdown(self) -> None:
        pass


# =============================================================================
# Intent Agent
# =============================================================================

class IntentAgent(BaseAgent):
    """
    Agent responsible for understanding user intent using LLMs.
    
    Features:
        - LLM-based intent extraction (Gemini/OpenAI)
        - Multi-command sequence detection
        - Intelligent entity extraction
        - Pattern-based fallback
        - Confidence thresholds
    
    Configuration (in settings.yaml):
        intent:
            provider: gemini  # gemini, openai, pattern
            confidence_threshold: 0.7
            gemini:
                model: gemini-2.0-flash
                temperature: 0.2
            openai:
                model: gpt-4o-mini
                temperature: 0.2
    
    Events Consumed:
        - VoiceInputEvent
    
    Events Produced:
        - IntentRecognizedEvent (single intent)
        - MultiIntentEvent (multiple intents)
        - IntentUnknownEvent (no match)
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="IntentAgent", event_bus=event_bus, config=config)
        
        # NLU providers (primary + fallback)
        self._primary_provider: Optional[NLUProvider] = None
        self._fallback_provider: PatternMatcherProvider = PatternMatcherProvider()
        
        # Intent catalog
        self._intents: List[IntentDefinition] = []
        
        # Configuration
        self._confidence_threshold = self._get_config(
            "intent.confidence_threshold", 0.6
        )
        self._last_provider_success: Optional[bool] = None
        self._reasoning_engine: Optional[ReasoningEngine] = None
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """Define agent capabilities."""
        return [
            AgentCapability(
                name="intent_recognition",
                description="Classify user intent from text using LLM",
                input_events=["VoiceInputEvent"],
                output_events=["IntentRecognizedEvent", "MultiIntentEvent", "IntentUnknownEvent"],
            ),
            AgentCapability(
                name="entity_extraction",
                description="Extract structured entities from user text",
                input_events=["VoiceInputEvent"],
                output_events=["IntentRecognizedEvent", "MultiIntentEvent"],
            ),
            AgentCapability(
                name="multi_command_detection",
                description="Detect and parse compound commands",
                input_events=["VoiceInputEvent"],
                output_events=["MultiIntentEvent"],
            ),
        ]

    def is_healthy(self) -> bool:
        """Return True if the configured NLU provider is healthy."""
        provider_name = self._get_config("intent.provider", "gemini")
        if provider_name != "gemini":
            return True

        api_key = get_gemini_api_key(self._get_config)
        if not api_key:
            return False

        if isinstance(self._primary_provider, GeminiNLUProvider):
            if not self._primary_provider.is_available:
                return False
            last_success = self._primary_provider.last_call_success
            return last_success is not False

        return True
    
    async def _setup(self) -> None:
        """Initialize the intent agent."""
        # Load intents
        self._intents = self._load_intents()
        
        # Initialize providers
        await self._initialize_providers()
        if self._is_truthy(self._get_config("reasoning.enabled", False)):
            self._reasoning_engine = ReasoningEngine(event_bus=self._event_bus, config=self._config)
        else:
            self._reasoning_engine = None

        # Subscribe to voice input
        self._subscribe(VoiceInputEvent, self._handle_voice_input)

        provider_name = self._get_config("intent.provider", "pattern")
        self._logger.info(
            f"Intent agent initialized with {len(self._intents)} intents "
            f"using {provider_name} provider"
        )

    async def _teardown(self) -> None:
        """Cleanup resources."""
        if self._primary_provider:
            await self._primary_provider.shutdown()
        await self._fallback_provider.shutdown()
        self._reasoning_engine = None
        self._logger.info("Intent agent shutdown complete")

    async def _initialize_providers(self) -> None:
        """Initialize NLU providers based on configuration."""
        provider_name = str(self._get_config("intent.provider", "pattern")).lower()
        
        if provider_name == "gemini":
            self._primary_provider = GeminiNLUProvider(
                model=self._get_config("intent.gemini.model", "gemini-2.0-flash"),
                temperature=self._get_config("intent.gemini.temperature", 0.2),
            )
        elif provider_name == "ollama":
            self._primary_provider = OllamaNLUProvider(
                model=self._get_config("intent.ollama.model", "qwen2.5:7b-instruct"),
                endpoint=self._get_config("intent.ollama.endpoint", "http://127.0.0.1:11434/api/generate"),
                temperature=self._get_config("intent.ollama.temperature", 0.2),
            )
        elif provider_name == "openai":
            self._primary_provider = OpenAINLUProvider(
                model=self._get_config("intent.openai.model", "gpt-4o-mini"),
                temperature=self._get_config("intent.openai.temperature", 0.2),
            )
        else:
            # Use pattern matcher as primary
            self._primary_provider = self._fallback_provider
        
        # Initialize primary
        await self._primary_provider.initialize(self._intents)
        
        # Initialize fallback
        await self._fallback_provider.initialize(self._intents)
    
    def _load_intents(self) -> List[IntentDefinition]:
        """Load intent definitions."""
        intents = list(DEFAULT_INTENTS)
        
        # Load custom intents from config
        custom_intents = self._get_config("intent.custom_intents", [])
        for intent_config in custom_intents:
            intents.append(IntentDefinition(
                name=intent_config.get("name", "CUSTOM"),
                description=intent_config.get("description", ""),
                examples=intent_config.get("examples", []),
                required_slots=intent_config.get("required_slots", []),
                optional_slots=intent_config.get("optional_slots", []),
                handler_agent=intent_config.get("handler_agent", "system_agent"),
            ))
        
        return intents
    
    async def _handle_voice_input(self, event: VoiceInputEvent) -> None:
        """
        Handle voice input and extract intent(s).
        
        Workflow:
        1. Try primary LLM provider
        2. Fall back to pattern matching if LLM fails
        3. Emit appropriate event based on results
        """
        text = event.text
        if not text or not text.strip():
            return

        self._logger.debug(f"Processing: '{text}'")

        if self._reasoning_engine and await self._reasoning_engine.is_complex_request(text):
            self._logger.info("Complex request detected, delegating to ReasoningEngine")
            await self._reasoning_engine.run(user_input=text, correlation_id=event.event_id)
            return
        
        # Get conversation context (TODO: integrate with memory)
        context = await self._get_context()
        
        # Try primary provider
        result = await self._primary_provider.process(text, context)
        self._last_provider_success = bool(result.get("intents")) or (
            isinstance(self._primary_provider, GeminiNLUProvider)
            and self._primary_provider.last_call_success is not False
        )
        
        # Check if we got valid results
        if not result.get("intents"):
            # Fall back to pattern matching
            self._logger.debug("Primary provider failed, using fallback")
            result = await self._fallback_provider.process(text, context)
            if isinstance(self._primary_provider, GeminiNLUProvider):
                self._last_provider_success = self._primary_provider.last_call_success
        
        # Process results
        await self._emit_results(result, text, event.event_id)

    async def _emit_results(
        self,
        result: Dict[str, Any],
        raw_text: str,
        correlation_id: str,
    ) -> None:
        """Emit appropriate events based on extraction results."""
        intents = result.get("intents", [])
        is_multi = result.get("is_multi_command", False)
        execution_mode = result.get("execution_mode", "sequential")
        
        if not intents:
            # No intents recognized
            await self._emit(IntentUnknownEvent(
                raw_text=raw_text,
                suggestions=self._get_suggestions(raw_text),
                source=self._name,
                correlation_id=correlation_id,
            ))
            self._logger.info(f"No intent recognized for: '{raw_text}'")
            return
        
        # Filter by confidence threshold
        valid_intents = [
            i for i in intents
            if i.get("confidence", 0) >= self._confidence_threshold
        ]
        
        if not valid_intents:
            # All intents below threshold
            await self._emit(IntentUnknownEvent(
                raw_text=raw_text,
                suggestions=self._get_suggestions(raw_text),
                source=self._name,
                correlation_id=correlation_id,
            ))
            self._logger.info(
                f"All intents below confidence threshold ({self._confidence_threshold})"
            )
            return
        
        if len(valid_intents) == 1 and not is_multi:
            # Single intent
            intent_data = valid_intents[0]
            await self._emit(IntentRecognizedEvent(
                intent=intent_data.get("intent", "UNKNOWN"),
                confidence=intent_data.get("confidence", 0.5),
                entities=intent_data.get("entities", {}),
                raw_text=raw_text,
                slots=intent_data.get("entities", {}),
                source=self._name,
                correlation_id=correlation_id,
            ))
            self._logger.info(
                f"Intent: {intent_data.get('intent')} "
                f"(confidence: {intent_data.get('confidence', 0):.2f})"
            )
        else:
            # Multiple intents
            await self._emit(MultiIntentEvent(
                intents=valid_intents,
                raw_text=raw_text,
                execution_mode=execution_mode,
                source=self._name,
                correlation_id=correlation_id,
            ))
            intent_names = [i.get("intent") for i in valid_intents]
            self._logger.info(f"Multi-intent: {intent_names} (mode: {execution_mode})")
    
    async def _get_context(self) -> Dict[str, Any]:
        """Get conversation context from memory agent."""
        # TODO: Query memory agent for recent history
        return {}

    @staticmethod
    def _is_truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    
    def _get_suggestions(self, text: str) -> List[str]:
        """Get suggestions for unknown input."""
        return [
            "Try 'open [app name]' to launch an application",
            "Say 'search for [topic]' to search the web",
            "Ask 'what's my CPU usage' for system stats",
            "Say 'help' to see all commands",
        ]
    
    def get_available_intents(self) -> List[str]:
        """Get list of available intent names."""
        return [i.name for i in self._intents]
    
    def get_intent_help(self) -> str:
        """Get help text for all available intents."""
        lines = ["Here's what I can help you with:\n"]
        
        # Group by category
        categories = {
            "Apps": ["OPEN_APP", "CLOSE_APP", "SWITCH_APP"],
            "System": ["GET_SYSTEM_STATS", "GET_TIME", "SYSTEM_CONTROL"],
            "Web": ["SEARCH_WEB", "OPEN_URL"],
            "Controls": ["CONTROL_VOLUME", "CONTROL_BRIGHTNESS"],
            "Vision": ["START_VISION", "STOP_VISION", "ENROLL_FACE"],
            "Reminders": ["SET_REMINDER"],
            "General": ["GENERAL_QUESTION", "HELP"],
        }
        
        for category, intent_names in categories.items():
            category_intents = [i for i in self._intents if i.name in intent_names]
            if category_intents:
                lines.append(f"\n**{category}**")
                for intent in category_intents:
                    lines.append(f"  • {intent.description}")
                    if intent.examples:
                        lines.append(f"    Example: \"{intent.examples[0]}\"")
        
        return "\n".join(lines)
