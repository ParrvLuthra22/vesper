"""
Prompt Templates Module - Versioned prompts for LLM-based intent extraction.

This module contains all prompt templates used by the IntentAgent.
Templates are versioned to track changes and enable A/B testing.

Usage:
    from utils.prompts import IntentPrompts
    
    prompt = IntentPrompts.get_intent_extraction_prompt(
        user_input="Open VS Code and check CPU",
        intents=intent_definitions,
    )

Architecture:
    - All prompts are class methods for easy access
    - Each prompt has a version string for tracking
    - Prompts return the full formatted string ready for LLM

TODO: Add prompt caching
TODO: Add prompt optimization based on usage metrics
TODO: Add multi-language support
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Prompt Version Tracking
# =============================================================================

@dataclass
class PromptVersion:
    """Track prompt versions for logging and A/B testing."""
    name: str
    version: str
    description: str


# =============================================================================
# Intent Extraction Prompts
# =============================================================================

class IntentPrompts:
    """
    Prompt templates for intent extraction.
    
    All prompts are designed to:
    - Return valid JSON only (no markdown, no explanations)
    - Handle multi-command sequences
    - Include confidence scores
    - Extract structured entities
    """
    
    # Version tracking
    INTENT_EXTRACTION_V1 = PromptVersion(
        name="intent_extraction",
        version="1.0.0",
        description="Initial multi-intent extraction prompt",
    )
    
    INTENT_EXTRACTION_V2 = PromptVersion(
        name="intent_extraction",
        version="2.0.0",
        description="Enhanced with better entity extraction and examples",
    )
    
    # Current version in use
    CURRENT_VERSION = INTENT_EXTRACTION_V2
    
    @classmethod
    def get_system_prompt(cls, intents: List[Dict[str, Any]]) -> str:
        """
        Get the system prompt for intent extraction.
        
        Args:
            intents: List of intent definitions with name, description, examples
        
        Returns:
            System prompt string
        """
        intent_docs = cls._format_intent_documentation(intents)
        
        return f"""You are VESPER, an advanced intent extraction system.
Your job is to analyze user commands and extract structured intents.

## AVAILABLE INTENTS

{intent_docs}

## RULES

1. ALWAYS return valid JSON - no markdown, no code blocks, no explanations
2. Detect MULTIPLE intents when user gives compound commands
3. Use "and", "then", "also", "after that" as command separators
4. Extract ALL entities mentioned (app names, metrics, values, etc.)
5. Assign confidence 0.0-1.0 based on clarity
6. Use "unknown" only when no intent matches at all
7. Normalize entity values (e.g., "VS Code" not "vscode")

## OUTPUT FORMAT

Return this EXACT JSON structure:
{{
    "is_multi_command": true/false,
    "intents": [
        {{
            "intent": "INTENT_NAME",
            "confidence": 0.95,
            "entities": {{"key": "value"}},
            "original_segment": "the part of input for this intent"
        }}
    ],
    "execution_mode": "sequential" or "parallel"
}}

Version: {cls.CURRENT_VERSION.version}"""

    @classmethod
    def get_user_prompt(cls, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the user prompt for intent extraction.
        
        Args:
            user_input: The raw user input text
            context: Optional conversation context
        
        Returns:
            User prompt string
        """
        context_section = ""
        if context:
            context_section = f"\n\nCONTEXT:\n{cls._format_context(context)}"
        
        return f"""USER INPUT: "{user_input}"{context_section}

Extract the intent(s) from this input. Remember:
- Check for compound commands (multiple intents)
- Extract all relevant entities
- Assign appropriate confidence scores"""

    @classmethod
    def get_full_prompt(
        cls,
        user_input: str,
        intents: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Get the complete prompt (system + user) for single-prompt models.
        
        Args:
            user_input: The raw user input text
            intents: List of intent definitions
            context: Optional conversation context
        
        Returns:
            Complete prompt string
        """
        system = cls.get_system_prompt(intents)
        user = cls.get_user_prompt(user_input, context)
        
        return f"""{system}

---

{user}"""

    @classmethod
    def _format_intent_documentation(cls, intents: List[Dict[str, Any]]) -> str:
        """Format intent definitions for the prompt."""
        lines = []
        
        for intent in intents:
            name = intent.get("name", "unknown").upper()
            description = intent.get("description", "")
            examples = intent.get("examples", [])
            required_slots = intent.get("required_slots", [])
            optional_slots = intent.get("optional_slots", [])
            
            lines.append(f"### {name}")
            lines.append(f"Description: {description}")
            
            if examples:
                lines.append("Examples:")
                for ex in examples[:3]:  # Limit to 3 examples
                    lines.append(f"  - \"{ex}\"")
            
            if required_slots:
                lines.append(f"Required entities: {', '.join(required_slots)}")
            
            if optional_slots:
                lines.append(f"Optional entities: {', '.join(optional_slots)}")
            
            lines.append("")  # Blank line between intents
        
        return "\n".join(lines)

    @classmethod
    def _format_context(cls, context: Dict[str, Any]) -> str:
        """Format conversation context for the prompt."""
        lines = []
        
        if "last_intent" in context:
            lines.append(f"Last intent: {context['last_intent']}")
        
        if "last_entities" in context:
            lines.append(f"Last entities: {context['last_entities']}")
        
        if "conversation_history" in context:
            history = context["conversation_history"][-3:]  # Last 3 turns
            lines.append("Recent conversation:")
            for turn in history:
                lines.append(f"  - {turn}")
        
        return "\n".join(lines) if lines else "No prior context"


# =============================================================================
# Fallback Patterns
# =============================================================================

class FallbackPatterns:
    """
    Pattern-based fallback when LLM is unavailable.
    
    Uses keyword matching as a last resort.
    """
    
    # Keyword to intent mapping (order matters - first match wins)
    KEYWORD_PATTERNS = [
        # Vision control (must come before app control to prevent "start vision" -> OPEN_APP)
        (["start vision", "enable vision", "vision on", "turn on vision", "activate vision", "start camera", "enable camera"], "START_VISION", []),
        (["stop vision", "disable vision", "vision off", "turn off vision", "deactivate vision", "stop camera", "disable camera"], "STOP_VISION", []),
        (["enroll face", "enroll my face", "save my face", "remember my face", "register my face", "add my face", "save face", "remember face", "register face", "add face", "learn my face", "learn face"], "ENROLL_FACE", ["name"]),
        
        # App control
        (["open", "launch", "start", "run"], "OPEN_APP", ["app"]),
        (["close", "quit", "exit", "kill"], "CLOSE_APP", ["app"]),
        
        # System stats
        (["cpu", "processor", "load"], "GET_SYSTEM_STATS", ["metric"]),
        (["memory", "ram", "usage"], "GET_SYSTEM_STATS", ["metric"]),
        (["disk", "storage", "space"], "GET_SYSTEM_STATS", ["metric"]),
        (["battery", "power", "charging"], "GET_SYSTEM_STATS", ["metric"]),
        
        # Web/Search
        (["search", "google", "look up", "find"], "SEARCH_WEB", ["query"]),
        (["browse", "go to", "navigate"], "OPEN_URL", ["url"]),
        
        # Volume
        (["volume", "sound", "audio"], "CONTROL_VOLUME", ["level", "action"]),
        (["mute", "unmute", "silence"], "CONTROL_VOLUME", ["action"]),
        
        # Time/Date
        (["time", "clock", "hour"], "GET_TIME", []),
        (["date", "day", "today", "calendar"], "GET_DATE", []),
        
        # Reminders
        (["remind", "reminder", "alert"], "SET_REMINDER", ["task", "time"]),
        (["schedule", "appointment"], "SET_REMINDER", ["task", "time"]),
        
        # Conversation
        (["hello", "hi", "hey", "greetings"], "GREETING", []),
        (["bye", "goodbye", "see you", "later"], "GOODBYE", []),
        (["help", "what can you", "commands"], "HELP", []),
        (["thank", "thanks"], "THANKS", []),
        
        # System control
        (["sleep", "lock", "shutdown", "restart"], "SYSTEM_CONTROL", ["action"]),
        (["brightness", "screen"], "CONTROL_BRIGHTNESS", ["level"]),
    ]
    
    @classmethod
    def match(cls, text: str) -> List[Dict[str, Any]]:
        """
        Match text against keyword patterns.
        
        Args:
            text: User input text
        
        Returns:
            List of matched intents with entities
        """
        text_lower = text.lower()
        results = []
        
        # Check for compound commands first
        segments = cls._split_compound(text_lower)
        
        for segment in segments:
            intent_match = cls._match_segment(segment)
            if intent_match:
                results.append(intent_match)
        
        # If no matches, return unknown
        if not results:
            results.append({
                "intent": "UNKNOWN",
                "confidence": 0.3,
                "entities": {},
                "original_segment": text,
            })
        
        return results
    
    @classmethod
    def _split_compound(cls, text: str) -> List[str]:
        """Split compound commands into segments."""
        # Common separators for compound commands
        separators = [
            " and ",
            " then ",
            " also ",
            " after that ",
            ", then ",
            "; ",
        ]
        
        segments = [text]
        
        for sep in separators:
            new_segments = []
            for segment in segments:
                new_segments.extend(segment.split(sep))
            segments = new_segments
        
        # Clean up
        return [s.strip() for s in segments if s.strip()]
    
    @classmethod
    def _match_segment(cls, segment: str) -> Optional[Dict[str, Any]]:
        """Match a single segment against patterns."""
        for keywords, intent, entity_hints in cls.KEYWORD_PATTERNS:
            for keyword in keywords:
                if keyword in segment:
                    # Extract basic entities
                    entities = cls._extract_entities(segment, intent, entity_hints)
                    
                    return {
                        "intent": intent,
                        "confidence": 0.6,  # Lower confidence for pattern match
                        "entities": entities,
                        "original_segment": segment,
                    }
        
        return None
    
    @classmethod
    def _extract_entities(
        cls,
        segment: str,
        intent: str,
        entity_hints: List[str],
    ) -> Dict[str, Any]:
        """Extract basic entities from segment."""
        entities = {}
        
        if intent == "OPEN_APP":
            # Extract app name (everything after trigger word)
            for trigger in ["open ", "launch ", "start ", "run "]:
                if trigger in segment:
                    app_name = segment.split(trigger, 1)[1].strip()
                    entities["app"] = cls._normalize_app_name(app_name)
                    break
        
        elif intent == "GET_SYSTEM_STATS":
            # Determine which metric
            if "cpu" in segment or "processor" in segment:
                entities["metric"] = "cpu"
            elif "memory" in segment or "ram" in segment:
                entities["metric"] = "memory"
            elif "disk" in segment or "storage" in segment:
                entities["metric"] = "disk"
            elif "battery" in segment:
                entities["metric"] = "battery"
        
        elif intent == "SEARCH_WEB":
            # Extract search query
            for trigger in ["search for ", "search ", "google ", "look up ", "find "]:
                if trigger in segment:
                    query = segment.split(trigger, 1)[1].strip()
                    entities["query"] = query
                    break
        
        elif intent == "CONTROL_VOLUME":
            # Determine action
            if "mute" in segment:
                entities["action"] = "mute"
            elif "unmute" in segment:
                entities["action"] = "unmute"
            elif "up" in segment or "louder" in segment:
                entities["action"] = "up"
            elif "down" in segment or "quieter" in segment:
                entities["action"] = "down"
            
            # Try to extract level
            import re
            level_match = re.search(r'(\d+)(?:%|\s*percent)?', segment)
            if level_match:
                entities["level"] = int(level_match.group(1))
        
        elif intent == "ENROLL_FACE":
            # Extract name for face enrollment
            # Patterns like "enroll my face as John" or "save my face as Sarah"
            import re
            # Match "as <name>" pattern
            name_match = re.search(r'\bas\s+([a-zA-Z][a-zA-Z\s]*?)(?:\.|,|$)', segment, re.IGNORECASE)
            if name_match:
                name = name_match.group(1).strip().rstrip('.')
                name = cls._normalize_user_name(name)
                entities["name"] = name
            # Also try "for <name>" pattern
            elif "for " in segment:
                for_match = re.search(r'\bfor\s+([a-zA-Z][a-zA-Z\s]*?)(?:\.|,|$)', segment, re.IGNORECASE)
                if for_match:
                    name = for_match.group(1).strip().rstrip('.')
                    name = cls._normalize_user_name(name)
                    entities["name"] = name
        
        return entities
    
    @classmethod
    def _normalize_user_name(cls, name: str) -> str:
        """Normalize user name, fixing common transcription errors."""
        # Common transcription corrections
        name_corrections = {
            "purv": "Parrv",
            "perv": "Parrv",
            "parv": "Parrv",
            "prav": "Parrv",
            "prev": "Parrv",
        }
        
        name_lower = name.lower().strip()
        if name_lower in name_corrections:
            return name_corrections[name_lower]
        
        # Default: capitalize properly
        return name.title()
    
    @classmethod
    def _normalize_app_name(cls, name: str) -> str:
        """Normalize app name to standard format."""
        # Common normalizations
        normalizations = {
            "vscode": "Visual Studio Code",
            "vs code": "Visual Studio Code",
            "code": "Visual Studio Code",
            "chrome": "Google Chrome",
            "firefox": "Firefox",
            "safari": "Safari",
            "slack": "Slack",
            "spotify": "Spotify",
            "terminal": "Terminal",
            "finder": "Finder",
            "notes": "Notes",
            "mail": "Mail",
            "messages": "Messages",
            "calendar": "Calendar",
            "music": "Music",
            "photos": "Photos",
        }
        
        name_lower = name.lower()
        return normalizations.get(name_lower, name.title())


# =============================================================================
# Prompt Utilities
# =============================================================================

def log_prompt_usage(prompt_version: PromptVersion, success: bool) -> None:
    """Log prompt usage for analytics."""
    status = "success" if success else "failure"
    logger.debug(f"Prompt {prompt_version.name} v{prompt_version.version}: {status}")


def get_current_prompt_version() -> str:
    """Get the current prompt version string."""
    return IntentPrompts.CURRENT_VERSION.version
