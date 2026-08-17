"""
Tool relevance selection — choose which subset of the registry is worth
sending to the planner for one turn.

Why this exists: the whole tool schema is re-sent on *every* planning call,
and it is by far the largest fixed cost in the prompt (measured on this
repo: 2,145 tokens of tool schema against 813 tokens of persona). A
six-iteration turn therefore paid for the full catalog six times over, which
is what pushed a normal multi-step request through Groq's 8k tokens/minute
free-tier ceiling.

The filter is deliberately biased toward recall over savings:

    1. A small CORE set is always sent — the tools a turn may need without
       ever naming them (the clock, and the web-search escape hatch).
    2. A tool matches if any of its index terms (name parts, group
       triggers, explicit synonyms) appears in the user's text.
    3. A match promotes its *whole group*, not just the one tool —
       "commit this" needs git_status and git_diff as much as git_commit.
       Groups are finer than the registry's `category`: "system" alone is
       15 tools, so expanding it wholesale gave back most of the savings.
       Volume and screen brightness have no business travelling together.
    4. If nothing matches at all, the full catalog is sent unchanged. A
       vague turn is exactly the turn where guessing is most likely to be
       wrong, so it is never the turn we economize on.

Rule 4 plus the Planner's widen-on-miss retry (see Planner._run_iteration)
means a filtering mistake costs an extra call, never a wrong answer.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from tools.registry import ToolRegistry, ToolSpec

#: Always sent, regardless of the user's text. Kept tiny (~270 tokens) —
#: these are the tools a model reaches for without the user naming them:
#: the current time/date for anything scheduling-shaped, and web search as
#: the general "I don't know this" escape hatch.
CORE_TOOL_NAMES: frozenset = frozenset({"get_time", "get_date", "search_web"})

#: Name fragments too generic to identify a tool on their own — they appear
#: across many tools and never disambiguate. Dropped from the index.
_GENERIC_NAME_PARTS: frozenset = frozenset({"get", "set", "current"})

#: Selection groups: finer than ToolSpec.category, and the unit that gets
#: promoted when a tool matches. Any tool not listed here falls back to its
#: registry category as its group, which is what lets MCP-provided tools
#: (category "mcp:<server>") group themselves per server with no entry here.
TOOL_GROUPS: Dict[str, str] = {
    "open_app": "apps", "close_app": "apps", "focus_app": "apps", "list_apps": "apps",
    "set_volume": "audio", "get_volume": "audio", "mute": "audio",
    "set_brightness": "display", "take_screenshot": "display", "lock_screen": "display",
    "get_battery": "sysinfo", "get_time": "sysinfo", "get_date": "sysinfo",
    "system_info": "sysinfo",
    "show_notification": "notify",
    "current_weather": "weather",
}

#: Words that promote an entire group. A hit here is the main driver of the
#: filter; per-tool synonyms below only add coverage the group misses.
GROUP_TRIGGERS: Dict[str, Tuple[str, ...]] = {
    "apps": (
        "app", "application", "program", "window", "launch", "quit", "switch",
        "open", "close", "focus", "running",
    ),
    "audio": (
        "volume", "sound", "audio", "loud", "quiet", "silence", "mute", "unmute",
        "louder", "softer", "speaker", "headphones",
    ),
    "display": (
        "brightness", "bright", "dim", "screen", "display", "lock", "unlock",
        "screenshot", "capture", "monitor",
    ),
    "sysinfo": (
        "battery", "charge", "charging", "power", "percent",
        "time", "clock", "date", "today", "tomorrow", "day", "month", "year",
        "mac", "macbook", "laptop", "system", "cpu", "memory", "ram", "disk",
        "storage", "uptime", "specs",
    ),
    "notify": ("notify", "notification", "alert", "remind", "reminder"),
    "weather": (
        "weather", "temperature", "forecast", "rain", "raining", "umbrella",
        "sunny", "cold", "hot", "warm", "outside", "degrees",
    ),
    "web": (
        "search", "google", "look up", "lookup", "web", "internet", "online",
        "browse", "browser", "url", "link", "site", "website", "news",
        "article", "wikipedia",
    ),
    "dev": (
        "git", "commit", "diff", "staged", "unstaged", "repo", "repository",
        "branch", "checkout", "push", "pull", "merge",
        "test", "pytest", "suite", "failing", "coverage",
        "editor", "vscode", "code", "codebase", "source",
    ),
    "creator": (
        "research", "investigate", "deep dive", "report", "brief",
        "script", "screenplay", "draft", "write", "outline", "video",
        "shell", "terminal", "bash", "command", "applescript", "automate",
        "automation",
    ),
}

#: Extra per-tool triggers, for phrasings the tool's own name misses.
TOOL_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "open_app": ("start", "run", "fire up", "bring up"),
    "close_app": ("kill", "exit", "shut", "stop"),
    "focus_app": ("front", "foreground", "bring to front"),
    "list_apps": ("running", "what's open", "whats open"),
    "set_volume": ("turn up", "turn down", "louder", "softer"),
    "mute": ("shut up", "silence"),
    "set_brightness": ("darker", "brighter"),
    "lock_screen": ("secure", "away", "afk"),
    "take_screenshot": ("screen shot", "grab", "snap"),
    "get_battery": ("percent", "percentage", "juice"),
    "system_info": ("hardware", "how much memory", "free space"),
    "search_web": ("find out", "who is", "what is", "latest"),
    "open_url": ("go to", "visit"),
    "current_weather": ("jacket", "coat"),
    "git_commit": ("save changes", "check in"),
    "git_status": ("changes", "modified", "dirty"),
    "git_diff": ("changed", "what changed"),
    "run_tests": ("green", "red", "broken"),
    "open_in_editor": ("edit", "open file"),
    "research": ("dig into", "find everything"),
    "write_script": ("screenplay", "voiceover", "narration"),
    "run_shell": ("cli", "run command"),
    "run_applescript": ("osascript",),
}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation to spaces, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _word_set(normalized: str) -> Set[str]:
    """
    Words in the text, plus a naive singular for each plural, so "apps"
    matches an "app" trigger and "tests" matches "test".
    """
    words = set(normalized.split())
    for word in list(words):
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            words.add(word[:-1])
    return words


def group_of(spec: ToolSpec) -> str:
    """
    The selection group a tool belongs to — its explicit TOOL_GROUPS entry,
    or its registry category as a fallback (which is what groups each MCP
    server's tools together under "mcp:<server>").
    """
    return TOOL_GROUPS.get(spec.name, spec.category)


def _index_terms(spec: ToolSpec) -> Tuple[Set[str], Tuple[str, ...]]:
    """
    Index terms for one tool, split into single words (matched against the
    text's word set) and multi-word phrases (matched as substrings).

    Terms come from the tool's own name, its group's triggers, the group
    string itself (which is what gives MCP tools — group "mcp:gmail" — a
    usable index with no hand-maintained entry), and explicit synonyms.
    """
    group = group_of(spec)
    raw: List[str] = []
    raw.extend(part for part in spec.name.split("_") if part not in _GENERIC_NAME_PARTS)
    raw.extend(re.split(r"[:_\-]", group))
    raw.extend(GROUP_TRIGGERS.get(group, ()))
    raw.extend(TOOL_SYNONYMS.get(spec.name, ()))

    words: Set[str] = set()
    phrases: List[str] = []
    for term in raw:
        term = term.strip().lower()
        if len(term) < 2:
            continue
        (phrases.append(term) if " " in term else words.add(term))
    return words, tuple(phrases)


def _matches(spec: ToolSpec, normalized: str, text_words: Set[str]) -> bool:
    words, phrases = _index_terms(spec)
    if words & text_words:
        return True
    return any(phrase in normalized for phrase in phrases)


def select_tools(
    registry: ToolRegistry,
    user_text: str,
    *,
    enabled_only: bool = True,
    core_names: Iterable[str] = CORE_TOOL_NAMES,
) -> Tuple[List[ToolSpec], str]:
    """
    Pick the tools worth sending for `user_text`.

    Returns `(tools, reason)`. `reason` is a short tag for logging:
    "filtered" when a real subset was chosen, or "full:<why>" when the
    complete catalog is being sent (no text to match on, or nothing
    matched — see rule 4 in the module docstring).
    """
    all_tools = registry.list_all(enabled_only=enabled_only)
    normalized = _normalize(user_text or "")
    if not normalized:
        return all_tools, "full:no-text"

    text_words = _word_set(normalized)
    matched_groups: Set[str] = {
        group_of(spec) for spec in all_tools if _matches(spec, normalized, text_words)
    }
    if not matched_groups:
        return all_tools, "full:no-match"

    core = set(core_names)
    selected = [
        spec for spec in all_tools if group_of(spec) in matched_groups or spec.name in core
    ]
    if len(selected) >= len(all_tools):
        return all_tools, "full:all-matched"
    return selected, "filtered"


def to_schema(tools: Sequence[ToolSpec]) -> List[dict]:
    """Render selected ToolSpecs into the OpenAI-style `tools` array."""
    return [spec.to_openai_function() for spec in tools]
