"""
Action-claim verifier — catches replies that say a state-changing action was done
when no matching tool call succeeded this turn ("Consider it noted, Sir" after
"Remind me to call the dentist" with no `add_reminder` call).

Rule-based on purpose: it runs on every final reply, so it must cost nothing —
no extra LLM call, no latency. Precision matters more than recall: a false
positive costs a retry, a false negative is the status quo.

Two kinds of claim:

  * EXPLICIT — the reply itself asserts completion ("I've set a reminder",
    "archived", "event created"). Flagged regardless of what the user asked.
  * ACK idioms — vague acknowledgements ("Consider it done", "Noted", "Done")
    flagged only when the USER's request was a request for a state change, and
    then against the tools that request needed.

A claim is "backed" when any tool in its `satisfied_by` set succeeded this turn.
Sentences that are questions, offers, hedges, negations or failure reports are
never claims ("Shall I set a reminder?", "I couldn't open Safari").

Deliberately not covered: claims about actions that no tool could ever perform
other than via `satisfied_by == ()` (e.g. "I've sent the email" — there is no
send tool, so that is always unbacked); and any LLM-judge fallback (an extra
Groq call per turn would spend the 8k TPM budget the router paces so carefully).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Tuple

NUDGE = (
    "System note: your last reply claimed an action was done, but no matching tool "
    "call succeeded this turn. Either call the appropriate tool now, or tell the "
    "user plainly that you could not do it. Never claim an action you did not perform."
)

HONEST_REPLY = (
    "I didn't actually do that, Sir — no action was taken. "
    "I can try again, or tell you what I'm able to do."
)

CORRECTION_NOTICE = "Correction: that reply claimed an action that did not happen."


@dataclass(frozen=True)
class Claim:
    category: str
    sentence: str
    satisfied_by: FrozenSet[str]
    kind: str  # "explicit" | "ack"


# ---------------------------------------------------------------- sentences

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

#: A sentence matching any of these is not asserting a completed action.
_NOT_A_CLAIM = re.compile(
    r"""(
        \?\s*$ | \bshall\ i\b | \bshould\ i\b | \bwould\ you\ like\b | \bdo\ you\ want\b |
        \bif\ you(?:'d|\ would)?\ (?:like|wish|want|prefer)\b | \blet\ me\ know\b |
        \bi\ (?:can|could|would|might|may)\b | \bi'd\b | \bi\ (?:won't|will\ not)\b |
        \b(?:nothing|none|never|not|unable|cannot|can't|couldn't|could\ not|didn't|did\ not|haven't|have\ not|
           not\ able|wasn't|weren't|isn't|aren't|unfortunately|failed|failure|error|
           denied|declined|refused|no\ such|i'm\ afraid|sorry|i\ don't|no\ way\ to)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_FIRST = r"(?:i(?:'ve|'m|'ll| have| am| will)?\s+(?:just\s+|now\s+|already\s+|also\s+|duly\s+)*)"


def _split(reply: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(reply or "") if s.strip()]


# ---------------------------------------------------------------- categories

def _rx(*patterns: str) -> Tuple["re.Pattern[str]", ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


# (category, explicit-claim regexes, satisfied_by tool names)
_EXPLICIT: Tuple[Tuple[str, Tuple["re.Pattern[str]", ...], Tuple[str, ...]], ...] = (
    ("reminder", _rx(
        rf"\b{_FIRST}?(?:set|added|created|scheduled|saved|made|put)\b[^.?!]{{0,40}}\breminder\b",
        r"\breminder\b[^.?!]{0,30}\b(?:is|has\s+been|was|'s)\s+(?:now\s+)?(?:set|added|created|saved|scheduled)\b",
        r"^\W*(?:the\s+|your\s+|a\s+)?reminder\s+(?:now\s+)?(?:set|added|created|saved|scheduled)\b",
        r"\bi(?:'ll|\s+will)\s+remind\s+you\b",
        rf"\b{_FIRST}(?:noted|made\s+a\s+note)\b",
    ), ("add_reminder", "create_event")),
    ("calendar event", _rx(
        rf"\b{_FIRST}?(?:scheduled|booked|added|created|put)\b[^.?!]{{0,60}}\b(?:calendar|event|meeting|appointment)\b",
        r"\b(?:event|meeting|appointment)\b[^.?!]{0,30}\b(?:is|has\s+been|was)\s+(?:now\s+)?(?:scheduled|created|added|booked)\b",
    ), ("create_event",)),
    ("email draft", _rx(
        rf"\b{_FIRST}?drafted\b",
        r"\b(?:draft|reply)\b[^.?!]{0,30}\b(?:is|has\s+been|was)\s+(?:now\s+)?(?:ready|saved|created|waiting|prepared)\b",
        rf"\b{_FIRST}(?:written|prepared|saved)\s+(?:a\s+|the\s+)?(?:reply|response|draft)\b",
    ), ("draft_reply",)),
    ("email archive", _rx(
        rf"\b{_FIRST}?archived\b",
        r"\b(?:email|message|thread)\b[^.?!]{0,30}\b(?:is|has\s+been|was)\s+(?:now\s+)?archived\b",
    ), ("archive",)),
    ("mark read", _rx(
        rf"\b{_FIRST}?marked\b[^.?!]{{0,40}}\bas\s+read\b",
    ), ("mark_read",)),
    ("message sent", _rx(
        rf"\b{_FIRST}?sent\b[^.?!]{{0,40}}\b(?:email|e-mail|message|reply|mail|text|dm)\b",
        r"\b(?:email|e-mail|message|reply|mail|text)\b[^.?!]{0,20}\b(?:has\s+been|was|is)\s+(?:now\s+)?sent\b",
        rf"\b{_FIRST}?(?:posted|replied)\b[^.?!]{{0,30}}\b(?:slack|channel|thread|discord)\b",
    ), ("post_message", "reply_thread")),
    ("note saved", _rx(
        rf"\b{_FIRST}?saved\b[^.?!]{{0,30}}\bnote\b",
        r"\bnote\b[^.?!]{0,20}\b(?:is|has\s+been|was)\s+(?:now\s+)?(?:saved|added|created)\b",
        r"\b(?:added|saved)\b[^.?!]{0,30}\bto\s+(?:your\s+)?notes\b",
    ), ()),
    ("app opened", _rx(
        rf"^\W*(?:opened|launched|focused|opening|launching)\s+\S",
        rf"\b{_FIRST}(?:opened|launched|focused|am\s+opening|am\s+launching|opening|launching)\s+\S",
        r"\b(?:app|application|browser|window|page|url)\b[^.?!]{0,25}\b(?:is|has\s+been|was)\s+(?:now\s+)?(?:open|opened|launched)\b",
    ), ("open_app", "focus_app", "open_url", "search_web", "setup_workspace")),
    ("app closed", _rx(
        rf"^\W*(?:closed|quit|closing)\s+\S",
        rf"\b{_FIRST}(?:closed|quit|am\s+closing|closing)\s+\S",
    ), ("close_app",)),
    ("volume / brightness", _rx(
        r"\b(?:volume|brightness)\b[^.?!]{0,25}\b(?:set|raised|lowered|increased|decreased|turned|now\s+at|is\s+now)\b",
        rf"\b{_FIRST}(?:muted|unmuted|raised|lowered|turned\s+(?:up|down)|set\s+the\s+(?:volume|brightness))\b",
    ), ("set_volume", "mute", "set_brightness")),
    ("screen locked", _rx(
        rf"\b{_FIRST}?locked\s+(?:the\s+|your\s+)?(?:screen|mac|computer)\b",
        r"\b(?:screen|mac)\b[^.?!]{0,15}\b(?:is|has\s+been)\s+(?:now\s+)?locked\b",
    ), ("lock_screen",)),
    ("git commit", _rx(rf"\b{_FIRST}?committed\b", r"\bcommit(?:ted)?\s+[0-9a-f]{7}\b"), ("git_commit",)),
    ("command run", _rx(rf"\b{_FIRST}(?:ran|run|executed)\s+(?:the\s+|that\s+)?(?:command|script|tests?|shell)\b"),
     ("run_shell", "run_applescript", "run_tests")),
    ("memory forgotten", _rx(rf"\b{_FIRST}?(?:forgotten|deleted|erased|removed)\b[^.?!]{{0,30}}\b(?:memory|that)\b"),
     ("forget_memory",)),
    ("screenshot", _rx(rf"\b{_FIRST}?(?:took|taken|captured|saved)\b[^.?!]{{0,20}}\bscreenshot\b"), ("take_screenshot",)),
)

#: Vague acknowledgements — only claims when the user's request needed an action.
_ACK = re.compile(
    r"""(
        \bconsider\ it\ (?:noted|done|set|handled|sorted|arranged|taken\ care\ of|booked|scheduled)\b |
        ^\W*(?:noted|done|very\ good|all\ set|it\ is\ done|it's\ done|taken\ care\ of|right\ away)\b |
        \b(?:all|that's|that\ is)\ (?:done|taken\ care\ of|sorted|arranged)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

#: What the USER asked for -> (category label, tools that would fulfil it).
_INTENTS: Tuple[Tuple[str, "re.Pattern[str]", Tuple[str, ...]], ...] = (
    ("reminder", re.compile(r"\bremind\s+me\b|\bset\s+(?:a\s+|an\s+)?(?:reminder|alarm|timer)\b|\badd\s+(?:a\s+)?reminder\b|\bdon'?t\s+let\s+me\s+forget\b", re.I),
     ("add_reminder", "create_event")),
    ("calendar event", re.compile(r"\b(?:schedule|book)\b|\b(?:add|put|create)\b[^.?!]{0,40}\b(?:calendar|event|meeting|appointment)\b", re.I),
     ("create_event",)),
    ("email draft", re.compile(r"\bdraft\b|\b(?:reply|respond)\s+to\b", re.I), ("draft_reply",)),
    ("email archive", re.compile(r"\barchive\b", re.I), ("archive",)),
    ("mark read", re.compile(r"\bmark\b[^.?!]{0,30}\bas\s+read\b", re.I), ("mark_read",)),
    ("message sent", re.compile(r"\b(?:send|text|dm)\b[^.?!]{0,40}\b(?:email|e-mail|message|mail|text|slack|dm)\b|\bsend\s+(?:an?\s+)?(?:email|message|text)\b", re.I),
     ("post_message", "reply_thread")),
    ("app opened", re.compile(r"\b(?:open|launch|start|switch\s+to|focus)\b", re.I), ("open_app", "focus_app", "open_url", "search_web", "setup_workspace")),
    ("app closed", re.compile(r"\b(?:close|quit)\b", re.I), ("close_app",)),
    ("volume / brightness", re.compile(r"\b(?:mute|unmute|volume|brightness)\b", re.I), ("set_volume", "mute", "set_brightness")),
    ("screen locked", re.compile(r"\block\b[^.?!]{0,15}\b(?:screen|mac|computer)\b", re.I), ("lock_screen",)),
    ("git commit", re.compile(r"\bcommit\b", re.I), ("git_commit",)),
)


# ---------------------------------------------------------------- matching

def tool_matches(succeeded: Iterable[str], names: Iterable[str]) -> bool:
    """True if any succeeded tool is one of `names` (tolerating the MCP bridge's
    `<server>_<tool>` rename on a name collision)."""
    wanted = tuple(names)
    return any(s == n or s.endswith(f"_{n}") for s in succeeded for n in wanted)


def find_unbacked_claim(
    reply: str, user_text: str, succeeded_tools: Iterable[str]
) -> Optional[Claim]:
    """Return the first claim in `reply` that no succeeded tool backs, else None."""
    succeeded = list(succeeded_tools)
    sentences = [s for s in _split(reply) if not _NOT_A_CLAIM.search(s)]
    if not sentences:
        return None

    for sentence in sentences:
        for category, patterns, tools in _EXPLICIT:
            if any(p.search(sentence) for p in patterns) and not tool_matches(succeeded, tools):
                return Claim(category, sentence, frozenset(tools), "explicit")

    ack = next((s for s in sentences if _ACK.search(s)), None)
    if ack is not None:
        for category, intent, tools in _INTENTS:
            if intent.search(user_text or "") and not tool_matches(succeeded, tools):
                return Claim(category, ack, frozenset(tools), "ack")
    return None
