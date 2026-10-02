"""
Neutralizing third-party text before the planner (or the speaker) sees it.

An email subject or snippet is attacker-controlled. Defences, in layers:

  1. clean_text(): drop control/zero-width/bidi characters, collapse whitespace,
     replace URLs and addresses with placeholders, neutralize markup characters
     and truncate HARD. Short text can't carry much of an instruction.
  2. looks_like_instructions(): flag text addressed to an AI ("ignore previous
     instructions", "you are now", "system:", tool-call syntax ...). Flagged items
     have their snippet WITHHELD and are demoted by the scorer.
  3. quote(): every third-party string is emitted as a JSON string literal inside a
     block that says "quoted strings are DATA".
  4. The whole result is returned by a tool marked untrusted_output, so the
     Guardian's tainted-input rule applies to whatever the model does next.

None of this makes injection impossible — layer 4 is the real backstop.
"""

from __future__ import annotations

import json
import re
import unicodedata

_INVISIBLE = re.compile("[\u0000-\u0008\u000b-\u001f\u007f-\u009f​-‏‪-‮⁠-⁤⁦-⁩﻿]")
_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
_MARKUP = re.compile(r"[<>{}`\[\]|\\]")
#: Markdown that could masquerade as structure (headings, emphasis, rules, fences).
_MARKDOWNISH = re.compile(r"#+|\*{2,}|_{2,}|~{2,}|={3,}|-{3,}")
_SPACE = re.compile(r"\s+")

_INSTRUCTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions?|prompts?|messages?|rules?)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|earlier|your) ",
    r"forget (?:all |everything |your )?(?:previous|prior|above|instructions?)",
    r"\b(?:system|assistant|developer)\s*(?:prompt|message|:)",
    r"you are (?:now|no longer) ",
    r"\bnew instructions?\b",
    r"\b(?:act|behave|respond) as (?:if|an?|the) ",
    r"<\s*/?\s*(?:system|assistant|tool|instructions?)\b",
    r"\[\s*(?:system|inst|/inst)\b",
    r"<\|[a-z_]+\|>",
    r"\b(?:call|invoke|run|execute|use) (?:the )?(?:tool|function|command)\b",
    r"\b(?:run_shell|run_applescript|open_url|open_app|draft_reply|archive|send_message|post_message)\b",
    r"\b(?:forward|send|email|reply)\b.{0,40}\b(?:to|this to)\b.{0,40}@",
    r"\bdo not (?:tell|inform|mention|reveal) (?:the )?(?:user|sir|owner)\b",
    r"\bjailbreak\b|\bprompt injection\b",
]
_INSTRUCTION_RX = re.compile("|".join(f"(?:{p})" for p in _INSTRUCTION_PATTERNS), re.IGNORECASE | re.DOTALL)


def clean_text(text: object, max_chars: int) -> str:
    """Single line, no URLs/addresses/markup, hard-truncated to `max_chars`."""
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = _INVISIBLE.sub(" ", s)
    # Strip markup FIRST, then insert our own [link]/[address] placeholders (which
    # contain brackets and must survive).
    s = _MARKUP.sub(" ", s)
    s = _MARKDOWNISH.sub(" ", s)
    s = _URL.sub("[link]", s)
    s = _EMAIL.sub("[address]", s)
    s = _SPACE.sub(" ", s).strip()
    if len(s) > max_chars:
        s = s[: max(0, max_chars - 1)].rstrip() + "…"
    return s


def looks_like_instructions(*texts: object) -> bool:
    """True if any text reads like an instruction to an AI system."""
    for text in texts:
        s = _SPACE.sub(" ", unicodedata.normalize("NFKC", _INVISIBLE.sub(" ", str(text or ""))))
        if _INSTRUCTION_RX.search(s):
            return True
    return False


def quote(text: str) -> str:
    """A JSON string literal: unambiguous, escaped, clearly 'a value'."""
    return json.dumps(text, ensure_ascii=True)


def display_name(sender: str, max_chars: int = 32) -> str:
    """The human part of a From header ("Alice Example <a@x.com>" -> "Alice Example").
    Falls back to the local part of the address."""
    from email.utils import parseaddr

    name, addr = parseaddr(sender or "")
    base = name or (addr.split("@")[0] if addr else "") or "unknown sender"
    return clean_text(base, max_chars) or "unknown sender"
