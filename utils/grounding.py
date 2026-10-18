"""Grounded tool results: what a tool really did, in a form the claim verifier (and the model) can rely on.

A UI/state tool used to return "Opening Messages" because the command exited 0 — and the model then wrote "iMessage is now open with a
conversation window for <name>, ready for you to type", none of which any tool did. Now a result ends with

    <what happened, one line>
    frontmost: yes|no|unknown                      (open_app / focus_app: checked with System Events, not assumed)
    can_claim: app_opened, app_frontmost           (exactly what may be said happened)
    cannot_claim: conversation, window             (things a listener would assume, that did NOT happen)

`orchestrator/claim_verifier.py` flags reply sentences about specifics no tool result supports and rewrites them from these lines.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Optional, Sequence

#: tokens -> how to say them in an honest sentence.
CAN_PHRASES: Dict[str, str] = {
    "app_opened": "opened {app}",
    "app_frontmost": "{app} is in front",
    "conversation_opened": "opened a Messages conversation",
    "compose_ready": "the conversation is ready for you to type",
    "body_prefilled": "put your text in the message box (not sent)",
    "contact_matches": "found matching contacts",
    "web_results": "searched the web",
    "notification_requested": "asked macOS to show a notification",
    "notification_displayed": "the notification appeared (confirmed working from this app)",
    "sound_played": "ran the sound player (exit 0)",
}
CANNOT_PHRASES: Dict[str, str] = {
    "conversation": "open a specific conversation",
    "window": "open a specific window",
    "page": "open a specific page or tab",
    "compose_ready": "get a message ready to type",
    "sent": "send messages",
    "draft": "write a draft there",
    "notification_displayed": "say that it appeared on screen",
    "sound_audible": "say that it was audible",
}


def grounded(text: str, *, can: Iterable[str] = (), cannot: Iterable[str] = (), frontmost: Optional[str] = None) -> str:
    lines = [text.rstrip()]
    if frontmost is not None:
        lines.append(f"frontmost: {frontmost}")
    lines.append("can_claim: " + (", ".join(can) if can else "nothing beyond this result"))
    cannot = list(cannot)
    if cannot:
        lines.append("cannot_claim: " + ", ".join(cannot))
    return "\n".join(lines)


@dataclass(frozen=True)
class Grounding:
    summary: str = ""
    can: FrozenSet[str] = frozenset()
    cannot: FrozenSet[str] = frozenset()
    frontmost: Optional[str] = None
    present: bool = False

    @property
    def app(self) -> str:
        m = re.match(r"(?:Opening|Opened|Focused|Switching to)\s+(.+)", self.summary.strip(), re.I)
        return m.group(1).strip(" .") if m else "the app"


_LINE = re.compile(r"^(frontmost|can_claim|cannot_claim):\s*(.*)$", re.I | re.M)


def parse(result: str) -> Grounding:
    text = str(result or "")
    found = {k.lower(): v.strip() for k, v in _LINE.findall(text)}
    if not found:
        return Grounding(summary=text.strip().splitlines()[0] if text.strip() else "")
    toks = lambda v: frozenset(t.strip() for t in v.split(",") if t.strip() and not t.startswith("nothing"))  # noqa: E731
    first = text.strip().splitlines()[0]
    return Grounding(summary=first, can=toks(found.get("can_claim", "")), cannot=toks(found.get("cannot_claim", "")),
                     frontmost=found.get("frontmost"), present=True)


def say(g: Grounding, claim_tokens: Sequence[str] = ()) -> str:
    """"I opened Messages, though it isn't in front, but I can't open a specific conversation." (built from the result only)"""
    did = []
    if "app_opened" in g.can:
        did.append(f"opened {g.app}")
    elif g.summary:
        did.append(g.summary.rstrip(". ").lower() if not g.summary[:1].isupper() else g.summary.rstrip(". "))
    out = "I " + did[0] if did and did[0].startswith("opened") else (did[0] if did else "I did what I could")
    if g.frontmost == "no":
        out += ", though it isn't in front"
    elif g.frontmost == "unknown":
        out += " (I couldn't confirm it is in front)"
    cannot = [CANNOT_PHRASES[t] for t in claim_tokens if t in CANNOT_PHRASES] or \
             [CANNOT_PHRASES[t] for t in sorted(g.cannot) if t in CANNOT_PHRASES][:1]
    if cannot:
        out += f", but I can't {cannot[0]}"
    return out + "."
