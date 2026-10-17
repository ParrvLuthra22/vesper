"""A name vocabulary for speech-to-text: "message Rahul" must not become "message Rahool".

faster-whisper takes an `initial_prompt` (and `hotwords`): text that biases the decoder toward those spellings. We build it from

  * first names of the people the user actually writes to (derived from the SENT-mail address local parts in the briefing's
    known-correspondent set, most-written-to first), and
  * first names from Contacts (read-only, only if the Contacts permission is already granted — never prompts),

capped at ~60 names. It lives only in this process's memory and is passed straight to the recogniser: it is NEVER logged (the log gets
a COUNT), never printed, never sent anywhere, never written to disk. Config: voice.stt.vocab (default on), voice.stt.max_names.
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Callable, Dict, Iterable, List, Optional, Sequence

MAX_NAMES_DEFAULT = 60
PROMPT_CHARS = 420                      # whisper's prompt budget is ~224 tokens; stay well under

#: local parts that are roles/automation, not people
_NOT_A_NAME = {"info", "hello", "hi", "support", "admin", "sales", "team", "contact", "office", "mail", "email", "billing", "help",
               "noreply", "no", "reply", "donotreply", "service", "hr", "jobs", "careers", "press", "news", "me", "accounts",
               "orders", "invoice", "invoices", "security", "notifications", "updates", "marketing", "account", "root", "test"}
#: first names that are also ordinary words are kept out of the prompt: biasing toward "will"/"mark" makes the recogniser invent them
_ORDINARY_WORDS = {"will", "mark", "art", "bill", "grace", "hope", "joy", "rose", "may", "june", "april", "dawn", "faith", "frank",
                   "gene", "guy", "jack", "pat", "ray", "rob", "sue", "van", "bob", "don", "kit", "ken", "sky", "storm"}


def first_name_from_address(address: str) -> Optional[str]:
    local = (address or "").split("@", 1)[0].lower()
    token = re.split(r"[._\-+0-9]+", local)[0]
    if not token.isalpha() or not 3 <= len(token) <= 15 or token in _NOT_A_NAME or token in _ORDINARY_WORDS:
        return None
    return token.capitalize()


def names_from_addresses(addresses: Dict[str, int]) -> List[str]:
    """First names, most-written-to first."""
    out: List[str] = []
    for addr, _n in sorted(addresses.items(), key=lambda kv: (-kv[1], kv[0])):
        name = first_name_from_address(addr)
        if name:
            out.append(name)
    return out


def clean_name(name: str) -> Optional[str]:
    n = re.sub(r"[^A-Za-zÀ-ɏ'\- ]", "", (name or "")).strip().split(" ")[0] if name else ""
    if not 2 <= len(n) <= 20 or n.lower() in _ORDINARY_WORDS or n.lower() in _NOT_A_NAME:
        return None
    return n[:1].upper() + n[1:]


def build_vocab(known_first_names: Iterable[str] = (), contact_first_names: Iterable[str] = (), cap: int = MAX_NAMES_DEFAULT) -> List[str]:
    """Known correspondents first (they matter most), then contacts; de-duplicated case-insensitively; at most `cap` names."""
    seen, out = set(), []
    for raw in list(known_first_names) + list(contact_first_names):
        n = clean_name(raw)
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
        if len(out) >= max(0, int(cap)):
            break
    return out


def prompt_from(names: Sequence[str]) -> str:
    """The text handed to the recogniser. Empty when there are no names."""
    if not names:
        return ""
    text = "Names that may be spoken: " + ", ".join(names) + "."
    return text if len(text) <= PROMPT_CHARS else text[:PROMPT_CHARS].rsplit(",", 1)[0] + "."


def known_addresses(db_path: str) -> Dict[str, int]:
    """Read-only: the known-correspondent addresses cached by the briefing engine ({} if there is no cache yet)."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
        try:
            row = con.execute("SELECT value FROM meta WHERE key=?", ("known_correspondents",)).fetchone()
        finally:
            con.close()
        data = json.loads(row[0]) if row else {}
        return {str(k): int(v) for k, v in (data.get("addresses") or {}).items()}
    except Exception:  # noqa: BLE001 — no cache, locked, malformed: no vocabulary from it
        return {}


def contact_names(backend=None) -> List[str]:
    """Given names from Contacts, only if access is ALREADY granted (never prompts). [] otherwise."""
    try:
        if backend is None:
            from tools.contacts_tools import SystemContacts

            backend = SystemContacts()
        if backend.authorization() != 3:
            return []
        return [c.given for c in backend.contacts() if c.given]
    except Exception:  # noqa: BLE001
        return []


def load_vocab(enabled: bool, db_path: str, cap: int = MAX_NAMES_DEFAULT, contacts: Optional[Callable[[], List[str]]] = None) -> List[str]:
    if not enabled:
        return []
    return build_vocab(names_from_addresses(known_addresses(db_path)), (contacts or contact_names)(), cap)
