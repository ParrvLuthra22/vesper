"""contacts_lookup (read-only) and messages_compose (confirm tier, NEVER sends).

    "message Priya"  ->  contacts_lookup("Priya")   one candidate, or several the user must choose between
                     ->  messages_compose(contact_ref, body?)   Messages opens at that conversation; the user reads, edits and
                                                               presses send THEMSELVES

Privacy, by construction:
  * The model never sees a phone number or e-mail address: lookup returns display names and short refs ("c1"); the handle stays in
    this process (ContactRefs, 10-minute memory) and is looked up again by messages_compose.
  * Both tools are `private`: the live events, the tracer, the audit log and the confirmation card carry COUNTS, never a name
    or a handle (tools/registry.py ToolSpec.private, guardian/gate.py, orchestrator/planner.py).
  * Contacts are read through the Contacts framework (read-only); nothing is cached to disk.

Sending: there is no send path. messages_compose only runs `open <url>` with an imessage:// or sms: URL (argv, no shell, no AppleScript).
tests/test_contacts_messages.py scans this file to prove it: no AppleScript, no keystrokes, no 'return' key, no click.
"""
from __future__ import annotations

import asyncio
import difflib
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from tools.registry import ToolSpec, get_registry
from utils.grounding import grounded
from utils.logger import get_logger

logger = get_logger(__name__)
registry = get_registry()

MAX_CANDIDATES = 5
REF_TTL_S = 600.0
MATCH_FLOOR = 0.80
AMBIGUITY_GAP = 0.08


@dataclass(frozen=True)
class Handle:
    kind: str            # "phone" | "email"
    value: str
    label: str = ""


@dataclass(frozen=True)
class Contact:
    display: str
    given: str = ""
    family: str = ""
    nickname: str = ""
    org: str = ""
    handles: Tuple[Handle, ...] = ()


class ContactsBackend(Protocol):
    def authorization(self) -> Optional[int]: ...           # 3 = authorized

    def contacts(self) -> List[Contact]: ...


# --------------------------------------------------------------------------- the real backend (read-only, PyObjC)

class SystemContacts:
    """CNContactStore, read-only. Fetches name + phone + e-mail fields only."""

    def authorization(self) -> Optional[int]:
        try:
            import Contacts

            return int(Contacts.CNContactStore.authorizationStatusForEntityType_(Contacts.CNEntityTypeContacts))
        except Exception:  # noqa: BLE001
            return None

    def contacts(self) -> List[Contact]:
        import Contacts

        store = Contacts.CNContactStore.alloc().init()
        keys = [Contacts.CNContactGivenNameKey, Contacts.CNContactFamilyNameKey, Contacts.CNContactNicknameKey,
                Contacts.CNContactOrganizationNameKey, Contacts.CNContactPhoneNumbersKey, Contacts.CNContactEmailAddressesKey]
        request = Contacts.CNContactFetchRequest.alloc().initWithKeysToFetch_(keys)
        out: List[Contact] = []

        def each(c, stop):
            handles = []
            for p in c.phoneNumbers() or []:
                handles.append(Handle("phone", str(p.value().stringValue()), str(p.label() or "")))
            for e in c.emailAddresses() or []:
                handles.append(Handle("email", str(e.value()), str(e.label() or "")))
            given, family = str(c.givenName() or ""), str(c.familyName() or "")
            display = (given + " " + family).strip() or str(c.organizationName() or "")
            if display and handles:
                out.append(Contact(display, given, family, str(c.nickname() or ""),
                                   str(c.organizationName() or ""), tuple(handles)))

        ok, err = store.enumerateContactsWithFetchRequest_error_usingBlock_(request, None, each)
        if not ok:
            raise PermissionError("Contacts could not be read")
        return out


# --------------------------------------------------------------------------- matching

def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split())


def sound_key(word: str) -> str:
    """A rough phonetic key so a mis-heard name still lands on the right person ("Priyah" ~ "Pria", "Katherine" ~ "Catherine")."""
    w = re.sub(r"[^a-z]", "", (word or "").lower())
    if not w:
        return ""
    for a, b in (("ph", "f"), ("ck", "k"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("sh", "s"), ("ch", "c"), ("wr", "r"), ("kn", "n"),
                 ("w", "v"), ("z", "s"), ("c", "k"), ("q", "k"), ("x", "ks"), ("y", "i")):
        w = w.replace(a, b)
    first, rest = w[0], re.sub(r"[aeiouh]", "", w[1:])
    rest = re.sub(r"(.)\1+", r"\1", rest)
    return (first + rest)[:6]


def score(spoken: str, c: Contact) -> float:
    s = _norm(spoken)
    if not s:
        return 0.0
    full, given, nick = _norm(c.display), _norm(c.given), _norm(c.nickname)
    if s in (full, nick):
        return 1.0
    best = 0.0
    tokens = s.split()
    if len(tokens) == 1:
        if s == given:
            best = max(best, 0.97)
        if s == _norm(c.family):
            best = max(best, 0.9)
        if sound_key(s) and sound_key(s) == sound_key(c.given):
            best = max(best, 0.88)
        best = max(best, difflib.SequenceMatcher(None, s, given).ratio() * 0.95 if given else 0.0)
    else:
        if sound_key(tokens[0]) == sound_key(c.given) and sound_key(tokens[-1]) == sound_key(c.family):
            best = max(best, 0.9)
        best = max(best, difflib.SequenceMatcher(None, s, full).ratio())
    return best


def rank(spoken: str, contacts: Sequence[Contact]) -> List[Tuple[float, Contact]]:
    scored = sorted(((score(spoken, c), c) for c in contacts), key=lambda t: (-t[0], t[1].display))
    if not scored or scored[0][0] < MATCH_FLOOR:
        return []
    top = scored[0][0]
    keep = [(sc, c) for sc, c in scored if sc >= MATCH_FLOOR and top - sc <= AMBIGUITY_GAP]
    seen, out = set(), []
    for sc, c in keep:                                  # the same person twice (two cards) is one candidate
        key = (_norm(c.display), tuple(sorted(h.value for h in c.handles)))
        if key not in seen:
            seen.add(key)
            out.append((sc, c))
    return out[:MAX_CANDIDATES]


# --------------------------------------------------------------------------- refs (handles never reach the model)

class ContactRefs:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock, self._items, self._n = clock, {}, 0

    def add(self, c: Contact) -> str:
        self._n += 1
        ref = f"c{self._n}"
        self._items[ref] = (self._clock() + REF_TTL_S, c)
        for k in [k for k, (exp, _c) in self._items.items() if exp < self._clock()]:
            del self._items[k]
        return ref

    def get(self, ref: str) -> Optional[Contact]:
        item = self._items.get(str(ref or "").strip().lower())
        return item[1] if item and item[0] >= self._clock() else None


REFS = ContactRefs()


def normalize_phone(value: str) -> Optional[str]:
    digits = re.sub(r"[^\d+]", "", value or "")
    plus = digits.startswith("+")
    digits = re.sub(r"\D", "", digits)
    if not 5 <= len(digits) <= 15:
        return None
    return ("+" if plus else "") + digits


_EMAIL = re.compile(r"^[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+$")


def best_handle(c: Contact) -> Optional[Handle]:
    """A phone (mobile first) if there is one, else an e-mail. Returns the normalised handle."""
    phones = sorted((h for h in c.handles if h.kind == "phone" and normalize_phone(h.value)),
                    key=lambda h: 0 if "mobile" in h.label.lower() or "iphone" in h.label.lower() else 1)
    if phones:
        return Handle("phone", normalize_phone(phones[0].value) or "", phones[0].label)
    for h in c.handles:
        if h.kind == "email" and _EMAIL.match(h.value):
            return h
    return None


def build_url(handle: Handle, body: str = "", service: str = "imessage") -> str:
    """imessage://<handle>?body=…  (or sms:<handle>&body=…). Only characters that can occur in a phone number / address pass."""
    value = handle.value
    if handle.kind == "phone":
        value = normalize_phone(value) or ""
    if not value or not re.fullmatch(r"[+\w.@\-]+", value):
        raise ValueError("unusable handle")
    text = (body or "").strip()[:500]
    if service == "sms":
        return f"sms:{value}" + (f"&body={urllib.parse.quote(text, safe='')}" if text else "")
    return f"imessage://{value}" + (f"?body={urllib.parse.quote(text, safe='')}" if text else "")


# --------------------------------------------------------------------------- the tools

def make_lookup(backend: Optional[ContactsBackend] = None, refs: ContactRefs = REFS):
    async def contacts_lookup(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        name = str((arguments or {}).get("name", "")).strip()
        if len(name) < 2:
            return "contacts_lookup needs a name."
        be = backend or SystemContacts()
        auth = be.authorization()
        if auth == 0:
            return ("Contacts access has never been requested, so I cannot look anyone up. Tell the user to run `vesper contacts-auth` "
                    "once in a terminal (it makes macOS ask) — `vesper doctor` shows the status.")
        if auth is not None and auth != 3:
            return ("Contacts access is not granted, so I cannot look anyone up. Tell the user to allow Contacts for the terminal app "
                    "(System Settings > Privacy & Security > Contacts) — `vesper doctor` says which.")
        try:
            people = await asyncio.get_running_loop().run_in_executor(None, be.contacts)
        except Exception as exc:  # noqa: BLE001
            return f"Contacts could not be read ({type(exc).__name__}). Say so in one sentence; do not guess a contact."
        found = rank(name, people)
        if not found:
            return ("No contact matches that name. Ask the user to repeat or spell it; do not guess and do not message anyone.")
        lines = []
        for _sc, c in found:
            ref = refs.add(c)
            kinds = sorted({h.kind for h in c.handles})
            lines.append(f"{ref}: {c.display} ({' and '.join(kinds)} on file)")
        if len(found) == 1:
            head = "1 contact matches."
            tail = "Confirm the name with the user, then call messages_compose with this contact_ref."
        else:
            head = f"{len(found)} contacts match."
            tail = "Ask the user which one (read the names aloud); do not choose for them. Then call messages_compose with that contact_ref."
        return grounded(head + "\n" + "\n".join(lines) + "\n" + tail, can=["contact_matches"], cannot=["conversation", "sent"])
    return contacts_lookup


def make_compose(backend: Optional[ContactsBackend] = None, refs: ContactRefs = REFS,
                 opener: Optional[Callable[[str], Tuple[bool, str]]] = None,
                 frontmost: Optional[Callable[[], str]] = None):
    async def messages_compose(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        ref = str((arguments or {}).get("contact_ref", "")).strip()
        c = refs.get(ref)
        if c is None:
            return "That contact_ref is unknown or expired. Call contacts_lookup again first."
        handle = best_handle(c)
        if handle is None:
            return "That contact has no usable phone number or address. Say so in one sentence."
        service = "sms" if str((arguments or {}).get("service", "imessage")).lower() == "sms" else "imessage"
        body = str((arguments or {}).get("body", "") or "")
        try:
            url = build_url(handle, body, service)
        except ValueError:
            return "That contact's number or address is not usable. Say so in one sentence."
        loop = asyncio.get_running_loop()
        ok, err = await loop.run_in_executor(None, opener or _open_url, url)
        if not ok:
            return grounded("Messages could not be opened for that contact", can=[], cannot=["conversation", "sent"])
        front = await loop.run_in_executor(None, frontmost or _messages_frontmost)
        can = ["app_opened"]
        if front == "yes":
            can += ["app_frontmost", "conversation_opened", "compose_ready"]
            if body.strip():
                can.append("body_prefilled")
        cannot = ["sent"] + ([] if front == "yes" else ["conversation", "compose_ready"])
        summary = ("Opened Messages at that contact's conversation. NOTHING WAS SENT: the user reads, edits and presses send themselves."
                   if front == "yes" else
                   "Asked macOS to open Messages for that contact, but Messages is not confirmed in front. NOTHING WAS SENT.")
        return grounded(summary, can=can, cannot=cannot, frontmost=front)
    return messages_compose


def _open_url(url: str) -> Tuple[bool, str]:
    from utils import safe_exec

    ok, _out, err = safe_exec.run_argv(["open", url], timeout=10)
    return ok, err


def _messages_frontmost() -> str:
    from agents.system_agent import AppleScriptExecutor

    return AppleScriptExecutor().verify_frontmost("Messages")


async def _compose_summary(arguments: Dict[str, Any]) -> str:
    # No name, no number: the card and the logs carry neither (the assistant has already said the name aloud).
    body = "with your text pre-filled" if str(arguments.get("body", "")).strip() else "empty"
    return f"Open Messages to the contact you chose ({body}). Nothing will be sent — you press send."


def _register() -> None:
    if registry.get("contacts_lookup") is None:
        registry.register(ToolSpec(
            name="contacts_lookup",
            description=("Find a person in the user's Contacts from a spoken name (fuzzy: tolerates mis-heard names). Read-only. "
                         "Returns candidate refs (c1, c2...) with display names; if several match, ask the user which."),
            parameters={"type": "object", "properties": {"name": {"type": "string", "description": "The name as spoken."}},
                        "required": ["name"]},
            tier="safe", handler=make_lookup(), category="messaging", private=True, untrusted_output=False,
        ))
    if registry.get("messages_compose") is None:
        registry.register(ToolSpec(
            name="messages_compose",
            description=("Open the Messages app at a conversation with a contact from contacts_lookup, optionally with text pre-filled. "
                         "It NEVER sends: the user reads, edits and presses send. Use only after contacts_lookup and the user's choice."),
            parameters={"type": "object", "properties": {
                "contact_ref": {"type": "string", "description": "A ref from contacts_lookup, e.g. c1."},
                "body": {"type": "string", "description": "Optional text to pre-fill (the user can edit it)."},
                "service": {"type": "string", "enum": ["imessage", "sms"]}}, "required": ["contact_ref"]},
            tier="confirm", handler=make_compose(), category="messaging", private=True, confirm_summary=_compose_summary,
        ))


_register()
