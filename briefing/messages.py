"""
MessagesCollector — unread iMessages / SMS as briefing Items. LOCAL and READ-ONLY.

    fetch_since(cursor)   incoming messages with ROWID > cursor that I have not read or answered
    live_ids()            which cached Messages items are still unread/unanswered (from the same snapshot)
    available()           False ("not configured") unless enabled AND the database can be opened

How it reads chat.db — the only place real message data lives on this Mac:
  * never in place: the database (and its -wal/-shm) is COPIED into a private temp directory, the copy is
    opened read-only (SQLite URI ``mode=ro`` plus ``PRAGMA query_only``), and the copy is deleted afterwards.
    The original is opened for reading by ``shutil.copyfile`` only; nothing here can write to it.
  * macOS needs Full Disk Access for that. Without it the collector reports "not configured" with the exact
    steps (see ``FDA_STEPS``) and the briefing is unaffected. ``permission_state`` only tries to open the file
    (no read, no prompt, nothing about the content).

What it maps (Item.source == "messages"):
    incoming 1:1 or group message, not a tapback/reaction, not attachments-only, not in a muted chat,
    and no newer message from me in the same chat                       -> unread/pending item
    my own messages, reactions, stickers, system rows (renames, joins)  -> ignored

Everything text-shaped is third-party data: neutralised (``clean_text``), digit runs of 4+ are replaced with
"(code)" AT INGEST so a one-time passcode can never be stored, quoted or spoken, and text that reads like
instructions to an AI is flagged ``hostile`` and stores NO snippet (the scorer demotes it, the builder counts
it as "filtered as unsafe").
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from briefing.collectors import Collector
from briefing.config import BriefingConfig
from briefing.items import Item
from briefing.sanitize import clean_text
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DB = "~/Library/Messages/chat.db"
#: Apple's Core Data epoch (2001-01-01) in Unix seconds. Modern macOS stores message.date in NANOseconds since it.
APPLE_EPOCH = 978307200

FDA_STEPS = (
    "Messages needs Full Disk Access: System Settings > Privacy & Security > Full Disk Access > "
    "add your terminal app (Terminal, iTerm, Ghostty ...) and switch it on, then quit and reopen the terminal. "
    "Vesper only copies chat.db to a private temp folder, reads the copy, and deletes it; it never writes to Messages."
)

# ----------------------------------------------------------------------------- text handling

_OBJ_REPLACEMENT = "￼"
#: Four or more digits, allowing single "-" / space / "." separators between digit groups ("123 456", "12-34-56").
_DIGIT_RUN = re.compile(r"\d(?:[ .\-]?\d){3,}")
_OTP_WORDS = re.compile(
    r"\b(?:otp|one[- ]time|verification|verify|passcode|security code|login code|auth(?:entication)? code|"
    r"your (?:\w+ ){0,2}code|code is|do not share|don't share|never share)\b", re.IGNORECASE)


def mask_codes(text: object) -> str:
    """Replace every run of 4+ digits (one-time codes, card/phone fragments, PINs) with "[code]"."""
    return _DIGIT_RUN.sub("[code]", str(text or ""))


def safe_snippet(raw: object, max_chars: int) -> str:
    """The stored preview: codes masked BEFORE truncation (so a cut can never leave half of a code), and masked as
    "(code)" because clean_text strips square brackets as markup."""
    return clean_text(mask_codes(raw).replace("[code]", "(code)"), max_chars)


def looks_like_otp(text: object) -> bool:
    return bool(_OTP_WORDS.search(str(text or "")))


def decode_attributed_body(blob: Optional[bytes]) -> Optional[str]:
    """The text of a message whose ``text`` column is NULL (macOS 13+ keeps it in a typedstream
    NSAttributedString in ``attributedBody``). Returns None when the blob is not understood — never raises,
    never guesses past the declared length.

    Layout after the ``NSString`` class name:  ... 0x01 0x2b <length> <utf-8 bytes>
      length < 0x80          one byte
      0x81 b0 b1             little-endian uint16
      0x82 b0 b1 b2 b3       little-endian uint32
    """
    if not blob or not isinstance(blob, (bytes, bytearray, memoryview)):
        return None
    data = bytes(blob)
    start = data.find(b"NSString")
    if start < 0:
        return None
    window = data[start + len(b"NSString"): start + len(b"NSString") + 24]
    marker = window.find(b"\x01\x2b")
    if marker < 0:
        return None
    pos = start + len(b"NSString") + marker + 2
    if pos >= len(data):
        return None
    first = data[pos]
    if first < 0x80:
        length, pos = first, pos + 1
    elif first == 0x81 and pos + 3 <= len(data):
        length, pos = int.from_bytes(data[pos + 1:pos + 3], "little"), pos + 3
    elif first == 0x82 and pos + 5 <= len(data):
        length, pos = int.from_bytes(data[pos + 1:pos + 5], "little"), pos + 5
    else:
        return None
    if length <= 0 or pos + length > len(data):
        return None
    try:
        text = data[pos:pos + length].decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text.replace(_OBJ_REPLACEMENT, " ").strip() or None


def encode_attributed_body(text: str) -> bytes:
    """Builds the same typedstream shape (used by the test fixtures; the inverse of decode_attributed_body)."""
    raw = text.encode("utf-8")
    if len(raw) < 0x80:
        length = bytes([len(raw)])
    elif len(raw) < 0x10000:
        length = b"\x81" + len(raw).to_bytes(2, "little")
    else:
        length = b"\x82" + len(raw).to_bytes(4, "little")
    return (b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x12NSAttributedString\x00\x84\x84\x08NSObject\x00"
            b"\x85\x92\x84\x84\x84\x08NSString\x01\x94\x84\x01+" + length + raw
            + b"\x86\x84\x02iI\x01" + bytes([min(len(raw), 0x7f)]) + b"\x92\x84\x84\x84\x0cNSDictionary\x00\x94\x84\x01i\x00\x86\x86")


# ----------------------------------------------------------------------------- permission + snapshot

def permission_state(path: str) -> str:
    """"ok" | "missing" | "denied" for the chat.db path, WITHOUT reading it. Opening the file for reading
    is what Full Disk Access gates; macOS shows no prompt, it simply refuses. Nothing is read."""
    p = Path(os.path.expanduser(path))
    if not p.exists():
        # TCC hides the file from stat() as well for protected folders: PermissionError distinguishes the two.
        try:
            p.parent.stat()
        except PermissionError:
            return "denied"
        except OSError:
            return "missing"
        return "missing"
    try:
        fd = os.open(str(p), os.O_RDONLY)
    except PermissionError:
        return "denied"
    except OSError:
        return "missing"
    os.close(fd)
    return "ok"


class Snapshot:
    """A private, read-only copy of a SQLite database (plus -wal/-shm), deleted on close."""

    def __init__(self, source: str):
        self._src = Path(os.path.expanduser(source))
        self._dir: Optional[str] = None
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> sqlite3.Connection:
        self._dir = tempfile.mkdtemp(prefix="vesper-msgs-")
        os.chmod(self._dir, 0o700)
        dst = Path(self._dir) / "chat.db"
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(self._src) + suffix)
            if src.exists():
                shutil.copyfile(src, str(dst) + suffix)
        # mode=ro: the connection cannot write. (A -wal beside the copy is replayed into memory, not the file.)
        self.conn = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
        self.conn.execute("PRAGMA query_only = ON")
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def __exit__(self, *exc: object) -> None:
        try:
            if self.conn is not None:
                self.conn.close()
        finally:
            if self._dir:
                shutil.rmtree(self._dir, ignore_errors=True)
            self._dir, self.conn = None, None


def _to_epoch(raw: Any) -> float:
    """message.date -> Unix seconds. Nanoseconds since 2001 (modern) or seconds since 2001 (old)."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if v <= 0:
        return 0.0
    if v > 1e12:                 # nanoseconds
        v /= 1e9
    return v + APPLE_EPOCH


def _from_epoch(ts: float) -> int:
    return int((ts - APPLE_EPOCH) * 1e9)


# ----------------------------------------------------------------------------- sender classification

_PHONE = re.compile(r"^\+?[\d\s().\-]{7,}$")
_EMAIL_ID = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def classify_handle(handle: str, service: str = "") -> str:
    """"person" (a phone number or an email address) or "automated" (a short code, an alphanumeric sender id such
    as "HDFCBK", or a business-chat urn). Deliberately conservative: only these shapes are called automated."""
    h = (handle or "").strip()
    if not h:
        return "person"
    if h.lower().startswith(("urn:biz", "biz:")):
        return "automated"
    if _EMAIL_ID.match(h):
        return "person"
    digits = re.sub(r"\D", "", h)
    if _PHONE.match(h):
        return "automated" if len(digits) <= 6 else "person"
    if re.fullmatch(r"\d{3,6}", h):
        return "automated"
    return "automated" if re.search(r"[A-Za-z]", h) else "person"


# ----------------------------------------------------------------------------- contacts (optional)

def _norm_phone(value: str) -> str:
    d = re.sub(r"\D", "", value or "")
    return d[-10:] if len(d) >= 7 else ""


class ContactBook:
    """handle -> display name from the local Address Book, read from a private COPY like chat.db. Optional:
    no Address Book (or no permission) simply means numbers are shown as-is."""

    def __init__(self, names_by_key: Dict[str, str]):
        self._names = names_by_key

    def lookup(self, handle: str) -> Optional[str]:
        h = (handle or "").strip().lower()
        return self._names.get(h) or self._names.get(_norm_phone(h) or "\0")

    @classmethod
    def from_paths(cls, paths: List[str]) -> "ContactBook":
        names: Dict[str, str] = {}
        for p in paths:
            try:
                with Snapshot(p) as conn:
                    for row in conn.execute(
                            "SELECT r.ZFIRSTNAME AS f, r.ZLASTNAME AS l, r.ZORGANIZATION AS o, p.ZFULLNUMBER AS num "
                            "FROM ZABCDPHONENUMBER p JOIN ZABCDRECORD r ON r.Z_PK = p.ZOWNER"):
                        name = " ".join(x for x in (row["f"], row["l"]) if x) or (row["o"] or "")
                        key = _norm_phone(row["num"] or "")
                        if name and key:
                            names.setdefault(key, name)
                    for row in conn.execute(
                            "SELECT r.ZFIRSTNAME AS f, r.ZLASTNAME AS l, r.ZORGANIZATION AS o, e.ZADDRESS AS addr "
                            "FROM ZABCDEMAILADDRESS e JOIN ZABCDRECORD r ON r.Z_PK = e.ZOWNER"):
                        name = " ".join(x for x in (row["f"], row["l"]) if x) or (row["o"] or "")
                        if name and row["addr"]:
                            names.setdefault(str(row["addr"]).lower(), name)
            except (sqlite3.Error, OSError):
                continue
        return cls(names)

    @classmethod
    def discover(cls, root: str = "~/Library/Application Support/AddressBook") -> "ContactBook":
        base = Path(os.path.expanduser(root))
        paths = [str(p) for p in sorted(base.glob("Sources/*/AddressBook-v22.abcddb"))]
        paths += [str(p) for p in [base / "AddressBook-v22.abcddb"] if p.exists()]
        return cls.from_paths(paths)


# ----------------------------------------------------------------------------- the collector

_REACTION_TYPES = range(1000, 4000)        # stickers (1000), tapbacks add (2000-2007) / remove (3000-3007)

_QUERY = """
SELECT m.ROWID AS rowid, m.guid AS guid, m.text AS text, m.attributedBody AS body, m.date AS date,
       m.is_from_me AS is_from_me, m.is_read AS is_read,
       COALESCE(m.associated_message_type, 0) AS assoc, COALESCE(m.cache_has_attachments, 0) AS has_att,
       COALESCE(m.item_type, 0) AS item_type, COALESCE(m.is_audio_message, 0) AS is_audio,
       COALESCE(h.id, '') AS handle, COALESCE(h.service, m.service, '') AS service,
       c.ROWID AS chat_id, COALESCE(c.style, 45) AS style, COALESCE(c.chat_identifier, '') AS chat_ident,
       COALESCE(c.display_name, '') AS chat_name
FROM message m
JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
JOIN chat c ON c.ROWID = cmj.chat_id
LEFT JOIN handle h ON h.ROWID = m.handle_id
WHERE (m.date >= :ns OR (m.date > 0 AND m.date < 1000000000000 AND m.date >= :sec))
ORDER BY m.ROWID
"""


class MessagesCollector(Collector):
    name = "messages"
    optional = True       # not enabled / no permission => skipped silently, never a recorded failure

    def __init__(self, cfg: BriefingConfig, *, contacts: Optional[Callable[[], ContactBook]] = None,
                 cached: Optional[Callable[[], List[Item]]] = None, clock: Callable[[], float] = time.time):
        self._cfg = cfg
        self._contacts_factory = contacts
        self._contacts: Optional[ContactBook] = None
        self._cached = cached
        self._clock = clock
        self._live: Optional[Set[str]] = None
        self._max_rowid = 0
        self.not_configured_reason: Optional[str] = None

    # --------------------------------------------------------------- configuration
    def available(self) -> bool:
        if not self._cfg.messages_enabled:
            self.not_configured_reason = "disabled (messages.enabled: false)"
            return False
        state = permission_state(self._cfg.messages_db_path)
        if state == "ok":
            self.not_configured_reason = None
            return True
        self.not_configured_reason = ("not configured: " + FDA_STEPS) if state == "denied" else \
            f"not configured: no Messages database at {self._cfg.messages_db_path}"
        return False

    # ----------------------------------------------------------------------- fetch
    async def fetch_since(self, cursor: Optional[str]) -> List[Item]:
        if not self.available():
            raise RuntimeError(self.not_configured_reason or "not configured")
        now = self._clock()
        floor = now - self._cfg.messages_lookback_days * 86400
        try:
            after_rowid = int(cursor) if cursor else 0
        except ValueError:
            after_rowid = 0
        with Snapshot(self._cfg.messages_db_path) as conn:
            rows = [dict(r) for r in conn.execute(_QUERY, {"ns": _from_epoch(floor), "sec": int(floor - APPLE_EPOCH)})]
        # newest message from me, per chat: an answered thread is not pending
        my_last: Dict[int, float] = {}
        for r in rows:
            if r["is_from_me"] and r["assoc"] not in _REACTION_TYPES and not r["item_type"]:
                my_last[r["chat_id"]] = max(my_last.get(r["chat_id"], 0.0), _to_epoch(r["date"]))
        pending: Dict[str, Item] = {}
        live: Set[str] = set()
        self._max_rowid = max([after_rowid] + [int(r["rowid"]) for r in rows])
        for r in rows:
            item = self._map(r, my_last, now)
            if item is None:
                continue
            live.add(item.id)
            if int(r["rowid"]) > after_rowid:
                pending[item.id] = item
        self._live = live
        return sorted(pending.values(), key=lambda i: -i.timestamp)

    def next_cursor(self, cursor: Optional[str], items: List[Item], now: float) -> Optional[str]:
        return str(self._max_rowid) if self._max_rowid else cursor

    async def live_ids(self) -> Optional[Set[str]]:
        return self._live

    # --------------------------------------------------------------------- mapping
    def _is_muted(self, r: Dict[str, Any]) -> bool:
        muted = set(self._cfg.messages_muted_chats)
        return any(str(v).strip().lower() in muted for v in (r["chat_ident"], r["chat_name"]) if v)

    def _display(self, handle: str) -> str:
        if self._contacts is None:
            self._contacts = (self._contacts_factory or (lambda: ContactBook({})))() \
                if self._cfg.messages_contacts else ContactBook({})
        return self._contacts.lookup(handle) or handle

    def _map(self, r: Dict[str, Any], my_last: Dict[int, float], now: float) -> Optional[Item]:
        from briefing.slack import hostile_text   # lazy: slack.py imports collectors

        if r["is_from_me"] or r["assoc"] in _REACTION_TYPES or r["item_type"] or r["is_audio"]:
            return None
        if r["is_read"] or self._is_muted(r):
            return None
        ts = _to_epoch(r["date"])
        if ts <= 0 or ts <= my_last.get(r["chat_id"], 0.0):
            return None                      # I answered after it
        raw = r["text"] if (r["text"] or "").strip() else decode_attributed_body(r["body"])
        raw = (raw or "").replace(_OBJ_REPLACEMENT, " ").strip()
        if not raw:
            return None                      # attachment-only (or an undecodable body): nothing to say
        handle = str(r["handle"] or r["chat_ident"] or "")
        group = int(r["style"]) == 43
        automated = classify_handle(handle, r["service"]) == "automated"
        otp = looks_like_otp(raw)
        if self._allowlisted(handle):
            automated = False
        hostile = hostile_text(raw, handle)
        name = self._display(handle)
        snippet = "" if hostile else safe_snippet(raw, self._cfg.messages_snippet_chars)
        age_h = max(0.0, (now - ts) / 3600.0)
        return Item(
            id=f"messages:{int(r['rowid'])}",
            source="messages",
            sender=clean_text(name, 32) or "unknown",
            title="group message" if group else "message",
            snippet=snippet,
            timestamp=ts,
            thread_id=str(r["chat_id"]),
            unread=True,
            signals={
                "kind": "group" if group else "dm", "service": str(r["service"] or ""), "automated": automated,
                "otp": otp, "hostile": hostile, "allowlisted": self._allowlisted(handle),
                "unanswered": age_h >= self._cfg.messages_unanswered_hours,
            },
        )

    def _allowlisted(self, handle: str) -> bool:
        allow = set(self._cfg.messages_allowlisted_senders)
        return bool(handle) and handle.strip().lower() in allow
