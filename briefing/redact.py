"""
`--redact`: print inbox-derived output with every sender, subject, snippet and domain
replaced by a stable placeholder (<sender:3>, <subject:7>, <domain:2>, <snippet:1>).

Fail closed, in two independent layers:

  1. STRUCTURAL: the CLI never formats a real value in redacted mode. Rows are built from
     `Redactor.sender()/subject()/...` and free-text pieces (scorer reasons) must match an
     allowlist of known-static labels, otherwise they become <reason:N>. Changing the table
     layout cannot leak, because the layout code only ever receives placeholders.
  2. GUARD: every byte the command would print (stdout AND stderr, plus error messages) is
     buffered and scanned against EVERYTHING sensitive the cache/known-set/rules hold — not
     just what the command meant to print. A hit withholds the whole output; the leaked
     text is never echoed. Unparseable or unexpected output therefore cannot pass through.

Placeholders are stable within a run (same sender -> same <sender:N>) and are numbered by
first appearance, so they carry no information about the value.
"""

from __future__ import annotations

import re
from email.utils import parseaddr
from typing import Dict, Iterable, List, Optional, Set

__all__ = ["Redactor", "RedactionLeak", "SAFE_WORDS", "safe_reason", "scan_fields"]


class RedactionLeak(Exception):
    """Raised by Redactor.guard; the message never contains the leaked text."""

    def __init__(self, count: int):
        super().__init__(f"{count} sensitive value(s) detected in output")
        self.count = count


_WORD = re.compile(r"[a-z0-9]{3,}")
_SPACE = re.compile(r"\s+")

#: Words that appear in static text we print (headers, labels, messages). Sensitive values
#: that happen to equal one of these are not flagged individually (their FULL value still is).
SAFE_WORDS: frozenset = frozenset(
    """
    score source when from title why unread mail priority other bulk meetings shown sources
    sender subject snippet domain reason rule rules mark ignore known correspondent correspondents
    address addresses domains calendar gmail gmail stale never synced ago old fresh days hours
    personal bulk signals signal noreply reply list header promo newsletter auto submitted
    automated important allowlist boost vip waived capped strong urgency words mentions
    meeting meetings today later tomorrow starts within progress now all day event beyond
    briefing window already read over start time excluded text reads like instructions
    cache empty try refresh with the and for this that sent message messages derived last
    header only stored mail from your have written others there thread penalties penalty
    through points point never reaches removed recorded what does stored file plain json
    edit delete any time model learning applied nothing else undo scores rank now use
    tools unavailable total are not tokens cap spoken words redacted placeholder withheld
    output guard tripped error none unknown
    """.split()
)

#: Ordinary English function words. Free-text n-grams made only of these (and of SAFE_WORDS) are not
#: guarded: they carry no information and would collide with our own static messages.
STOPWORDS: frozenset = frozenset(
    """
    a about after again all also and any are been before being but can could did does done each even
    for from get got had has have her here him his how into its just like made make many may more most
    much must not now off one only our out over own said same see she should some such than that the
    their them then there these they this those too under up very was way were what when where which
    while who will with would yes yet you your
    """.split()
)


def _distinctive(gram: str) -> bool:
    return any(len(w) >= 4 and w not in SAFE_WORDS and w not in STOPWORDS for w in gram.split())


_STATIC_REASONS = [
    r"already read", r"unread mail", r"noreply sender", r"mailing-list header", r"promo signal",
    r"newsletter signal", r"auto-submitted", r"personal mail \(no bulk signals\)",
    r"important automated sender \(allowlist boost\)",
    r"known correspondent \(I've written to this address\)",
    r"known domain \(I've written to others there\)", r"reply in a thread I sent in", r"is a reply",
    r"fresh \(\d+h old\)", r"unread for \d+ days", r"text reads like instructions to an AI",
    r"title reads like instructions to an AI", r"no start time", r"already over",
    r"all-day, today", r"all-day, tomorrow", r"beyond the briefing window", r"all-day event",
    r"in progress now", r"starts within [\d.]+h", r"later today", r"tomorrow", r"no signals",
]
_SAFE_REASON = re.compile("^(?:" + "|".join(f"(?:{p})" for p in _STATIC_REASONS) + ")$")
# Reasons that embed config words or a sender/rule are reduced to their static prefix.
_PREFIXED = [
    (re.compile(r"^urgency words \([a-z ,'\-]*\)$"), "urgency words (…)"),
    (re.compile(r"^strong urgency \([a-z ,'\-]*\): bulk penalties capped at .*$"), "strong urgency (…): bulk penalties capped"),
    (re.compile(r"^title mentions [a-z ,'\-]*$"), "title mentions (…)"),
    (re.compile(r"^VIP sender \(.*\)$"), "VIP sender (<sender>)"),
    (re.compile(r"^your rule #\d+: .*$"), "your rule"),
    (re.compile(r"^[a-z ():]+: bulk penalties waived$"), "bulk penalties waived"),
]


def safe_reason(label: str, redactor: "Redactor") -> str:
    """A scorer reason label made safe: static labels pass, known dynamic ones are reduced to
    their static part, and anything unrecognised becomes <reason:N> (fail closed)."""
    label = (label or "").strip()
    if _SAFE_REASON.match(label):
        return label
    for rx, replacement in _PREFIXED:
        if rx.match(label):
            return replacement
    return redactor.reason(label)


def _domain_tokens(domain: str) -> Set[str]:
    """Word parts of a domain WITHOUT its last label (the TLD: "com", "test", "io" ...). The full
    domain is still guarded as a whole; a TLD on its own identifies nothing and is a common word."""
    labels = [x for x in domain.split(".") if x]
    if len(labels) >= 2:
        labels = labels[:-1]
    return set(_WORD.findall(" ".join(labels)))


def _norm(value: object) -> str:
    return _SPACE.sub(" ", str(value or "").casefold()).strip()


class Redactor:
    def __init__(self) -> None:
        self._maps: Dict[str, Dict[str, int]] = {}
        self._full: Set[str] = set()      # normalised complete values
        self._tokens: Set[str] = set()    # name/address/domain parts (3+ chars)
        self._ngrams: Set[str] = set()    # word 3-grams of free text (subjects/snippets/titles)

    # ----------------------------------------------------------- placeholders
    def _ph(self, kind: str, key: str) -> str:
        table = self._maps.setdefault(kind, {})
        k = _norm(key)
        if not k:
            return f"<{kind}:0>"
        if k not in table:
            table[k] = len(table) + 1
        return f"<{kind}:{table[k]}>"

    # ----------------------------------------------------------- seeding (guard only)
    def _add_full(self, value: str) -> None:
        """A lone word that is part of our own static vocabulary ("noreply", "newsletter", "unread")
        cannot be told apart from that text, so it is not guarded on its own; the full address,
        domain, name and subject it came from still are."""
        v = _norm(value)
        if v and (v in SAFE_WORDS or v in STOPWORDS):
            return
        if v:
            self._full.add(v)

    def seed_text(self, value: object) -> None:
        """Free text (subject, snippet, title): guard on the full value and on word 2/3-grams, so a
        TRUNCATED copy of it still trips the guard."""
        n = _norm(value)
        if not n:
            return
        self._add_full(n)
        words = re.findall(r"[a-z0-9]+", n)
        if len(words) <= 3:
            self._add_full(" ".join(words))
        for size in (2, 3):
            for i in range(len(words) - size + 1):
                gram = " ".join(words[i:i + size])
                if _distinctive(gram):
                    self._ngrams.add(gram)

    def seed_identity(self, value: object) -> None:
        """A sender / address / domain / rule value: guard on the full value and its parts."""
        n = _norm(value)
        if not n:
            return
        self._add_full(n)
        name, addr = parseaddr(str(value))
        nn = _norm(name)
        if nn:
            self._add_full(nn)
            self._tokens.update(_WORD.findall(nn))
        an = _norm(addr)
        if an:
            self._add_full(an)
            if "@" in an:
                local, _, domain = an.partition("@")
                self._add_full(local)
                self._add_full(domain)
                self._tokens.update(_WORD.findall(local))
                self._tokens.update(_domain_tokens(domain))
            elif " " not in an and "." in an:      # a bare domain ("example.com")
                self._tokens.update(_domain_tokens(an))
            else:
                self._tokens.update(_WORD.findall(an))
        if not nn and not an:
            self._tokens.update(_WORD.findall(n))

    # ----------------------------------------------------------- redacting
    def sender(self, raw: object) -> str:
        """A From header (or bare address): one placeholder per sender, keyed by address."""
        self.seed_identity(raw)
        name, addr = parseaddr(str(raw or ""))
        return self._ph("sender", addr or name or str(raw or ""))

    def address(self, addr: object) -> str:
        return self.sender(addr)

    def domain(self, domain: object) -> str:
        self.seed_identity(str(domain or "").lstrip("@"))
        return self._ph("domain", str(domain or "").lstrip("@"))

    def subject(self, text: object) -> str:
        self.seed_text(text)
        return self._ph("subject", str(text or ""))

    def snippet(self, text: object) -> str:
        self.seed_text(text)
        return self._ph("snippet", str(text or ""))

    def reason(self, text: object) -> str:
        self.seed_text(text)
        return self._ph("reason", str(text or ""))

    def rule_value(self, scope: str, value: object) -> str:
        return self.domain(value) if scope == "domain" else self.sender(value)

    # ----------------------------------------------------------- the guard
    def findings(self, text: str) -> int:
        """How many distinct sensitive values appear in `text`."""
        hay = _norm(text)
        hay_words = " ".join(re.findall(r"[a-z0-9]+", hay))
        hits = 0
        for full in self._full:
            # Boundaries matter: a short address part like "news" must not match "newsletter".
            if len(full) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(full)}(?![a-z0-9])", hay):
                hits += 1
        for gram in self._ngrams:
            if f" {gram} " in f" {hay_words} ":
                hits += 1
        token_words = set(re.findall(r"[a-z0-9]+", hay))
        for tok in self._tokens:
            if tok in token_words and tok not in SAFE_WORDS:
                hits += 1
        return hits

    def guard(self, text: str) -> str:
        n = self.findings(text)
        if n:
            raise RedactionLeak(n)
        return text


def scan_fields(redactor: Redactor, items: Iterable[object], extra: Optional[List[str]] = None) -> None:
    """Seed the guard with everything sensitive an Item exposes (sender, title, snippet), whether
    or not the command prints it. `items` are briefing.items.Item objects."""
    for it in items:
        redactor.seed_identity(getattr(it, "sender", ""))
        redactor.seed_text(getattr(it, "title", ""))
        redactor.seed_text(getattr(it, "snippet", ""))
    for e in extra or []:
        redactor.seed_identity(e)
