"""Sentence splitting for TTS chunking. Mirrors hud/src/sentences.ts (the HUD display): keep the two in step.

A sentence ends at . ! ? followed by whitespace, EXCEPT when the word before the period is an abbreviation or a single initial
(No., Mr., Mrs., Dr., St., vs., etc., a.m., p.m., e.g., i.e. ...) or the next word starts with a digit or a lowercase letter.
"RFID card at Gate No. 2." is one chunk; "Good evening, Sir. You have two meetings." is two.
"""
from __future__ import annotations

import re
from typing import List

ABBREVIATIONS = frozenset(
    "no nos mr mrs ms mx dr prof sr jr st mt ft vs etc e.g i.e a.m p.m am pm inc ltd co corp dept approx est fig vol rev gen col "
    "lt sgt capt gov sen rep jan feb mar apr jun jul aug sep sept oct nov dec cf ca ave blvd rd".split())

_BOUNDARY = re.compile(r"([.!?]+[\"'”’)\]]*)(\s+)")


def _is_abbreviation(before: str) -> bool:
    parts = before.split()
    word = parts[-1] if parts else ""
    w = re.sub(r"^[\"'“‘(\[]+", "", word).rstrip(".").lower()
    if not w:
        return False
    return bool(re.fullmatch(r"[a-z]", w)) or w in ABBREVIATIONS


def split_sentences(text: str) -> List[str]:
    out: List[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        nxt = text[m.end():m.end() + 1]
        ends_period = m.group(1).startswith(".")
        if ends_period and m.group(1) == "." and (_is_abbreviation(text[start:m.start()]) or re.match(r"[a-z0-9]", nxt or " ")):
            continue
        if not ends_period and re.match(r"[a-z]", nxt or " "):
            continue
        out.append(text[start:m.start() + len(m.group(1))].strip())
        start = m.end()
    rest = text[start:].strip()
    if rest:
        out.append(rest)
    return [s for s in out if s]
