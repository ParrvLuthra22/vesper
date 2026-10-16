"""
Make component logs safe to paste: `vesper logs <component> --redact-transcripts`.

The component logs are written for debugging, and several line shapes carry what you SAID or what was said to you:

    [EMIT] Event: VoiceInputEvent | ID: ... | Source: ... | Payload: {'text': '<your transcript>', ...}
    [EMIT] Event: VoiceOutputEvent | ... | Payload: {'text': '<the reply>'}
    Observation: kind=meeting_soon detail='<calendar text>'
    [VoiceAgent] Received VoiceOutputEvent: '<text>' from Brain
    heard: <transcript>  /  transcript=<...>  /  user_text=<...>
    voice transcribing -> injected (<transcript>)        (the voice pipeline's own state-change line)

Rules (fail closed, mirroring `--redact` for the briefing):

  1. Inside an event `Payload: {...}`, every value is replaced by a placeholder UNLESS its key is on a short allow-list
     of identifiers / numbers / flags (event ids, source, priority, tool name, success...). An unknown key is treated as
     text. A payload cut off mid-string by the logger is replaced to the end of the line.
  2. `text=...`, `transcript=...`, `detail=...` ... style fields: the value (quoted, or the rest of the line) is replaced.
  3. Quoted text after "Received <Event>:" and "heard:/transcript:/said:/speaking:" style prefixes is replaced.

A placeholder is `<text:N>`: the same text gets the same N within one invocation, so a repeated utterance is still
visibly repeated, and N carries no information about the content. Lines that match no rule are left untouched, so this is
only as good as the rules above: it recognises the shapes Vesper itself writes, and cannot know about a new log line that
interpolates text some other way. Read the output once before pasting it anywhere.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

#: Payload keys whose values are identifiers, numbers or flags — never speech. Everything else in a payload is treated as text.
SAFE_PAYLOAD_KEYS = frozenset({
    "event_id", "correlation_id", "request_id", "observation_id", "plan_id", "schedule_name", "source", "priority", "kind",
    "type", "action", "target_agent", "tool_name", "tool", "context_type", "context_key", "turn_number", "total_steps",
    "success", "approved", "active", "animate", "model", "latency_ms", "channel", "channel_user", "timestamp",
    "confidence", "duration_ms", "voice", "streaming", "level",
})

_QUOTED = r"""'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*\""""
_UNTERMINATED = r"""'(?:\\.|[^'\\])*$|"(?:\\.|[^"\\])*$"""
_PAYLOAD = re.compile(r"(Payload:\s*)(.+)$")          # ANY payload, not only a {...} dict
_PAIR = re.compile(
    r"""(?P<key>'(?P<k1>[^']+)'|"(?P<k2>[^"]+)")(?P<sep>\s*:\s*)(?P<val>""" + _QUOTED + r"""|\[[^\]]*\]?|\{[^}]*\}?|""" + _UNTERMINATED + r"""|[^,}]+)""")
_FIELD = re.compile(
    r"(?i)\b(?P<k>text|transcript|transcription|utterance|user_text|user_input|heard|detail|summary|snippet|subject|body|"
    r"message|reply|said|query|prompt|content|title|context_value)(?P<sep>\s*=\s*)(?P<val>" + _QUOTED + r"|\S.*)$")
_AFTER_VERB = re.compile(
    r"(?i)(?P<pre>\b(?:heard|transcript|transcription|transcribed|said|speaking|saying|utterance|you said|user said)\s*[:\-]\s*)(?P<val>.+)$")
#: The voice pipeline logs what it heard as the detail of its "injected" transition: "voice transcribing -> injected (<text>)".
_TRANSITION = re.compile(r"(?P<pre>\bvoice \w+ -> injected \()(?P<val>.*)\)\s*$")
_RECEIVED = re.compile(r"(?P<pre>Received \w*Event:\s*)(?P<val>" + _QUOTED + r")")


class Placeholders:
    """`<text:N>`, numbered by first appearance; the same text always gets the same N."""

    def __init__(self) -> None:
        self._seen: Dict[str, int] = {}

    def __call__(self, value: str) -> str:
        key = " ".join(value.split()).lower()
        if key not in self._seen:
            self._seen[key] = len(self._seen) + 1
        return f"<text:{self._seen[key]}>"


def _redact_payload(m: "re.Match[str]", ph: Placeholders) -> str:
    body = m.group(2)

    def pair(pm: "re.Match[str]") -> str:
        key = pm.group("k1") or pm.group("k2") or ""
        val = pm.group("val")
        if key in SAFE_PAYLOAD_KEYS and not re.match(r"""^['"].{40,}""", val):
            return pm.group(0)
        return f"{pm.group('key')}{pm.group('sep')}{ph(val)}"

    out, last = [], 0
    for pm in _PAIR.finditer(body):
        out.append(body[last:pm.start()])
        out.append(pair(pm))
        last = pm.end()
    tail = body[last:]
    # Whatever the pair pattern could not parse (a payload truncated by the logger, odd nesting) is not left in the clear.
    if tail.strip(" ,}") and not re.fullmatch(r"[\s,}\]]*", tail):
        tail = ph(tail) + ("}" if tail.rstrip().endswith("}") else "")
    return m.group(1) + "".join(out) + tail


def redact_line(line: str, ph: Placeholders) -> Tuple[str, bool]:
    """(line with transcripts / message text replaced, whether anything was replaced)."""
    original = line
    line = _PAYLOAD.sub(lambda m: _redact_payload(m, ph), line)
    line = _TRANSITION.sub(lambda m: m.group("pre") + ph(m.group("val")) + ")", line)
    line = _RECEIVED.sub(lambda m: m.group("pre") + ph(m.group("val")), line)
    line = _FIELD.sub(lambda m: f"{m.group('k')}{m.group('sep')}{ph(m.group('val'))}", line)
    line = _AFTER_VERB.sub(lambda m: m.group("pre") + ph(m.group("val")), line)
    return line, line != original


def redact_lines(lines: Iterable[str], ph: "Placeholders | None" = None) -> Tuple[List[str], int]:
    ph = ph or Placeholders()
    out, changed = [], 0
    for ln in lines:
        new, did = redact_line(ln, ph)
        out.append(new)
        changed += 1 if did else 0
    return out, changed
