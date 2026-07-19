"""
Best-effort JSON repair for LLM tool-call arguments.

Providers occasionally emit tool arguments that are almost-but-not-quite
valid JSON (trailing commas, single-quoted strings). This module tries a
strict parse first, then progressively more lenient rewrites, before giving
up and returning an empty dict rather than raising — a single malformed
tool call shouldn't blow up an entire completion.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_SINGLE_QUOTED_RE = re.compile(r"'([^']*)'")


def parse_tool_arguments(raw: Any) -> Dict[str, Any]:
    """
    Parse tool-call arguments into a dict.

    `raw` may already be a dict (some providers, e.g. Ollama, hand back
    parsed arguments directly) or a JSON-ish string (Groq/OpenAI-style, where
    `function.arguments` is a string that is supposed to be strict JSON but
    isn't always). Returns `{}` if nothing usable can be recovered.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}

    text = str(raw).strip()
    if not text:
        return {}

    for candidate in _repair_candidates(text):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    return {}


def _repair_candidates(text: str) -> Iterator[str]:
    """Yield `text` and progressively more aggressive rewrites of it."""
    yield text

    no_trailing_commas = _TRAILING_COMMA_RE.sub(r"\1", text)
    yield no_trailing_commas

    # Naive single->double quote swap. Safe enough for a last-resort repair
    # pass on tool-call arguments (short key/value payloads), even though it
    # would mangle apostrophes inside prose.
    double_quoted = _SINGLE_QUOTED_RE.sub(r'"\1"', no_trailing_commas)
    yield double_quoted
