"""Where a finished transcript goes. Voice is JUST ANOTHER CLIENT: the default
sink POSTs the transcript to the gateway's /message endpoint — the exact path
the CLI and any other client use — so there is no special voice planner path.
"""
from __future__ import annotations

import json
from typing import Callable

from voice.input.config import VoiceInputConfig


class TranscriptSink:
    def inject(self, text: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class CallableSink(TranscriptSink):
    """Delegates to a callback. Used by tests (and any in-process embedding)."""

    def __init__(self, fn: Callable[[str], object]):
        self._fn = fn
        self.injected: list[str] = []

    def inject(self, text: str) -> bool:
        self.injected.append(text)
        result = self._fn(text)
        return True if result is None else bool(result)


class GatewayRestSink(TranscriptSink):
    """POST http://host:port/message {"text": ...} with the bearer token —
    identical to typing the same words into the CLI/gateway."""

    def __init__(self, config: VoiceInputConfig):
        self._url = f"http://{config.gateway_host}:{config.gateway_port}/message"
        self._token = config.gateway_token

    def inject(self, text: str) -> bool:
        import urllib.error
        import urllib.request

        body = json.dumps({"text": text}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        req = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return 200 <= resp.status < 300
        except urllib.error.URLError:
            return False
