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

    def __post(self, path: str, payload: dict, timeout: float) -> bool:
        import urllib.error
        import urllib.request

        base = self._url.rsplit("/message", 1)[0]
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        req = urllib.request.Request(
            base + path, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= resp.status < 300
        except urllib.error.URLError:
            return False

    def inject(self, text: str) -> bool:
        return self.__post("/message", {"text": text}, timeout=10)

    def signal_wake(self) -> bool:
        """Tell the gateway the wake word fired — triggers the HUD flare, the
        spoken greeting, and voice-output barge-in (the cinematic reveal)."""
        return self.__post("/wake", {}, timeout=3)
