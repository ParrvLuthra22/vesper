"""Where a finished transcript goes. Voice is JUST ANOTHER CLIENT: the default
sink POSTs the transcript to the gateway's /message endpoint — the exact path
the CLI and any other client use — so there is no special voice planner path.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

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

    def _request(self, method: str, path: str, payload: Optional[dict], timeout: float):
        """(ok, parsed JSON body or None). Never raises: the gateway being down is an ordinary condition."""
        import urllib.error
        import urllib.request

        base = self._url.rsplit("/message", 1)[0]
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ok = 200 <= resp.status < 300
                try:
                    body = json.loads(resp.read().decode("utf-8") or "null")
                except ValueError:
                    body = None
                return ok, body
        except (urllib.error.URLError, OSError):
            return False, None

    def __post(self, path: str, payload: dict, timeout: float) -> bool:
        return self._request("POST", path, payload, timeout)[0]

    def inject(self, text: str) -> bool:
        """True = queued. False = the gateway refused it ("dropped": it was the assistant's own voice, or voice input is paused) or was down."""
        ok, body = self._request("POST", "/message", {"text": text, "origin": "voice"}, timeout=10)
        if ok and isinstance(body, dict) and body.get("status") == "dropped":
            return False
        return ok

    def speech_snapshot(self):
        """GET /speech as a dict, or None if the gateway cannot be asked (the echo gate then stays open)."""
        ok, body = self._request("GET", "/speech", None, timeout=1.5)
        return body if ok and isinstance(body, dict) else None

    def signal_wake(self, greet: bool = True, animate: bool = True) -> bool:
        """Tell the gateway the wake word fired — the HUD flare and voice-output barge-in. `greet` asks for the spoken
        persona greeting too; voice input only sets it after a long idle period (voice.session.greet_after_idle_min)."""
        return self.__post("/wake", {"greet": bool(greet), "animate": bool(animate)}, timeout=3)

    def reply_busy(self) -> Optional[bool]:
        """Is a reply still being worked out or spoken? None = the gateway could not be asked (do not wait for it)."""
        ok, body = self._request("GET", "/speech", None, timeout=2)
        if not ok or not isinstance(body, dict):
            return None
        return bool(body.get("busy"))
