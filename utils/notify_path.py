"""Which path delivers a macOS notification: the HUD (its own app identity, "Vesper HUD") or `osascript` (fallback)?

`osascript display notification` is attributed to whatever app runs it — the terminal, or "Script Editor" — and that app often has no
notification permission, so a banner can silently never appear. The HUD is a real app (com.vesper.hud): it asks macOS for permission under
its OWN name, the user sees a normal "Vesper HUD wants to send notifications" prompt, and System Settings lists it. So:

    HUD connected and not known to be denied/unavailable  ->  "hud": the gateway tells the HUD, the HUD posts the banner
    otherwise                                              ->  "osascript": the old path, attributed to the terminal app

The HUD reports its state over the websocket (`hello`, then `hud_notify`); the gateway process (which also hosts the Brain, so the notify tool
and the watcher read this same object) records it here and in data/notifications.json for `vesper doctor` / `vesper notify-test`.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

HUD_APP = "Vesper HUD"
HUD, OSASCRIPT = "hud", "osascript"


class NotifyPath:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.hud_connected = False
        self.hud_state = "unknown"          # unknown | granted | denied | unavailable
        self._clients = 0

    def hud_hello(self) -> None:
        with self._lock:
            self._clients += 1
            self.hud_connected = True

    def hud_gone(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)
            self.hud_connected = self._clients > 0

    def hud_report(self, state: str) -> None:
        if state in ("granted", "denied", "unavailable"):
            with self._lock:
                self.hud_state = state

    def active(self) -> str:
        """"hud" only while a HUD is connected and has not said it cannot post banners."""
        with self._lock:
            return HUD if (self.hud_connected and self.hud_state not in ("denied", "unavailable")) else OSASCRIPT

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {"path": HUD if (self.hud_connected and self.hud_state not in ("denied", "unavailable")) else OSASCRIPT,
                    "hud_connected": self.hud_connected, "hud_state": self.hud_state}


#: One per process. The gateway (Brain in-process) writes it; the notify tool and the watcher read it.
PATH = NotifyPath()


def app_for(path: str, terminal: str) -> str:
    """The app macOS attributes the banner to — and so the one whose notification permission matters."""
    return HUD_APP if path == HUD else terminal
