"""`vesper notify-test`: post one notification through the ACTIVE path, ask "Did you see it? (y/n)", store the answer.

Active path (utils/notify_path.py): if the stack is running and its HUD is connected, the HUD posts the banner under its own identity ("Vesper HUD");
otherwise `osascript` posts it under this terminal app. The answer is stored for THAT app, so a confirmation made for one path never
vouches for the other."""
from __future__ import annotations

import json
import sys
import urllib.request
from typing import Optional, Sequence, Tuple

from utils import notify_path as np_
from utils.notifications import NotifyState, run_notify_test, terminal_app


def ask_gateway(title: str, body: str, base: str, token: str, timeout: float = 3.0) -> Optional[dict]:
    """POST /notify; the gateway answers which path it used. None = no gateway (stack not running)."""
    req = urllib.request.Request(base + "/notify", data=json.dumps({"title": title, "body": body}).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:  # noqa: BLE001
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    from agents.system_agent import AppleScriptExecutor
    from launcher.state import RunState
    from launcher.stack import PROJECT_ROOT, gateway_address, load_app_config

    ex = AppleScriptExecutor()
    term = terminal_app()
    cfg = load_app_config()
    host, port = gateway_address(cfg)
    token = RunState(PROJECT_ROOT / "data" / "run").read_token() or ""
    seen = {"path": np_.OSASCRIPT}

    def post(title: str, body: str) -> Tuple[bool, str]:
        got = ask_gateway(title, body, f"http://{host}:{port}", token)
        if got and got.get("path") == np_.HUD:
            seen["path"] = np_.HUD
            return True, ""
        return ex.show_notification(title=title, message=body)

    def asker(prompt: str) -> str:
        return input(prompt)

    # Post first (decides the path), then store the answer for the app that path belongs to.
    print("Posting one test notification (HUD if the stack is running with its HUD, otherwise via this terminal)...")
    ok, err = post("Vesper notification test", "If you can read this, notifications work. You can dismiss it.")
    app = np_.app_for(seen["path"], term)
    print(f"Path used: {'the HUD (Vesper HUD)' if seen['path'] == np_.HUD else 'osascript (' + term + ')'}.")
    return run_notify_test(lambda t, m: (ok, err), asker, NotifyState(), app, print, posted=True)


if __name__ == "__main__":
    sys.exit(main())
