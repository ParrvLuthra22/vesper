"""`vesper notify-test`: post one notification, ask "Did you see it? (y/n)", store the answer (status, timestamp, terminal app)."""
from __future__ import annotations

import sys
from typing import Optional, Sequence

from utils.notifications import NotifyState, run_notify_test, terminal_app


def main(argv: Optional[Sequence[str]] = None) -> int:
    from agents.system_agent import AppleScriptExecutor

    ex = AppleScriptExecutor()
    app = terminal_app()
    print(f"Posting one test notification from {app} (a banner should appear within a few seconds)...")
    return run_notify_test(lambda t, m: ex.show_notification(title=t, message=m), input, NotifyState(), app)


if __name__ == "__main__":
    sys.exit(main())
