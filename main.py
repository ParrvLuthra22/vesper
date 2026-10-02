#!/usr/bin/env python3
"""
VESPER — legacy entry point, now a redirect.

`python main.py` used to boot the legacy stack: continuous listening with no wake
word, Google cloud speech recognition and the pyttsx3 voice, with no HUD. That
path is retired. Everything now runs under the launcher:

    vesper up         # gateway -> voice output -> HUD -> voice input, supervised
    vesper status
    vesper down
    vesper            # text-only terminal REPL (no microphone, no speaker)

`python main.py` simply runs `vesper up`. Nothing in this file touches the
microphone; no audio leaves the machine by default (docs/PRIVACY.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run() -> None:
    argv = sys.argv[1:]
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        sys.exit(0)
    if argv:
        print(
            "main.py no longer takes options (--config/--debug). Set VESPER_CONFIG for an "
            "alternate settings file; run `vesper up --help` for launcher options.",
            file=sys.stderr,
        )
    print("main.py is a redirect: starting `vesper up` (the legacy voice path is retired).\n")
    from launcher.cli import main as launcher_main

    sys.exit(launcher_main(["up"]))


if __name__ == "__main__":
    run()
