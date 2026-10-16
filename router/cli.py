"""`vesper route "<utterance>"` — explain which skill (if any) would answer an utterance, and why. Runs nothing: it only
normalises the text and walks the patterns, so it is safe to use on anything."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from router.skills import build_router


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="vesper route", description=__doc__.split("\n\n")[0])
    ap.add_argument("utterance", nargs="+", help="what you would say")
    ap.add_argument("--list", action="store_true", help="also list every skill and its rules")
    args = ap.parse_args(argv)
    try:
        from briefing.config import load_briefing_config
        bcfg = load_briefing_config()
    except Exception:  # noqa: BLE001 — the router works without the briefing config
        bcfg = None
    router = build_router(briefing_cfg=bcfg)
    if args.list:
        for s in router.skills:
            print(f"{s.name:16} tier>={s.tier:8} {s.description}  [{', '.join(r.id for r in s.rules)}]")
        print()
    print(router.explain(" ".join(args.utterance)).render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
