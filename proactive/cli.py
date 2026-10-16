"""`vesper dnd on|off|status` — the do-not-disturb switch for the ProactiveWatcher (proactive/watcher.py).

`on` writes a small flag file (default data/proactive_dnd.json); the watcher checks it every tick, so it takes effect within
seconds and survives restarts. `on --for 90` turns itself off after 90 minutes. Held-back interruptions are dropped, not
queued: when DND ends nothing arrives late. `status` prints counts only — never any message content."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import List, Optional

from proactive.watcher import ProactiveWatcher, WatcherConfig, read_dnd, write_dnd


def _config() -> WatcherConfig:
    try:
        from config.settings import load_config_dict
        raw = (((load_config_dict().get("proactive") or {}).get("watcher")) or {})
        return WatcherConfig.from_dict(raw)
    except Exception:  # noqa: BLE001 — DND must work even if the config is unreadable
        return WatcherConfig()


def main(argv: Optional[List[str]] = None, now: Optional[datetime] = None) -> int:
    ap = argparse.ArgumentParser(prog="vesper dnd", description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("on", "off", "status"))
    ap.add_argument("--for", dest="minutes", type=float, default=None, metavar="MINUTES",
                    help="with `on`: switch DND off again after this many minutes")
    args = ap.parse_args(argv)
    cfg = _config()
    now = now or datetime.now(timezone.utc)
    if args.action == "on":
        data = write_dnd(cfg.dnd_path, True, now, args.minutes)
        print("Do not disturb is ON" + (f" until {data['until']}" if data.get("until") else " until you turn it off")
              + ". Proactive interruptions are silenced (they are dropped, not queued).")
        return 0
    if args.action == "off":
        write_dnd(cfg.dnd_path, False, now)
        print("Do not disturb is OFF.")
        return 0
    state = read_dnd(cfg.dnd_path, now)
    print("Do not disturb: " + ("ON" + (f" until {state['until']}" if state.get("until") else "") if state.get("on") else "off"))
    w = ProactiveWatcher(cache=None, cfg=cfg)           # type: ignore[arg-type]  # status() never touches the cache
    s = w.status()
    print(f"Watcher: {'enabled' if cfg.enabled else 'disabled'} | today {s['day']}: {s['delivered_today']}/{s['max_per_day']} "
          f"delivered | quiet hours now: {'yes' if s['quiet_hours_now'] else 'no'}")
    if s["by_kind"]:
        print("  delivered: " + ", ".join(f"{k}={v}" for k, v in sorted(s["by_kind"].items())))
    if s["suppressed"]:
        print("  held back: " + ", ".join(f"{k}={v}" for k, v in sorted(s["suppressed"].items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
