"""`vesper up | down | status | logs` — thin argparse front end over launcher.stack."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

from launcher.state import RunState, pid_alive

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXIT_NOT_RUNNING = 3


def _state() -> RunState:
    return RunState(PROJECT_ROOT / "data" / "run")


def _fmt_uptime(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def cmd_up(args: argparse.Namespace) -> int:
    from launcher.stack import run_up

    try:
        return asyncio.run(run_up(strict=args.strict))
    except KeyboardInterrupt:
        return 130


def cmd_down(args: argparse.Namespace) -> int:
    state = _state()
    pid = state.running_supervisor_pid()
    if pid is None:
        state.clear()  # drop any stale file from a crashed supervisor
        print("Vesper is not running.")
        return 0
    print(f"Stopping Vesper (supervisor pid {pid}) ...")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            print("Stopped.")
            return 0
        time.sleep(0.2)
    print(f"Supervisor {pid} did not exit within {args.timeout:g}s. "
          f"Inspect with `vesper status`; as a last resort: kill -9 {pid}.")
    return 1


def cmd_status(args: argparse.Namespace) -> int:
    state = _state()
    pid = state.running_supervisor_pid()
    snap = state.read()
    if pid is None or snap is None:
        if args.json:
            print(json.dumps({"running": False}))
        else:
            print("Vesper is not running.  Start it with: vesper up")
        return EXIT_NOT_RUNNING

    live = None
    token = state.read_token()
    if token:
        from launcher.stack import gateway_status, load_app_config

        live = gateway_status(load_app_config(), token)

    if args.json:
        print(json.dumps({"running": True, "live": live, **snap}, indent=2, default=str))
        return 0

    up = time.time() - float(snap.get("started_at", time.time()))
    print(f"Vesper is running — supervisor pid {pid}, up {_fmt_uptime(up)}")
    print(f"  {'COMPONENT':<14}{'STATE':<10}{'PID':<8}{'UPTIME':<9}{'RESTARTS':<9}DETAIL")
    for c in snap.get("components", []):
        since = c.get("ready_since")
        print(f"  {c['name']:<14}{c['state'].upper():<10}{str(c['pid'] or '-'):<8}"
              f"{_fmt_uptime(time.time() - since if since else None):<9}{c['restarts']:<9}{c.get('detail') or ''}")
    if live is not None:
        providers = ", ".join(
            f"{p['tier']}:{p['provider']}{'' if p.get('available') else ' (unavailable)'}"
            for p in live.get("providers", [])
        )
        print(f"  gateway live: {live.get('clients', '?')} client(s) connected; LLM {providers}")
    else:
        print("  gateway live: NOT responding")
    bad = [c for c in snap.get("components", []) if c["state"] in ("failed", "backoff")]
    if bad:
        print("  attention: " + ", ".join(f"{c['name']} ({c['state']})" for c in bad)
              + "  — see `vesper logs <component>`")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    log_dir = PROJECT_ROOT / "logs" / "launcher"
    names = [args.component] if args.component else sorted(p.stem for p in log_dir.glob("*.log"))
    if not names:
        print(f"No launcher logs in {log_dir} yet.")
        return 1
    for name in names:
        path = log_dir / f"{name}.log"
        if not path.exists():
            print(f"No log for {name!r} ({path}).")
            return 1
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-args.lines:]
        print(f"===== {name} (last {len(lines)} lines) =====")
        print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vesper", description="Start, supervise and stop Vesper.")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="start the whole stack and supervise it (foreground)")
    up.add_argument("--strict", action="store_true",
                    help="stop if any optional component is skipped or fails at startup")
    up.set_defaults(func=cmd_up)

    down = sub.add_parser("down", help="stop a running stack cleanly")
    down.add_argument("--timeout", type=float, default=30.0)
    down.set_defaults(func=cmd_down)

    status = sub.add_parser("status", help="show component state")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    logs = sub.add_parser("logs", help="tail a component's log")
    logs.add_argument("component", nargs="?")
    logs.add_argument("-n", "--lines", type=int, default=40)
    logs.set_defaults(func=cmd_logs)
    return parser


COMMANDS = ("up", "down", "status", "logs")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return int(args.func(args))
