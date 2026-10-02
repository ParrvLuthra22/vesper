"""`vesper briefing` — inspect the briefing engine.

    vesper briefing                 # the compact block the planner receives (+ token count)
    vesper briefing --explain       # every item's score and exactly why
    vesper briefing --spoken        # the 20-30 s script
    vesper briefing --refresh       # force a collector run first (starts the MCP servers)
    vesper briefing --cache-only    # never touch the network; read data/briefing.db as it is
    vesper briefing --json          # structured output

Read-only: it runs the same read-only collectors the app does and writes only the
local cache file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from typing import List, Optional

from briefing.builder import Briefing, render_tool_result, spoken_script
from briefing.config import load_briefing_config
from briefing.sanitize import clean_text, display_name
from briefing.service import BriefingService


def _age(ts: float, now: float) -> str:
    seconds = max(0, now - ts)
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def explain_table(b: Briefing) -> str:
    lines = [f"{'#':>2}  {'SCORE':>5}  {'SOURCE':<8} {'WHEN':<18} {'FROM / TITLE':<62} WHY"]
    for i, s in enumerate(b.all_scored, 1):
        it = s.item
        if it.source == "calendar":
            when = datetime.fromtimestamp(it.start_at or 0).strftime("%a %H:%M")
            who = f"{clean_text(it.title, 58)}"
        else:
            when = _age(it.timestamp, b.as_of)
            who = f"{display_name(it.sender, 20)} — {clean_text(it.title, 38)}"
        flag = " [excluded]" if s.excluded else ""
        lines.append(f"{i:>2}  {s.score:>5}  {it.source:<8} {when:<18} {who:<62} {s.explain()}{flag}")
    if len(lines) == 1:
        lines.append("   (cache is empty — try --refresh)")
    return "\n".join(lines)


def status_lines(b: Briefing) -> List[str]:
    return ["sources: " + "; ".join(s.describe() + ("" if s.available else " [tools unavailable]") for s in b.sources),
            f"unread mail: {b.total_unread} | priority: {len(b.priority)} | other: {b.other_count} "
            f"({b.bulk_count} bulk) | meetings shown: {len(b.meetings)}"]


async def _run(args: argparse.Namespace) -> int:
    cfg = load_briefing_config()
    if args.cache_only:
        from briefing.cache import BriefingCache
        from briefing.collectors import CalendarCollector, GmailCollector
        from briefing.readonly import ReadOnlyTools

        service = BriefingService([], cfg, BriefingCache(cfg.cache_path))
        service._collectors = []  # no tools: report health straight from the cache file
    else:
        from launcher.stack import load_app_config
        from tools.mcp_bridge import MCPBridge
        from tools.registry import get_registry

        bridge = MCPBridge(config=load_app_config(), registry=get_registry())
        print("[briefing] connecting MCP servers (read-only collectors)...", file=sys.stderr)
        await bridge.start()
        service = BriefingService.from_registry(get_registry(), cfg)
        try:
            if args.refresh:
                reports = await service.refresh()
            else:
                await service.ensure_fresh()
                reports = {}
            for r in reports.values():
                note = "ok" if r.ok else f"FAILED: {r.error}"
                print(f"[briefing] {r.name}: {note} ({r.new_items} item(s) fetched)", file=sys.stderr)
        finally:
            await bridge.stop()

    b = service.briefing(time.time())
    if args.json:
        print(json.dumps({
            "as_of": b.as_of, "unread": b.total_unread,
            "meetings": [{"title": s.item.title, "start": s.item.start_at, "score": s.score} for s in b.meetings],
            "priority": [{"id": s.item.id, "score": s.score, "why": s.explain()} for s in b.priority],
            "sources": [vars(s) for s in b.sources],
        }, indent=2, default=str))
    elif args.explain:
        print(explain_table(b))
        print()
        print("\n".join(status_lines(b)))
    elif args.spoken:
        text = spoken_script(b, cfg)
        print(text)
        print(f"\n({len(text.split())} words, ≈{len(text.split()) / 2.7:.0f}s spoken)")
    else:
        text, tokens = render_tool_result(b, cfg)
        print(text)
        print(f"\n[{tokens} tokens / cap {cfg.token_cap}]")
    service.cache.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vesper briefing", description="Inspect the briefing engine (read-only).")
    p.add_argument("--explain", action="store_true", help="print each item's score and why")
    p.add_argument("--spoken", action="store_true", help="print the spoken script")
    p.add_argument("--json", action="store_true")
    p.add_argument("--refresh", action="store_true", help="force a collector run first")
    p.add_argument("--cache-only", action="store_true", help="do not start MCP servers / touch the network")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
