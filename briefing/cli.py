"""`vesper briefing` — inspect the briefing engine.

    vesper briefing                 # the compact block the planner receives (+ token count)
    vesper briefing --explain       # every item's score and exactly why
    vesper briefing --spoken        # the 20-30 s script
    vesper briefing --refresh       # force a collector run first (starts the MCP servers)
    vesper briefing --cache-only    # never touch the network; read data/briefing.db as it is
    vesper briefing --json          # structured output
    vesper briefing --known         # who is in the known-correspondent set (derived from sent mail)
    vesper briefing --mark ID priority|ignore [--domain]   # record a local feedback rule
    vesper briefing --rules [--remove N]                   # list / remove those rules

Read-only against Gmail and Calendar. It writes only the local cache file and, when you
run --mark, your rules file (data/briefing_rules.json).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from typing import List, Optional

from briefing.builder import Briefing, build_briefing, load_known, render_tool_result, spoken_script
from briefing.config import BriefingConfig, load_briefing_config
from briefing.items import Item
from briefing.rules import ACTIONS, RuleSet
from briefing.sanitize import clean_text, display_name
from briefing.scorer import sender_address
from briefing.service import BriefingService

MIN_REF_CHARS = 6


def _age(ts: float, now: float) -> str:
    seconds = max(0, now - ts)
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def short_id(item_id: str) -> str:
    """The id shown in --explain and accepted by --mark: the Gmail id without its prefix."""
    return item_id.split(":", 1)[-1][:10]


def explain_table(b: Briefing) -> str:
    lines = [f"{'#':>2}  {'ID':<10}  {'SCORE':>5}  {'SOURCE':<8} {'WHEN':<18} {'FROM / TITLE':<62} WHY"]
    for i, s in enumerate(b.all_scored, 1):
        it = s.item
        if it.source == "calendar":
            when = datetime.fromtimestamp(it.start_at or 0).strftime("%a %H:%M")
            who = f"{clean_text(it.title, 58)}"
        else:
            when = _age(it.timestamp, b.as_of)
            who = f"{display_name(it.sender, 20)} — {clean_text(it.title, 38)}"
        flag = " [excluded]" if s.excluded else ""
        lines.append(f"{i:>2}  {short_id(it.id) if it.source == 'gmail' else '-':<10}  {s.score:>5}  {it.source:<8} "
                     f"{when:<18} {who:<62} {s.explain()}{flag}")
    if len(lines) == 1:
        lines.append("   (cache is empty — try --refresh)")
    return "\n".join(lines)


def status_lines(b: Briefing) -> List[str]:
    return ["sources: " + "; ".join(s.describe() + ("" if s.available else " [tools unavailable]") for s in b.sources),
            f"unread mail: {b.total_unread} | priority: {len(b.priority)} | other: {b.other_count} "
            f"({b.bulk_count} bulk) | meetings shown: {len(b.meetings)}"]


# --------------------------------------------------------------------- known / rules / mark

def cmd_known(cfg: BriefingConfig, cache) -> int:
    known = load_known(cache, cfg)
    _, updated = cache.get_meta("known_correspondents")
    if not known.addresses and not known.domains:
        print(f"No known correspondents yet: nothing found in the last {cfg.sent_days} days of sent mail "
              "(or the sent summary has not been collected: run `vesper briefing --refresh --known`).")
        return 0
    when = datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M") if updated else "unknown"
    print(f"Known correspondents — derived from {known.sent_messages} sent message(s) in the last "
          f"{known.window_days} days (header addresses only; no message text is read or stored). Updated {when}.")
    print(f"\nAddresses ({len(known.addresses)}) — mail from these gets +{cfg.weights.known_correspondent:g}:")
    for addr, n in sorted(known.addresses.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:>3}×  {addr}")
    print(f"\nDomains ({len(known.domains)}) — mail from these gets +{cfg.weights.known_domain:g} "
          "(public mailbox providers such as gmail.com are never listed):")
    for dom, n in sorted(known.domains.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:>3}×  {dom}")
    return 0


def cmd_rules(cfg: BriefingConfig, remove: Optional[int]) -> int:
    rules = RuleSet(cfg.rules_path)
    if remove is not None:
        gone = rules.remove(remove)
        if gone is None:
            print(f"No rule #{remove}. List them with: vesper briefing --rules")
            return 1
        print(f"Removed rule {gone.describe()}. Future briefings no longer apply it.")
        return 0
    if not rules.rules:
        print(f"No rules yet ({cfg.rules_path}). Create one with: vesper briefing --mark <ID> priority|ignore")
        return 0
    print(f"Rules in {cfg.rules_path} (plain JSON; applied by briefing/scorer.py; nothing is learned):")
    for r in rules.rules:
        print("  " + r.describe())
    print("Remove one with: vesper briefing --rules --remove <N>")
    return 0


def resolve_item(cache, ref: str) -> Item:
    """An item id, with or without the 'gmail:' prefix, or any unique prefix of at least
    MIN_REF_CHARS characters (the ID column of --explain)."""
    ref = (ref or "").strip()
    if ref.startswith("gmail:"):
        ref = ref[len("gmail:"):]
    if len(ref) < MIN_REF_CHARS:
        raise SystemExit(f"'{ref}' is too short: give at least {MIN_REF_CHARS} characters of the ID column from --explain.")
    matches = [i for i in cache.all_items("gmail") if i.id.split(":", 1)[-1].startswith(ref)]
    if not matches:
        raise SystemExit(f"No cached mail item matches '{ref}'. (Only mail can be marked; list ids with --explain.)")
    if len(matches) > 1:
        raise SystemExit(f"'{ref}' matches {len(matches)} items; give more characters.")
    return matches[0]


def cmd_mark(cfg: BriefingConfig, cache, ref: str, action: str, by_domain: bool, now: float) -> int:
    if action not in ACTIONS:
        raise SystemExit(f"action must be one of: {', '.join(ACTIONS)}")
    item = resolve_item(cache, ref)
    address = sender_address(item.sender)
    if not address:
        raise SystemExit("That item has no sender address to build a rule from.")
    scope, value = ("domain", address.rsplit("@", 1)[1]) if by_domain else ("sender", address)
    rules = RuleSet(cfg.rules_path)
    rule = rules.add(scope, value, action)
    w = cfg.weights
    effect = (f"+{w.rule_priority:g} points, and the newsletter/promo/noreply/mailing-list penalties are waived"
              if action == "priority" else f"{w.rule_ignore:g} points, so it never reaches the priority list")
    target = (f"every sender at @{value} (and its subdomains)" if scope == "domain" else f"the address {value}")
    print(f"Recorded rule {rule.describe()}")
    print(f"  What it does: mail from {target} now gets {effect}.")
    print(f"  What was stored: only the {scope} ('{value}') and the action — no subject, no text. "
          f"File: {cfg.rules_path} (plain JSON; edit or delete it any time).")
    print("  What it is not: there is no learning model; the rule is applied by briefing/scorer.py and nothing else.")
    if scope == "sender":
        print(f"  (Use --domain to apply it to everyone at @{address.rsplit('@', 1)[1]} instead.)")
    print(f"  Undo: vesper briefing --rules --remove {rule.id}")
    b = build_briefing(cache, cfg, now)
    for rank, s in enumerate(b.all_scored, 1):
        if s.item.id == item.id:
            print(f"  That message now scores {s.score} (rank {rank}): {s.explain()}")
            break
    return 0


async def _run(args: argparse.Namespace) -> int:
    cfg = load_briefing_config()
    # rules / mark / known need only the local cache + rules file — never the network
    if args.rules:
        return cmd_rules(cfg, args.remove)
    if args.mark:
        from briefing.cache import BriefingCache

        cache = BriefingCache(cfg.cache_path)
        try:
            return cmd_mark(cfg, cache, args.mark[0], args.mark[1], args.domain, time.time())
        finally:
            cache.close()
    if args.cache_only or (args.known and not args.refresh):
        args.cache_only = True
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

    if args.known:
        code = cmd_known(cfg, service.cache)
        service.cache.close()
        return code
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
    p = argparse.ArgumentParser(
        prog="vesper briefing",
        description="Inspect and tune the briefing engine. Read-only against Gmail and Calendar; the only things "
                    "it ever writes are the local cache and your own rules file.")
    p.add_argument("--explain", action="store_true", help="print each item's score and why")
    p.add_argument("--spoken", action="store_true", help="print the spoken script")
    p.add_argument("--json", action="store_true")
    p.add_argument("--refresh", action="store_true", help="force a collector run first")
    p.add_argument("--cache-only", action="store_true", help="do not start MCP servers / touch the network")
    p.add_argument("--known", action="store_true", help="list the known correspondents derived from your sent mail")
    p.add_argument("--mark", nargs=2, metavar=("ITEM_ID", "priority|ignore"),
                   help="record a local rule from a mail item (ID column of --explain)")
    p.add_argument("--domain", action="store_true", help="with --mark: apply the rule to the sender's whole domain")
    p.add_argument("--rules", action="store_true", help="list your rules")
    p.add_argument("--remove", type=int, metavar="N", help="with --rules: remove rule N")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
