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
    ... --redact                    # --explain/--json/--known/--mark/--rules with every sender, subject,
                                    # snippet and domain replaced by <sender:N>/<subject:N>/... (fail closed)

Read-only against Gmail and Calendar. It writes only the local cache file and, when you
run --mark, your rules file (data/briefing_rules.json).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from typing import List, Optional

from briefing.builder import Briefing, build_briefing, load_known, render_tool_result, spoken_script
from briefing.config import BriefingConfig, load_briefing_config
from briefing.items import Item
from briefing.redact import RedactionLeak, Redactor, safe_reason, scan_fields
from briefing.rules import ACTIONS, Rule, RuleSet
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


def row_fields(s, b: Briefing, red: Optional[Redactor] = None) -> dict:
    """The values of one --explain row. With `red`, NO real sender/subject/title is ever produced
    here, so whatever layout formats these fields cannot leak them."""
    it = s.item
    if it.source == "calendar":
        when = datetime.fromtimestamp(it.start_at or 0).strftime("%a %H:%M")
        who = red.subject(it.title) if red else f"{clean_text(it.title, 58)}"
    else:
        when = _age(it.timestamp, b.as_of)
        who = (f"{red.sender(it.sender)} — {red.subject(it.title)}" if red
               else f"{display_name(it.sender, 20)} — {clean_text(it.title, 38)}")
    why = (", ".join(f"{r.delta:+g} {safe_reason(r.label, red)}" for r in s.reasons) or "no signals") if red else s.explain()
    return {"id": short_id(it.id) if it.source == "gmail" else "-", "score": s.score, "source": it.source,
            "when": when, "who": who, "why": why, "flag": " [excluded]" if s.excluded else ""}


def explain_table(b: Briefing, red: Optional[Redactor] = None) -> str:
    lines = [f"{'#':>2}  {'ID':<10}  {'SCORE':>5}  {'SOURCE':<8} {'WHEN':<18} {'FROM / TITLE':<62} WHY"]
    for i, s in enumerate(b.all_scored, 1):
        f = row_fields(s, b, red)
        lines.append(f"{i:>2}  {f['id']:<10}  {f['score']:>5}  {f['source']:<8} "
                     f"{f['when']:<18} {f['who']:<62} {f['why']}{f['flag']}")
    if len(lines) == 1:
        lines.append("   (cache is empty — try --refresh)")
    return "\n".join(lines)


def status_lines(b: Briefing, red: Optional[Redactor] = None) -> List[str]:
    sources = b.sources if red is None else [
        replace(s, error="(details withheld)" if s.error else None) for s in b.sources]
    return ["sources: " + "; ".join(s.describe() + ("" if s.available else " [tools unavailable]") for s in sources),
            f"unread mail: {b.total_unread} | priority: {len(b.priority)} | other: {b.other_count} "
            f"({b.bulk_count} bulk) | meetings shown: {len(b.meetings)}"]


# --------------------------------------------------------------------- known / rules / mark

def cmd_known(cfg: BriefingConfig, cache, red: Optional[Redactor] = None) -> int:
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
        print(f"  {n:>3}×  {red.address(addr) if red else addr}")
    print(f"\nDomains ({len(known.domains)}) — mail from these gets +{cfg.weights.known_domain:g} "
          "(public mailbox providers such as gmail.com are never listed):")
    for dom, n in sorted(known.domains.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:>3}×  {red.domain(dom) if red else dom}")
    return 0


def describe_rule(r: Rule, red: Optional[Redactor] = None) -> str:
    if red is None:
        return r.describe()
    value = red.rule_value(r.scope, r.value)
    who = f"mail from {value}" if r.scope == "sender" else f"all mail from @{value} (and its subdomains)"
    return f"#{r.id}: {who} -> {r.action.upper()}"


def cmd_rules(cfg: BriefingConfig, remove: Optional[int], red: Optional[Redactor] = None) -> int:
    rules = RuleSet(cfg.rules_path)
    if remove is not None:
        gone = rules.remove(remove)
        if gone is None:
            print(f"No rule #{remove}. List them with: vesper briefing --rules")
            return 1
        print(f"Removed rule {describe_rule(gone, red)}. Future briefings no longer apply it.")
        return 0
    if not rules.rules:
        print(f"No rules yet ({cfg.rules_path}). Create one with: vesper briefing --mark <ID> priority|ignore")
        return 0
    print(f"Rules in {cfg.rules_path} (plain JSON; applied by briefing/scorer.py; nothing is learned):")
    for r in rules.rules:
        print("  " + describe_rule(r, red))
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


def cmd_mark(cfg: BriefingConfig, cache, ref: str, action: str, by_domain: bool, now: float,
             red: Optional[Redactor] = None) -> int:
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
    shown = (red.rule_value(scope, value) if red else value)
    target = (f"every sender at @{shown} (and its subdomains)" if scope == "domain" else f"the address {shown}")
    print(f"Recorded rule {describe_rule(rule, red)}")
    print(f"  What it does: mail from {target} now gets {effect}.")
    print(f"  What was stored: only the {scope} ('{shown}') and the action — no subject, no text. "
          f"File: {cfg.rules_path} (plain JSON; edit or delete it any time).")
    print("  What it is not: there is no learning model; the rule is applied by briefing/scorer.py and nothing else.")
    if scope == "sender":
        other = address.rsplit('@', 1)[1]
        print(f"  (Use --domain to apply it to everyone at @{red.domain(other) if red else other} instead.)")
    print(f"  Undo: vesper briefing --rules --remove {rule.id}")
    b = build_briefing(cache, cfg, now)
    for rank, s in enumerate(b.all_scored, 1):
        if s.item.id == item.id:
            why = ", ".join(f"{r.delta:+g} {safe_reason(r.label, red)}" for r in s.reasons) if red else s.explain()
            print(f"  That message now scores {s.score} (rank {rank}): {why}")
            break
    return 0


def seed_guard(red: Redactor, cfg: BriefingConfig, cache) -> None:
    """Teach the guard EVERYTHING sensitive that is stored locally — every cached sender, subject,
    snippet and title, the known-correspondent set and the rules — whether or not this command
    prints it. That is what makes an unexpected print path fail instead of leak."""
    scan_fields(red, cache.all_items())
    known = load_known(cache, cfg)
    for addr in known.addresses:
        red.seed_identity(addr)
    for dom in known.domains:
        red.seed_identity(dom)
    for r in RuleSet(cfg.rules_path).rules:
        red.seed_identity(r.value)


#: Views that interleave real text into sentences / a token budget; there is no safe way to
#: placeholder them after the fact, so --redact refuses them instead of guessing.
REDACT_UNSUPPORTED = "--redact supports --explain, --json, --known, --mark and --rules (not the spoken script or the planner block)."


async def _run(args: argparse.Namespace) -> int:
    cfg = load_briefing_config()
    red: Optional[Redactor] = getattr(args, "redactor", None)
    if red is not None and (args.spoken or not (args.explain or args.json or args.known or args.mark or args.rules)):
        print(REDACT_UNSUPPORTED, file=sys.stderr)
        return 2
    # rules / mark / known need only the local cache + rules file — never the network
    if args.rules:
        if red is not None:
            from briefing.cache import BriefingCache

            cache = BriefingCache(cfg.cache_path)
            try:
                seed_guard(red, cfg, cache)
            finally:
                cache.close()
        return cmd_rules(cfg, args.remove, red)
    if args.mark:
        from briefing.cache import BriefingCache

        cache = BriefingCache(cfg.cache_path)
        try:
            if red is not None:
                seed_guard(red, cfg, cache)
            return cmd_mark(cfg, cache, args.mark[0], args.mark[1], args.domain, time.time(), red)
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
                note = "ok" if r.ok else ("FAILED (details withheld by --redact)" if red is not None else f"FAILED: {r.error}")
                print(f"[briefing] {r.name}: {note} ({r.new_items} item(s) fetched)", file=sys.stderr)
        finally:
            await bridge.stop()

    if red is not None:
        seed_guard(red, cfg, service.cache)
    if args.known:
        code = cmd_known(cfg, service.cache, red)
        service.cache.close()
        return code
    b = service.briefing(time.time())
    if args.json:
        if red is None:
            payload = {
                "as_of": b.as_of, "unread": b.total_unread,
                "meetings": [{"title": s.item.title, "start": s.item.start_at, "score": s.score} for s in b.meetings],
                "priority": [{"id": s.item.id, "score": s.score, "why": s.explain()} for s in b.priority],
                "sources": [vars(s) for s in b.sources],
            }
        else:  # every string is a placeholder, an allowlisted label, or a fixed word
            payload = {
                "as_of": b.as_of, "unread": b.total_unread,
                "meetings": [{"title": red.subject(s.item.title), "start": s.item.start_at, "score": s.score}
                             for s in b.meetings],
                "priority": [{"id": s.item.id, "score": s.score,
                              "why": ", ".join(f"{r.delta:+g} {safe_reason(r.label, red)}" for r in s.reasons)}
                             for s in b.priority],
                "sources": [{"name": x.name, "available": x.available, "age_seconds": x.age_seconds,
                             "stale": x.stale, "error": "(details withheld)" if x.error else None}
                            for x in b.sources],
            }
        print(json.dumps(payload, indent=2, default=str))
    elif args.explain:
        print(explain_table(b, red))
        print()
        print("\n".join(status_lines(b, red)))
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
    p.add_argument("--redact", action="store_true",
                   help="replace every sender, subject, snippet and domain with a placeholder (<sender:3>); "
                        "fails closed: output that cannot be proven clean is withheld")
    return p


def run_redacted(args: argparse.Namespace) -> int:
    """Run with every byte of stdout/stderr (and any error text) buffered, then released only if the
    guard finds none of the locally stored sensitive values in it. A leak withholds ALL output."""
    red = Redactor()
    args.redactor = red
    out, err = io.StringIO(), io.StringIO()
    code = 1
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = asyncio.run(_run(args))
        except KeyboardInterrupt:
            code = 130
        except SystemExit as exc:
            if isinstance(exc.code, str):
                print(exc.code, file=sys.stderr)
                code = 1
            else:
                code = exc.code if isinstance(exc.code, int) else 1
        except Exception as exc:  # the message could quote content: keep only the type
            print(f"error: {type(exc).__name__} (details withheld by --redact)", file=sys.stderr)
            code = 1
    try:
        text_out, text_err = red.guard(out.getvalue()), red.guard(err.getvalue())
    except RedactionLeak as leak:
        sys.stderr.write(f"REDACTION GUARD TRIPPED: output withheld ({leak.count} sensitive value(s) would have "
                         "been printed). Nothing was printed.\n")
        return 3
    sys.stdout.write(text_out)
    sys.stderr.write(text_err)
    return code if isinstance(code, int) else 1


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    if args.redact:
        return run_redacted(args)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
