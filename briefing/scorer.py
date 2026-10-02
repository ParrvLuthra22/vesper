"""
Deterministic scorer: 0-100 per item, no LLM, every point explained.

    mail      15 base + sender importance + reply signals + urgency words + freshness
              - newsletter / promo / mailing-list / noreply / automated signals
              (clamped to 0..mail_max_score so mail can never outrank an imminent meeting)
    calendar  100 if it starts within `imminent_hours` or is in progress (always top),
              60 later today, 45 tomorrow, +bonus for interview/exam-style titles

All weights live in config/briefing.yaml with a comment each. `score_item` returns
the reasons so `vesper briefing --explain` can show exactly why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parseaddr
from typing import Iterable, List, Optional, Set

from briefing.config import BriefingConfig
from briefing.items import Item
from briefing.sanitize import looks_like_instructions

_CATEGORY_PROMO = {"CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL"}
_CATEGORY_NEWSLETTER = {"CATEGORY_UPDATES", "CATEGORY_FORUMS"}
_BULK_PRECEDENCE = {"bulk", "list", "junk"}


@dataclass
class Reason:
    label: str
    delta: float


@dataclass
class ScoringContext:
    now: float
    sent_threads: Set[str] = field(default_factory=set)       # threads containing mail I sent
    sent_recipients: Set[str] = field(default_factory=set)    # addresses I have written to (lowercase)


@dataclass
class ScoredItem:
    item: Item
    score: int
    reasons: List[Reason]
    excluded: bool = False          # not briefing-relevant (read, ended, beyond the window)
    bulk: bool = False              # newsletter / promo / mailing list / noreply
    suspicious: bool = False        # text reads like instructions to an AI

    def explain(self) -> str:
        return ", ".join(f"{r.delta:+g} {r.label}" for r in self.reasons) or "no signals"


def sender_address(sender: str) -> str:
    return parseaddr(sender or "")[1].lower()


def _compile(patterns: Iterable[str]) -> List["re.Pattern[str]"]:
    out = []
    for p in patterns:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            continue
    return out


def _any(rx: List["re.Pattern[str]"], text: str) -> Optional[str]:
    for r in rx:
        m = r.search(text)
        if m:
            return m.group(0)
    return None


def _is_vip(sender: str, cfg: BriefingConfig) -> Optional[str]:
    name, addr = parseaddr(sender or "")
    addr = addr.lower()
    domain = addr.split("@", 1)[1] if "@" in addr else ""
    if addr and addr in cfg.vip_addresses:
        return "address"
    if domain and domain in cfg.vip_domains:
        return "domain"
    lname = (name or "").lower()
    if lname and any(v and v in lname for v in cfg.vip_names):
        return "name"
    return None


def score_item(item: Item, ctx: ScoringContext, cfg: BriefingConfig) -> ScoredItem:
    if item.source == "calendar":
        return _score_event(item, ctx, cfg)
    return _score_mail(item, ctx, cfg)


# --------------------------------------------------------------------------- mail

def _score_mail(item: Item, ctx: ScoringContext, cfg: BriefingConfig) -> ScoredItem:
    w = cfg.weights
    if not item.unread:
        return ScoredItem(item, 0, [Reason("already read", 0)], excluded=True)

    reasons: List[Reason] = [Reason("unread mail", w.mail_base)]
    text = f"{item.title} {item.snippet}"
    sig = item.signals or {}
    labels = set(sig.get("labels") or [])
    addr = sender_address(item.sender)

    # ---- bulk / automated signals first: they gate the urgency words below ----
    bulk = False
    if _any(_compile(cfg.pattern_noreply), addr or item.sender):
        reasons.append(Reason("noreply sender", w.noreply_sender)); bulk = True
    if sig.get("list_id") or sig.get("has_list_unsubscribe") or str(sig.get("precedence", "")) in _BULK_PRECEDENCE:
        reasons.append(Reason("mailing-list header", w.mailing_list_header)); bulk = True
    if labels & _CATEGORY_PROMO or _any(_compile(cfg.pattern_promo), text):
        reasons.append(Reason("promo signal", w.promo_signal)); bulk = True
    if labels & _CATEGORY_NEWSLETTER or _any(_compile(cfg.pattern_newsletter), f"{item.title} {item.sender}"):
        reasons.append(Reason("newsletter signal", w.newsletter_signal)); bulk = True
    if sig.get("auto_submitted") and str(sig["auto_submitted"]) != "no":
        reasons.append(Reason("auto-submitted", w.auto_submitted)); bulk = True

    # ---- who it is from ----
    vip = _is_vip(item.sender, cfg)
    if vip:
        reasons.append(Reason(f"VIP sender ({vip})", w.vip_sender))
    if addr and addr in ctx.sent_recipients:
        reasons.append(Reason("I've written to this sender before", w.replied_to_sender_before))
    if item.thread_id and item.thread_id in ctx.sent_threads:
        reasons.append(Reason("reply in a thread I sent in", w.reply_in_my_thread))
    elif item.is_reply:
        reasons.append(Reason("is a reply", w.is_reply))

    # ---- urgency words (ignored for bulk: "URGENT: 50% off" earns nothing) ----
    if not bulk:
        total = 0.0
        hits = []
        for word, weight in cfg.urgency_keywords.items():
            if re.search(rf"(?<![\w]){re.escape(word)}(?![\w])", text, re.IGNORECASE):
                total += weight; hits.append(word)
        if hits:
            reasons.append(Reason(f"urgency words ({', '.join(sorted(hits))})", min(total, w.urgency_total_cap)))

    # ---- age ----
    if item.timestamp:
        age_h = max(0.0, (ctx.now - item.timestamp) / 3600.0)
        if age_h <= w.fresh_hours:
            reasons.append(Reason(f"fresh ({age_h:.0f}h old)", w.fresh_bonus))
        elif age_h >= w.stale_days * 24:
            reasons.append(Reason(f"unread for {age_h / 24:.0f} days", w.stale_penalty))

    # ---- hostile content ----
    suspicious = looks_like_instructions(item.title, item.snippet, parseaddr(item.sender or "")[0])
    if suspicious:
        reasons.append(Reason("text reads like instructions to an AI", w.injection_suspected))

    score = max(0.0, min(w.mail_max_score, sum(r.delta for r in reasons)))
    return ScoredItem(item, int(round(score)), reasons, bulk=bulk, suspicious=suspicious)


# ------------------------------------------------------------------------ calendar

def _score_event(item: Item, ctx: ScoringContext, cfg: BriefingConfig) -> ScoredItem:
    w = cfg.weights
    start, end = item.start_at, item.end_at
    if start is None:
        return ScoredItem(item, 0, [Reason("no start time", 0)], excluded=True)
    if end is not None and end <= ctx.now and start < ctx.now:
        return ScoredItem(item, 0, [Reason("already over", 0)], excluded=True)

    now_dt = datetime.fromtimestamp(ctx.now)
    start_dt = datetime.fromtimestamp(start)
    day_delta = (start_dt.date() - now_dt.date()).days
    hours_until = (start - ctx.now) / 3600.0
    in_progress = start <= ctx.now and (end is None or end > ctx.now)

    reasons: List[Reason] = []
    if item.all_day:
        # all-day entries are "in progress" the whole day; never treat them as imminent
        if day_delta == 0:
            reasons.append(Reason("all-day, today", w.meeting_today_later))
        elif day_delta == 1:
            reasons.append(Reason("all-day, tomorrow", w.meeting_tomorrow))
        else:
            return ScoredItem(item, 0, [Reason("beyond the briefing window", 0)], excluded=True)
        reasons.append(Reason("all-day event", w.all_day_event))
    elif in_progress:
        reasons.append(Reason("in progress now", w.meeting_imminent))
    elif 0 <= hours_until <= w.imminent_hours:
        reasons.append(Reason(f"starts within {w.imminent_hours:g}h", w.meeting_imminent))
    elif day_delta == 0:
        reasons.append(Reason("later today", w.meeting_today_later))
    elif day_delta == 1:
        reasons.append(Reason("tomorrow", w.meeting_tomorrow))
    else:
        return ScoredItem(item, 0, [Reason("beyond the briefing window", 0)], excluded=True)

    title = (item.title or "").lower()
    kw = [k for k in cfg.meeting_keywords if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", title)]
    if kw:
        reasons.append(Reason(f"title mentions {', '.join(kw)}", w.meeting_keyword_bonus))

    suspicious = looks_like_instructions(item.title)
    if suspicious:
        reasons.append(Reason("title reads like instructions to an AI", w.injection_suspected))

    score = max(0.0, min(100.0, sum(r.delta for r in reasons)))
    return ScoredItem(item, int(round(score)), reasons, suspicious=suspicious)
