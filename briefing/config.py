"""Typed view over config/briefing.yaml. Defaults mirror the shipped file, so a
missing file or key behaves identically."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "briefing.yaml"


def _w(**kw: float) -> Dict[str, float]:
    return dict(kw)


@dataclass
class Weights:
    mail_base: float = 15
    vip_sender: float = 45
    replied_to_sender_before: float = 15
    reply_in_my_thread: float = 20
    is_reply: float = 5
    fresh_bonus: float = 10
    fresh_hours: float = 6
    stale_penalty: float = -10
    stale_days: float = 7
    urgency_total_cap: float = 30
    noreply_sender: float = -25
    newsletter_signal: float = -25
    promo_signal: float = -30
    mailing_list_header: float = -20
    auto_submitted: float = -15
    injection_suspected: float = -40
    mail_max_score: float = 94
    meeting_imminent: float = 100
    imminent_hours: float = 3
    meeting_today_later: float = 60
    meeting_tomorrow: float = 45
    meeting_keyword_bonus: float = 10
    all_day_event: float = -30


@dataclass
class BriefingConfig:
    # refresh
    enabled: bool = True
    interval_minutes: float = 10
    jitter_fraction: float = 0.2
    max_cache_age_minutes: float = 30
    collector_timeout_seconds: float = 45
    # collect
    gmail_lookback_days: float = 3
    gmail_max_new: int = 50
    snippet_chars: int = 200
    calendar_window_days: int = 2
    sent_days: int = 60
    sent_refresh_hours: float = 24
    cache_path: str = "data/briefing.db"
    # people
    vip_addresses: List[str] = field(default_factory=list)
    vip_domains: List[str] = field(default_factory=list)
    vip_names: List[str] = field(default_factory=list)
    # scoring
    weights: Weights = field(default_factory=Weights)
    urgency_keywords: Dict[str, float] = field(default_factory=lambda: _w(
        urgent=12, asap=12, eod=10, deadline=10, due=8, interview=15, offer=12, exam=12, tomorrow=6))
    meeting_keywords: List[str] = field(default_factory=lambda: [
        "interview", "exam", "offer", "deadline", "review", "demo", "presentation"])
    pattern_noreply: List[str] = field(default_factory=lambda: [
        r"no[-_.]?reply", r"do[-_.]?not[-_.]?reply", r"donotreply", r"mailer-daemon", r"postmaster"])
    pattern_newsletter: List[str] = field(default_factory=lambda: [
        r"newsletter", r"weekly digest", r"daily digest", r"monthly digest", r"round-?up",
        r"view (?:this )?in (?:your )?browser", r"unsubscribe", r"digest"])
    pattern_promo: List[str] = field(default_factory=lambda: [
        r"\d+\s?% off", r"\bsale\b", r"discount", r"coupon", r"promo code", r"limited[- ]time",
        r"free shipping", r"special offer", r"exclusive offer", r"offer ends", r"deal of the day", r"flash sale"])
    # builder
    max_meetings: int = 3
    max_priority_mail: int = 5
    min_priority_score: float = 35
    token_cap: int = 600
    subject_chars: int = 70
    sender_chars: int = 32
    preview_chars: int = 90
    # speech
    address_as: str = "Sir"
    max_words: int = 85
    morning_until_hour: int = 12
    # fast path
    fast_path_enabled: bool = True
    fast_path_phrases: List[str] = field(default_factory=lambda: [
        "good morning", "morning", "morning briefing", "daily briefing", "brief me",
        "give me my briefing", "give me my morning briefing", "whats my day look like",
        "what is my day look like", "what does my day look like"])


def _section(d: Dict[str, Any], name: str) -> Dict[str, Any]:
    v = d.get(name)
    return v if isinstance(v, dict) else {}


def from_dict(data: Optional[Dict[str, Any]]) -> BriefingConfig:
    d = data or {}
    cfg = BriefingConfig()

    coerce = {"float": float, "int": int, "bool": bool, "str": str}

    def put(section: str, mapping: Dict[str, str]) -> None:
        sec = _section(d, section)
        for key, attr in mapping.items():
            if key in sec and sec[key] is not None:
                value = sec[key]
                declared = str(BriefingConfig.__dataclass_fields__[attr].type)  # annotations are strings here
                setattr(cfg, attr, coerce[declared](value) if declared in coerce else value)

    put("refresh", {"enabled": "enabled", "interval_minutes": "interval_minutes", "jitter_fraction": "jitter_fraction",
                    "max_cache_age_minutes": "max_cache_age_minutes",
                    "collector_timeout_seconds": "collector_timeout_seconds"})
    put("collect", {"gmail_lookback_days": "gmail_lookback_days", "gmail_max_new": "gmail_max_new",
                    "snippet_chars": "snippet_chars", "calendar_window_days": "calendar_window_days",
                    "sent_days": "sent_days", "sent_refresh_hours": "sent_refresh_hours", "cache_path": "cache_path"})
    put("vips", {"addresses": "vip_addresses", "domains": "vip_domains", "names": "vip_names"})
    put("builder", {k: k for k in ("max_meetings", "max_priority_mail", "min_priority_score", "token_cap",
                                   "subject_chars", "sender_chars", "preview_chars")})
    put("speech", {"address_as": "address_as", "max_words": "max_words", "morning_until_hour": "morning_until_hour"})
    fp = _section(d, "fast_path")
    if "enabled" in fp:
        cfg.fast_path_enabled = bool(fp["enabled"])
    if isinstance(fp.get("phrases"), list):
        cfg.fast_path_phrases = [str(p) for p in fp["phrases"]]
    for key, value in _section(d, "weights").items():
        if hasattr(cfg.weights, key) and isinstance(value, (int, float)):
            setattr(cfg.weights, key, float(value))
    if isinstance(d.get("urgency_keywords"), dict):
        cfg.urgency_keywords = {str(k).lower(): float(v) for k, v in d["urgency_keywords"].items()}
    if isinstance(d.get("meeting_keywords"), list):
        cfg.meeting_keywords = [str(k).lower() for k in d["meeting_keywords"]]
    pats = _section(d, "patterns")
    for key, attr in (("noreply_sender", "pattern_noreply"), ("newsletter", "pattern_newsletter"), ("promo", "pattern_promo")):
        if isinstance(pats.get(key), list):
            setattr(cfg, attr, [str(p) for p in pats[key]])
    cfg.vip_addresses = [a.lower() for a in cfg.vip_addresses]
    cfg.vip_domains = [x.lower().lstrip("@") for x in cfg.vip_domains]
    cfg.vip_names = [n.lower() for n in cfg.vip_names]
    return cfg


def load_briefing_config(path: Optional[str] = None) -> BriefingConfig:
    import yaml

    target = Path(path or os.environ.get("VESPER_BRIEFING_CONFIG") or DEFAULT_PATH)
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        data = {}
    cfg = from_dict(data)
    # Tests (and anyone who wants the cache elsewhere) redirect the SQLite file.
    cfg.cache_path = os.environ.get("VESPER_BRIEFING_DB") or cfg.cache_path
    return cfg
