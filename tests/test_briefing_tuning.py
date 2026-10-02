"""Scorer tuning: known correspondents, personal mail, the important-automated allowlist, the
strong-urgency cap on bulk penalties, feedback rules (--mark / --rules / --known), the local
config overlay, and a calibration regression on a synthetic copy of the real inbox's shape."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from briefing.builder import build_briefing
from briefing.cache import BriefingCache
from briefing.config import BriefingConfig, from_dict, load_briefing_config
from briefing.items import Item
from briefing.known import KnownSet, derive_known, domain_matches
from briefing.rules import RuleSet
from briefing.scorer import ScoringContext, score_item
from briefing.service import BriefingService

NOW = datetime(2026, 10, 2, 8, 0).timestamp()
H = 3600.0


def mail(subject="Hello", sender="Pat Person <pat@example.com>", snippet="", age_h=1.0, **kw) -> Item:
    signals = kw.pop("signals", {})
    return Item(id=f"gmail:{abs(hash((subject, sender, age_h))) % 10**10:010d}", source="gmail", sender=sender,
                title=subject, snippet=snippet, timestamp=NOW - age_h * H, signals=signals, **kw)


def ctx(**kw) -> ScoringContext:
    return ScoringContext(now=NOW, **kw)


def cfg(**over) -> BriefingConfig:
    return from_dict(over)


def reasons(s) -> str:
    return s.explain()


# ===================================================================== known correspondents

SENT = {
    "recipient_counts": {
        "Prof Rao <prof@college.edu>": 5, "prof2@college.edu": 2, "friend@gmail.com": 3,
        "noreply@service.com": 4, "do-not-reply@bank.com": 1, "recruiter@corp.io": 1, "broken-address": 2,
        "PAT@EXAMPLE.COM": 1, "pat@example.com": 2,
    },
    "message_count": 21, "recipients": [], "thread_ids": [],
}


def test_derive_known_addresses_domains_and_counts():
    k = derive_known(SENT, cfg(collect={"sent_days": 90}))
    assert k.addresses["prof@college.edu"] == 5 and k.addresses["prof2@college.edu"] == 2
    assert k.addresses["pat@example.com"] == 3                           # case-folded duplicates merge
    assert k.domains["college.edu"] == 7 and k.domains["corp.io"] == 1 and k.domains["example.com"] == 3
    assert (k.window_days, k.sent_messages) == (90, 21)


@pytest.mark.parametrize("bad", ["noreply@service.com", "do-not-reply@bank.com", "broken-address"])
def test_derive_known_drops_automated_and_malformed_addresses(bad):
    k = derive_known(SENT, cfg())
    assert bad not in k.addresses
    assert not any(d in k.domains for d in ("service.com", "bank.com"))   # an automated reply never makes a service "known"


def test_public_mailbox_domains_are_never_known_domains_but_the_address_is():
    k = derive_known(SENT, cfg())
    assert "friend@gmail.com" in k.addresses and "gmail.com" not in k.domains
    assert derive_known({"recipient_counts": {"x@mail.yahoo.com": 1}}, cfg()).domains == {}   # subdomain of a public one


def test_derive_known_from_an_older_cache_entry_with_addresses_only():
    k = derive_known({"recipients": ["a@school.edu", "b@school.edu"]}, cfg())
    assert k.addresses == {"a@school.edu": 1, "b@school.edu": 1} and k.domains == {"school.edu": 2}


@pytest.mark.parametrize("summary", [None, {}, {"recipients": []}])
def test_derive_known_empty_inputs(summary):
    k = derive_known(summary, cfg())
    assert k.addresses == {} and k.domains == {}


def test_known_set_roundtrips_and_stores_nothing_but_counts():
    k = derive_known(SENT, cfg())
    data = k.to_dict()
    assert set(data) == {"addresses", "domains", "window_days", "sent_messages"}      # no subjects, bodies or snippets
    assert KnownSet.from_dict(data) == k


@pytest.mark.parametrize("domain,known,expected", [
    ("college.edu", {"college.edu"}, True), ("mail.college.edu", {"college.edu"}, True),
    ("notcollege.edu", {"college.edu"}, False), ("edu", {"college.edu"}, False), ("x.com", set(), False),
])
def test_domain_matching_includes_subdomains_only(domain, known, expected):
    assert domain_matches(domain, known) is expected


@pytest.mark.parametrize("label,sender,context,expected_delta,expected_label", [
    ("exact address", "Prof <prof@college.edu>", dict(sent_recipients={"prof@college.edu"}, known_domains={"college.edu"}),
     25, "known correspondent"),
    ("domain only", "Registrar <office@college.edu>", dict(known_domains={"college.edu"}), 10, "known domain"),
    ("subdomain", "Reg <o@mail.college.edu>", dict(known_domains={"college.edu"}), 10, "known domain"),
    ("stranger", "Rando <r@other.com>", dict(sent_recipients={"prof@college.edu"}, known_domains={"college.edu"}), 0, None),
    ("address beats domain, no double count", "Prof <prof@college.edu>",
     dict(sent_recipients={"prof@college.edu"}, known_domains={"college.edu"}), 25, "known correspondent"),
])
def test_known_boosts(label, sender, context, expected_delta, expected_label):
    base = score_item(mail(sender="Rando <r@other.com>"), ctx(), cfg())
    s = score_item(mail(sender=sender), ctx(**context), cfg())
    assert s.score - base.score == expected_delta, (label, reasons(s))
    assert (expected_label in reasons(s)) if expected_label else ("known" not in reasons(s))


def test_known_boosts_are_configurable():
    s = score_item(mail(sender="P <p@college.edu>"), ctx(sent_recipients={"p@college.edu"}),
                   cfg(weights={"known_correspondent": 60}))
    assert any(r.delta == 60 for r in s.reasons)


# ============================================================================= personal mail

@pytest.mark.parametrize("label,kwargs,personal", [
    ("plain", dict(), True),
    ("noreply", dict(sender="Shop <noreply@shop.com>"), False),
    ("list header", dict(signals={"has_list_unsubscribe": True}), False),
    ("promo label", dict(signals={"labels": ["CATEGORY_PROMOTIONS"]}), False),
    ("newsletter wording", dict(subject="Your weekly digest"), False),
    ("auto-submitted", dict(signals={"auto_submitted": "auto-replied"}), False),
])
def test_personal_mail_boost_only_without_any_bulk_signal(label, kwargs, personal):
    s = score_item(mail(**kwargs), ctx(), cfg())
    assert ("personal mail" in reasons(s)) is personal, label


def test_allowlisted_automated_mail_is_not_personal_mail():
    s = score_item(mail(sender="GitHub <notifications@github.com>"), ctx(), cfg(important_automated={"domains": ["github.com"]}))
    assert "personal mail" not in reasons(s)


# ============================================================ important-automated allowlist

ALLOW = {"important_automated": {"domains": ["github.com", "vercel.com"], "patterns": [r"\.edu$", "placement portal"]}}
PENALIZED = dict(sender="Notify <noreply@{d}>", signals={"labels": ["CATEGORY_PROMOTIONS"], "has_list_unsubscribe": True,
                                                         "auto_submitted": "auto-generated"},
                 subject="Weekly digest: sale 50% off")


def _penalized(domain: str, **over) -> Item:
    kw = dict(PENALIZED)
    kw["sender"] = kw["sender"].format(d=domain)
    kw.update(over)
    return mail(**kw)


@pytest.mark.parametrize("label,domain,waived", [
    ("exact domain", "github.com", True), ("second domain", "vercel.com", True),
    ("subdomain", "notifications.github.com", True), ("lookalike suffix", "evilgithub.com", False),
    ("pattern anchored at the address end", "cs.college.edu", True), ("unlisted", "shop.com", False),
])
def test_allowlist_waives_every_bulk_penalty(label, domain, waived):
    s = score_item(_penalized(domain), ctx(), cfg(**ALLOW))
    if waived:
        assert "bulk penalties waived" in reasons(s), label
        assert not any(r.delta < 0 for r in s.reasons) and not s.bulk
        assert s.score == 25                                      # 15 base + 10 fresh: neutral, not boosted
    else:
        assert "waived" not in reasons(s) and s.bulk and s.score == 0, label


def test_allowlist_boost_is_off_by_default_and_configurable():
    plain = score_item(_penalized("github.com"), ctx(), cfg(**ALLOW))
    assert plain.score == 25 and "allowlist boost" not in reasons(plain)
    boosted = score_item(_penalized("github.com"), ctx(), cfg(**ALLOW, weights={"important_automated": 15}))
    assert boosted.score == 40 and "+15 important automated sender (allowlist boost)" in reasons(boosted)
    unlisted = score_item(_penalized("shop.com"), ctx(), cfg(**ALLOW, weights={"important_automated": 15}))
    assert "allowlist boost" not in reasons(unlisted) and unlisted.score == 0


def test_allowlist_pattern_can_match_the_subject():
    s = score_item(_penalized("random.com", subject="Placement portal: new posting"), ctx(), cfg(**ALLOW))
    assert "waived" in reasons(s)


def test_allowlist_is_empty_by_default_and_shipped_examples_are_commented():
    assert load_briefing_config(None).important_domains == [] and load_briefing_config(None).important_patterns == []
    text = (Path(__file__).resolve().parents[1] / "config" / "briefing.yaml").read_text()
    for name in ("github.com", "vercel.com", "stripe.com", "devpost.com"):
        assert re.search(rf"^\s*#.*{re.escape(name)}", text, re.M), name        # present as a comment
        assert not re.search(rf"^\s*-\s*{re.escape(name)}", text, re.M), name   # and not active


def test_urgency_words_apply_to_allowlisted_automated_mail():
    s = score_item(_penalized("github.com", subject="CI failed: deadline for submission"), ctx(), cfg(**ALLOW))
    assert "urgency words" in reasons(s) and s.score >= 30


# ================================================================ strong-urgency override

PEN_ALL = dict(sender="Portal <noreply@portal.com>", signals={"has_list_unsubscribe": True})     # noreply -25, list -20


@pytest.mark.parametrize("label,kwargs,capped", [
    ("deadline in subject", dict(subject="Submission deadline today"), True),
    ("term only in the snippet", dict(subject="Update", snippet="your payment failed, please retry"), True),
    ("interview", dict(subject="Interview slot confirmed"), True),
    ("action required", dict(subject="Action required on your account"), True),
    ("shortlisted", dict(subject="You have been shortlisted"), True),
    ("evaluation", dict(subject="Evaluation results"), True),
    ("no strong term", dict(subject="Your monthly statement"), False),
    ("weak urgency word only", dict(subject="Report due tomorrow"), False),
])
def test_strong_urgency_caps_combined_bulk_penalties_at_minus_ten(label, kwargs, capped):
    s = score_item(mail(**{**PEN_ALL, **kwargs}), ctx(), cfg())
    penalties = sum(r.delta for r in s.reasons if r.label in ("noreply sender", "mailing-list header"))
    capture = [r for r in s.reasons if r.label.startswith("strong urgency")]
    assert penalties == -45
    if capped:
        assert capture and penalties + capture[0].delta == -10, label           # net bulk effect is exactly -10
        assert not s.bulk and "urgency words" in reasons(s)
    else:
        assert not capture and s.bulk and s.score == 0, label


def test_cap_is_a_cap_not_a_floor_when_penalties_are_already_small():
    c = cfg(weights={"noreply_sender": -4, "mailing_list_header": -3})
    s = score_item(mail(**{**PEN_ALL, "subject": "Deadline today"}), ctx(), c)
    assert not [r for r in s.reasons if r.label.startswith("strong urgency")]    # -7 is already above -10


def test_cap_value_is_configurable():
    s = score_item(mail(**{**PEN_ALL, "subject": "Deadline today"}), ctx(), cfg(weights={"strong_urgency_bulk_cap": -20}))
    assert sum(r.delta for r in s.reasons if r.label in ("noreply sender", "mailing-list header", "strong urgency (deadline): bulk penalties capped at -20")) == -20


@pytest.mark.parametrize("label,kwargs,lifted", [
    ("promo wording + only 'expires'", dict(subject="Flash sale! Coupon expires tonight"), False),
    ("promo wording + a real strong term", dict(subject="Flash sale ends; submission deadline Friday"), True),
    ("Gmail Promotions label + 'action required'", dict(subject="Action required: complete your profile",
                                                         signals={"labels": ["CATEGORY_PROMOTIONS"]}), False),
    ("Gmail Social label + 'interview'", dict(subject="Interview tips", signals={"labels": ["CATEGORY_SOCIAL"]}), False),
    ("Gmail Updates label + 'deadline'", dict(subject="Deadline reminder", signals={"labels": ["CATEGORY_UPDATES"]}), True),
])
def test_promotions_do_not_get_the_override(label, kwargs, lifted):
    s = score_item(mail(**{**PEN_ALL, **kwargs}), ctx(), cfg())
    assert bool([r for r in s.reasons if r.label.startswith("strong urgency")]) is lifted, (label, reasons(s))


def test_gmail_promotions_respect_can_be_turned_off():
    kw = {**PEN_ALL, "subject": "Action required: complete your profile", "signals": {"labels": ["CATEGORY_PROMOTIONS"]}}
    assert not [r for r in score_item(mail(**kw), ctx(), cfg()).reasons if r.label.startswith("strong urgency")]
    off = cfg(strong_urgency_respects_gmail_promotions=False)
    assert [r for r in score_item(mail(**kw), ctx(), off).reasons if r.label.startswith("strong urgency")]


@pytest.mark.parametrize("subject,age_h,surfaces", [
    ("Payment failed for your subscription", 50, True),         # stale but strong enough on its own
    ("Interview scheduled for Monday", 50, True),
    ("Submission deadline: evaluation round", 50, True),
    ("Action required on your account", 2, True),               # fresh
    ("Action required on your account", 50, False),             # weaker term, stale
    ("Your monthly statement", 2, False),
])
def test_strongly_urgent_automated_mail_reaches_the_priority_bar(subject, age_h, surfaces):
    s = score_item(mail(**{**PEN_ALL, "subject": subject, "age_h": age_h}), ctx(), cfg())
    assert (s.score >= cfg().min_priority_score) is surfaces, (subject, s.score, reasons(s))


# ===================================================================================== rules

def test_rule_store_persists_numbers_replaces_and_removes(tmp_path):
    path = tmp_path / "rules.json"
    rs = RuleSet(path)
    a = rs.add("sender", "Alice@Example.com", "priority", now=1.0)
    b = rs.add("domain", "@news.example.org", "ignore", now=2.0)
    assert (a.id, a.value, b.id, b.value) == (1, "alice@example.com", 2, "news.example.org")
    again = rs.add("sender", "alice@example.com", "ignore", now=3.0)           # same target: decision replaced
    assert [r.id for r in rs.rules] == [2, 3] and again.action == "ignore"
    assert rs.remove(2).value == "news.example.org" and rs.remove(99) is None
    reloaded = RuleSet(path)
    assert [(r.id, r.scope, r.value, r.action) for r in reloaded.rules] == [(3, "sender", "alice@example.com", "ignore")]
    nxt = reloaded.add("sender", "z@z.com", "priority")
    assert nxt.id == 4                                                          # ids are never reused downward


def test_rule_file_is_plain_inspectable_json_with_no_message_text(tmp_path):
    path = tmp_path / "rules.json"
    RuleSet(path).add("sender", "a@b.com", "priority", now=5.0)
    data = json.loads(path.read_text())
    assert data["version"] == 1 and "note" in data
    assert set(data["rules"][0]) == {"id", "scope", "value", "action", "created"}
    assert not list(tmp_path.glob(".rules-*"))                                  # atomic write left nothing behind


@pytest.mark.parametrize("content", ["", "not json", '{"rules": [{"bad": 1}]}', "[]"])
def test_a_corrupt_rules_file_means_no_rules_not_a_crash(tmp_path, content):
    path = tmp_path / "rules.json"
    path.write_text(content)
    assert RuleSet(path).rules == []


@pytest.mark.parametrize("scope,value,action", [("team", "x", "priority"), ("sender", "x", "boost"), ("sender", " ", "ignore")])
def test_rule_validation(tmp_path, scope, value, action):
    with pytest.raises(ValueError):
        RuleSet(tmp_path / "r.json").add(scope, value, action)


@pytest.mark.parametrize("sender,expected_id", [
    ("A <alice@example.com>", 1),            # exact address
    ("ALICE@EXAMPLE.COM", 1),                # case-insensitive
    ("Bob <bob@example.com>", 2),            # domain rule
    ("Sub <s@mail.example.com>", 2),         # subdomain of the domain rule
    ("X <x@other.com>", None), ("", None),
])
def test_rule_matching_and_precedence(tmp_path, sender, expected_id):
    rs = RuleSet(tmp_path / "r.json")
    rs.add("sender", "alice@example.com", "ignore")           # id 1
    rs.add("domain", "example.com", "priority")               # id 2
    got = rs.match(sender)
    assert (got.id if got else None) == expected_id           # the address rule beats the domain rule


@pytest.mark.parametrize("label,action,item_kw,check", [
    ("priority waives penalties and boosts", "priority",
     dict(sender="News <news@list.example.org>", signals={"has_list_unsubscribe": True}, subject="Weekly digest"),
     lambda s: ("bulk penalties waived" in reasons(s)) and s.score == 15 + 10 + 40 and not s.bulk),
    ("ignore forces it out of priority", "ignore",
     dict(sender="Pat <pat@example.com>", subject="Interview tomorrow ASAP"),
     lambda s: s.score < 30 and s.bulk and "your rule" in reasons(s)),
])
def test_rules_change_scores(tmp_path, label, action, item_kw, check):
    rs = RuleSet(tmp_path / "r.json")
    address = re.search(r"<([^>]+)>", item_kw["sender"]).group(1)
    rs.add("sender", address, action)
    s = score_item(mail(**item_kw), ctx(rules=rs), cfg())
    assert check(s), (label, reasons(s))


def test_domain_rule_applies_to_everyone_there_and_not_to_calendar(tmp_path):
    rs = RuleSet(tmp_path / "r.json")
    rs.add("domain", "spam.example", "ignore")
    assert score_item(mail(sender="a <a@spam.example>"), ctx(rules=rs), cfg()).score == 0
    assert score_item(mail(sender="b <b@ok.example>"), ctx(rules=rs), cfg()).score == 40
    ev = Item(id="calendar:x", source="calendar", sender="spam.example", title="Standup", start_at=NOW + H,
              end_at=NOW + 2 * H, timestamp=NOW + H)
    assert score_item(ev, ctx(rules=rs), cfg()).score == 100


def test_rules_are_inspectable_through_the_explain_reasons(tmp_path):
    rs = RuleSet(tmp_path / "r.json")
    rs.add("sender", "pat@example.com", "priority")
    s = score_item(mail(), ctx(rules=rs), cfg())
    assert "your rule #1: mail from pat@example.com -> PRIORITY" in reasons(s)


# ======================================================================== config overlay

def test_local_config_overlay_is_deep_merged_and_lists_replace(tmp_path):
    (tmp_path / "briefing.yaml").write_text(
        "vips:\n  addresses: [a@x.com]\n  domains: [x.com]\nweights:\n  vip_sender: 45\n  mail_base: 15\n"
        "important_automated:\n  domains: [github.com]\n")
    (tmp_path / "briefing.local.yaml").write_text(
        "vips:\n  addresses: [me@mine.com, boss@corp.com]\nweights:\n  vip_sender: 70\n"
        "important_automated:\n  domains: [school.edu]\n")
    c = load_briefing_config(str(tmp_path / "briefing.yaml"))
    assert c.vip_addresses == ["me@mine.com", "boss@corp.com"]       # list replaced
    assert c.vip_domains == ["x.com"]                                # untouched key survives the merge
    assert c.weights.vip_sender == 70 and c.weights.mail_base == 15  # scalar overridden, sibling kept
    assert c.important_domains == ["school.edu"]


def test_local_overlay_absent_or_broken_is_harmless(tmp_path):
    (tmp_path / "briefing.yaml").write_text("weights:\n  mail_base: 11\n")
    assert load_briefing_config(str(tmp_path / "briefing.yaml")).weights.mail_base == 11
    (tmp_path / "briefing.local.yaml").write_text("::: not yaml :::\n  - [")
    assert load_briefing_config(str(tmp_path / "briefing.yaml")).weights.mail_base == 11


def test_new_defaults():
    c = BriefingConfig()
    assert c.sent_days == 365 and c.sent_max_messages == 500 and c.min_priority_score == 30
    assert c.weights.known_correspondent == 25 and c.weights.known_domain == 10 and c.weights.personal_mail == 15
    assert c.weights.strong_urgency_bulk_cap == -10 and c.weights.fresh_hours == 24
    assert "gmail.com" in c.public_domains and "payment failed" in c.strong_urgency


# ========================================================================= service: derivation

class _Gmail:
    name, read_only = "gmail", True

    def __init__(self, aux):
        self._aux, self.aux_calls = aux, 0

    def available(self):
        return True

    async def fetch_since(self, cursor):
        return []

    def next_cursor(self, cursor, items, now):
        return "1"

    async def live_ids(self):
        return None

    async def aux(self):
        self.aux_calls += 1
        return dict(self._aux)


@pytest.mark.asyncio
async def test_service_derives_and_stores_the_known_set_from_the_sent_summary():
    c = _Gmail({"recipient_counts": {"p@college.edu": 3, "noreply@x.com": 9}, "recipients": ["p@college.edu"],
                "thread_ids": ["t1"], "message_count": 12})
    svc = BriefingService([c], cfg(), BriefingCache(":memory:"), clock=lambda: NOW)
    await svc.refresh()
    known, _ = svc.cache.get_meta("known_correspondents")
    assert known["addresses"] == {"p@college.edu": 3} and known["domains"] == {"college.edu": 3}
    assert known["window_days"] == 365 and known["sent_messages"] == 12
    blob = json.dumps([svc.cache.get_meta(k)[0] for k in ("known_correspondents", "gmail_sent")])
    assert "subject" not in blob and "snippet" not in blob and "body" not in blob


@pytest.mark.asyncio
async def test_changing_the_sent_window_refetches_immediately():
    c = _Gmail({"recipient_counts": {"p@college.edu": 1}})
    svc = BriefingService([c], cfg(collect={"sent_days": 60}), BriefingCache(":memory:"), clock=lambda: NOW)
    await svc.refresh(); await svc.refresh()
    assert c.aux_calls == 1                                    # within the refresh interval, same window
    svc.cfg.sent_days = 365                                    # the user widened the window
    await svc.refresh()
    assert c.aux_calls == 2 and svc.cache.get_meta("known_correspondents")[0]["window_days"] == 365


def test_a_pre_existing_cache_without_the_known_set_still_scores_known_senders():
    cache = BriefingCache(":memory:")
    cache.upsert_items([mail(subject="hi", sender="P <p@college.edu>")], now=NOW)
    cache.record_success("gmail", "1", now=NOW); cache.record_success("calendar", "1", now=NOW)
    cache.set_meta("gmail_sent", {"recipients": ["p@college.edu"], "thread_ids": []}, now=NOW)   # old shape, no known meta
    c = cfg(collect={"rules_path": "/nonexistent/rules.json"})
    b = build_briefing(cache, c, NOW)
    assert "known correspondent" in b.all_scored[0].explain()


# ============================================================================ calibration

def _inbox_like_the_real_one() -> List[Item]:
    """The shape of the real inbox: ONE human mail, ~45 bulk items of every flavour, one
    Gmail-Promotions mail that says 'action required', plus the automated mail that matters."""
    items = [mail(subject="Re: project meeting", sender="Prof Rao <prof@college.edu>", age_h=9, is_reply=True)]
    flavours = [
        dict(sender="Deals <noreply@shop.com>", subject="Flash sale: 50% off", signals={"labels": ["CATEGORY_PROMOTIONS"], "has_list_unsubscribe": True}),
        dict(sender="Digest <digest@news.io>", subject="Your weekly digest", signals={"has_list_unsubscribe": True, "list_id": "<d.news.io>"}),
        dict(sender="Updates <updates@app.com>", subject="New features this month", signals={"labels": ["CATEGORY_UPDATES"], "has_list_unsubscribe": True}),
        dict(sender="Notify <noreply@social.com>", subject="Someone viewed your profile", signals={"labels": ["CATEGORY_SOCIAL"]}),
    ]
    for i in range(45):
        f = dict(flavours[i % len(flavours)])
        f["subject"] = f"{f['subject']} #{i}"
        items.append(mail(age_h=1 + (i % 40), **f))
    items.append(mail(subject="Action required: complete your profile", sender="Network <noreply@social.com>", age_h=3,
                      signals={"labels": ["CATEGORY_PROMOTIONS"], "has_list_unsubscribe": True}))
    items.append(mail(subject="Final submission deadline Friday", sender="Hackathon <noreply@devpost.com>", age_h=5,
                      signals={"has_list_unsubscribe": True}))
    items.append(mail(subject="Payment failed for your subscription", sender="Billing <noreply@vercel.com>", age_h=50,
                      signals={"auto_submitted": "auto-generated"}))
    items.append(mail(subject="[repo] new comment on your pull request", sender="GitHub <notifications@github.com>", age_h=2,
                      signals={"has_list_unsubscribe": True, "list_id": "<repo.github.com>"}))
    return items


def _build(items, **over):
    cache = BriefingCache(":memory:")
    cache.upsert_items(items, now=NOW)
    cache.record_success("gmail", "1", now=NOW); cache.record_success("calendar", "1", now=NOW)
    cache.set_meta("known_correspondents", derive_known({"recipient_counts": {"prof@college.edu": 4}}, cfg()).to_dict(), now=NOW)
    c = cfg(collect={"rules_path": "/nonexistent/rules.json"}, **over)
    return build_briefing(cache, c, NOW), c


def _subjects(b) -> List[str]:
    return [s.item.title for s in b.priority]


def test_calibration_default_config_surfaces_the_human_and_the_strongly_urgent_and_keeps_bulk_out():
    b, c = _build(_inbox_like_the_real_one())
    got = _subjects(b)
    assert got[0] == "Re: project meeting"                                    # the human reply, a known sender, first
    assert "Final submission deadline Friday" in got and "Payment failed for your subscription" in got
    assert len(got) == 3
    assert not any("#" in t for t in got)                                      # none of the 45 bulk items
    assert "Action required: complete your profile" not in got                 # Gmail-Promotions: no override
    assert "[repo] new comment on your pull request" not in got                # allowlist is off by default
    assert b.other_count == len(_inbox_like_the_real_one()) - 3


def test_calibration_with_the_allowlist_a_plain_notification_stays_out_but_an_urgent_one_gets_in():
    allow = {"important_automated": {"domains": ["github.com", "vercel.com", "devpost.com"]}}
    b, _ = _build(_inbox_like_the_real_one(), **allow)
    got = _subjects(b)
    assert "[repo] new comment on your pull request" not in got                # waived but merely informational: 25 < 30
    assert {"Re: project meeting", "Final submission deadline Friday", "Payment failed for your subscription"} <= set(got)
    urgent = _inbox_like_the_real_one() + [mail(subject="[repo] deadline: security alert action required",
                                                sender="GitHub <notifications@github.com>", age_h=1,
                                                signals={"has_list_unsubscribe": True})]
    b2, _ = _build(urgent, **allow)
    assert "[repo] deadline: security alert action required" in _subjects(b2)


def test_calibration_without_the_new_signals_would_have_reported_no_priority_mail():
    """Documents the original failure: with the old weights the human mail scored 20 against a bar of 35."""
    old = dict(weights={"personal_mail": 0, "known_correspondent": 0, "known_domain": 0, "fresh_hours": 6,
                        "strong_urgency_bulk_cap": -999},
               builder={"min_priority_score": 35})
    b, _ = _build(_inbox_like_the_real_one(), **old)
    assert b.priority == []
