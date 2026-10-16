"""The built-in skills. Each is a small registered object: name, patterns, handler (calls existing tools through the
Guardian), spoken formatter, tier floor. English first, plus the Hinglish a user actually says ("aaj ka schedule batao")."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from briefing.sanitize import clean_text, display_name, looks_like_instructions
from router.core import (
    Match, Rule, RouterContext, Skill, SkillResult, SkillRouter, SkillRuntime, SkillUnavailable, normalize)

WHAT = r"(?:what is|whats)"
ASK = r"(?:(?:tell me|show me|give me|let me know|let me see|check)\s+)?"      # optional lead-in that carries no meaning


def R(rule_id: str, pattern: str, validate=None) -> Rule:
    return Rule(rule_id, re.compile(pattern), validate)


def _say(count: int, one: str, many: str) -> str:
    return one if count == 1 else many


def _clock(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def _title(raw: object, limit: int = 45) -> str:
    text = clean_text(raw, limit)
    return "a meeting with a flagged title" if looks_like_instructions(raw) else (text or "an untitled event")


def _parse_dt(value: object) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt.astimezone().replace(tzinfo=None) if dt.tzinfo else dt


def _events(raw: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        raise SkillUnavailable("calendar returned something unreadable")
    if not isinstance(data, list):
        raise SkillUnavailable("calendar returned something unexpected")
    return [e for e in data if isinstance(e, dict)]


# ----------------------------------------------------------------------------- time / date

async def _time(match: Match, rt: SkillRuntime) -> SkillResult:
    return SkillResult(f"It's {_clock(rt.clock())}, {rt.address_as}.")


async def _date(match: Match, rt: SkillRuntime) -> SkillResult:
    now = rt.clock()
    return SkillResult(f"It's {now.strftime('%A')}, {now.day} {now.strftime('%B')}, {rt.address_as}.")


TIME = Skill("time", "what time is it", (
    R("time.plain", rf"{ASK}(?:the )?(?:current |local )?time"),
    R("time.what", rf"{ASK}what time is it(?: currently| today)?"),
    R("time.what_order", rf"{ASK}what time it is(?: currently| today)?"),
    R("time.what_is", rf"{WHAT} (?:the )?(?:current |local )?time(?: is it)?"),
    R("time.hinglish_a", r"(?:abhi )?(?:time|samay) kya (?:hua )?hai"),
    R("time.hinglish_b", r"(?:abhi )?(?:time|samay) kya hua"),
    R("time.hinglish_c", r"(?:abhi )?kitne baje (?:hain|hai|hue hain)"),
), _time)

DATE = Skill("date", "what is today's date / day", (
    R("date.what", rf"{ASK}{WHAT} (?:the )?(?:todays )?date(?: today)?"),
    R("date.day", rf"what day is it(?: today)?|{WHAT} (?:the )?day(?: today)?|{WHAT} today"),
    R("date.plain", rf"{ASK}(?:todays )?date"),
    R("date.hinglish", r"aaj (?:ki )?(?:kya )?date (?:kya )?hai|aaj (?:kaun sa|kya) din hai|aaj kya tareekh hai"),
), _date)


# ----------------------------------------------------------------------------- briefing

async def _briefing(match: Match, rt: SkillRuntime) -> SkillResult:
    if rt.briefing is None:
        raise SkillUnavailable("no briefing engine")
    try:
        await rt.briefing.ensure_fresh()
    except Exception:           # a stale briefing is still a briefing
        pass
    return SkillResult(rt.briefing.spoken(), tainted=True)


def _briefing_validator(groups: Dict[str, str], ctx: RouterContext) -> Optional[Dict[str, Any]]:
    phrase = groups.get("p", "")
    if phrase in ("good morning", "morning") and ctx.now.hour >= ctx.morning_until_hour:
        return None                                  # a bare greeting only counts before noon (same rule as the fast path)
    return {"phrase": phrase}


def briefing_skill(phrases: Iterable[str]) -> Skill:
    extra = ["subah ka briefing", "aaj ka briefing", "meri briefing batao", "briefing do", "briefing batao",
             "morning update", "daily update", "give me the briefing", "good morning briefing"]
    normed = sorted({normalize(p)[0] for p in list(phrases) + extra if normalize(p)[0]}, key=len, reverse=True)
    alt = "|".join(re.escape(p) for p in normed)
    return Skill("briefing", "the morning briefing, straight from the cache", (
        R("briefing.phrase", rf"(?P<p>{alt})", _briefing_validator),), _briefing, untrusted=True)


# ----------------------------------------------------------------------------- calendar

async def _calendar_today(match: Match, rt: SkillRuntime) -> SkillResult:
    now = rt.clock()
    events = _events(await rt.call("today_events", {}))
    upcoming = []
    for e in events:
        start, end = _parse_dt(e.get("start")), _parse_dt(e.get("end"))
        if e.get("all_day"):
            upcoming.append((datetime.combine(now.date(), datetime.min.time()), e, True))
        elif start and (end is None or end > now):
            upcoming.append((start, e, False))
    upcoming.sort(key=lambda t: t[0])
    if not upcoming:
        return SkillResult(f"Your calendar is clear for the rest of today, {rt.address_as}.", tainted=False)
    parts = []
    for start, e, all_day in upcoming[:4]:
        parts.append(f"{_title(e.get('title'))}, all day" if all_day else f"{_title(e.get('title'))} at {_clock(start)}")
    more = f", plus {len(upcoming) - 4} more" if len(upcoming) > 4 else ""
    lead = f"You have {len(upcoming)} thing{'s' if len(upcoming) != 1 else ''} left today, {rt.address_as}: "
    return SkillResult(lead + "; ".join(parts) + more + ".", tainted=True)


async def _next_meeting(match: Match, rt: SkillRuntime) -> SkillResult:
    now = rt.clock()
    events = _events(await rt.call("upcoming", {"days": 2}))
    future = []
    for e in events:
        start = _parse_dt(e.get("start"))
        if start and not e.get("all_day") and start > now:
            future.append((start, e))
    if not future:
        return SkillResult(f"Nothing else on your calendar for the next two days, {rt.address_as}.")
    start, e = min(future, key=lambda t: t[0])
    delta = start - now
    mins = int(delta.total_seconds() // 60)
    when = f"in {mins} minutes" if mins < 90 else f"in about {round(mins / 60)} hours" if mins < 24 * 60 else "tomorrow"
    return SkillResult(f"Your next one is {_title(e.get('title'))} at {_clock(start)}, {when}.", tainted=True)


CALENDAR_TODAY = Skill("calendar_today", "what is left on today's calendar", (
    R("cal.whats_on", rf"{ASK}{WHAT} (?:on )?my (?:calendar|schedule|agenda)(?: for)? today"),
    R("cal.whats_on_the", rf"{ASK}{WHAT} (?:on )?the (?:calendar|schedule|agenda) today"),
    R("cal.my_today", rf"{ASK}{WHAT} my (?:calendar|schedule|agenda) (?:like )?today"),
    R("cal.what_do_i_have", r"what do i have (?:on )?(?:today|on my calendar today|scheduled today|on today)"),
    R("cal.any", r"(?:do i have|have i got|got) any (?:meetings?|events?|appointments?|calls?) today|"
                 r"any (?:meetings?|events?|appointments?) today|am i free today"),
    R("cal.todays", rf"{ASK}(?:my |the )?(?:todays|today) (?:schedule|agenda|calendar|meetings|events)"),
    R("cal.check", r"check my calendar(?: for)? today|check my schedule(?: for)? today"),
    R("cal.hinglish_a", r"(?:mera )?aaj ka (?:schedule|agenda|calendar|program) (?:batao|bataiye|dikhao|bata do|dikha do|kya hai)"),
    R("cal.hinglish_b", r"aaj (?:kya|kaun sa) (?:schedule|meeting|meetings|program) hai"),
    R("cal.hinglish_c", r"aaj (?:meri )?(?:meetings?|schedule|appointments?) (?:kya|kaun si|kaun kaun si) (?:hain|hai)"),
    R("cal.hinglish_d", r"aaj koi (?:meeting|event|appointment) hai"),
), _calendar_today, untrusted=True)

NEXT_MEETING = Skill("next_meeting", "when is the next meeting", (
    R("next.when", rf"(?:whens|when is) my next (?:meeting|event|appointment|call)"),
    R("next.what", rf"{WHAT} my next (?:meeting|event|appointment|call)"),
    R("next.plain", rf"{ASK}(?:my )?next (?:meeting|event|appointment|call)"),
    R("next.the", r"when is the next (?:meeting|event)"),
    R("next.on_cal", rf"{WHAT} next on my (?:calendar|schedule)"),
    R("next.hinglish", r"(?:meri |mera )?(?:agli|agla|next) (?:meeting|event|call) kab hai"),
), _next_meeting, untrusted=True)


# ----------------------------------------------------------------------------- mail / counts

SRC = (r"(?P<src>e ?mails?|mails?|gmail|inbox|slack(?: messages| dms| mentions)?|texts?|text messages?|imessages?|"
       r"messages?|unread things|notifications)")


def _source(groups: Dict[str, str], ctx: RouterContext) -> Dict[str, Any]:
    raw = groups.get("src", "")
    if re.match(r"e ?mails?|mails?|gmail|inbox", raw):
        return {"source": "mail"}
    if raw.startswith("slack"):
        return {"source": "slack"}
    if re.match(r"texts?|text messages?|imessages?", raw):
        return {"source": "texts"}
    return {"source": "all"}               # "messages", "notifications", nothing named: every source


async def _unread_counts(match: Match, rt: SkillRuntime) -> SkillResult:
    if rt.briefing is None:
        raise SkillUnavailable("no briefing engine")
    try:
        await rt.briefing.ensure_fresh()
    except Exception:
        pass
    b = rt.briefing.briefing()
    want = match.args.get("source", "all")
    bits: List[str] = []
    if want in ("all", "mail"):
        bits.append(f"{b.total_unread} unread email{'s' if b.total_unread != 1 else ''}")
    if want in ("all", "slack"):
        if b.slack_dms is not None:
            n = (b.slack_dms or 0) + b.slack_mentions
            bits.append(f"{n} on Slack")
        elif want == "slack":
            return SkillResult(f"Slack isn't set up, {rt.address_as}.")
    if want in ("all", "texts"):
        if b.messages_unread is not None:
            bits.append(f"{b.messages_unread} unread text{'s' if b.messages_unread != 1 else ''}")
        elif want == "texts":
            return SkillResult(f"Messages isn't set up, {rt.address_as}.")
    flagged = b.mail_flagged + b.slack_flagged + b.messages_flagged
    tail = f" {flagged} {_say(flagged, 'was', 'were')} filtered as unsafe." if flagged else ""
    return SkillResult(f"You have {', and '.join(bits) if len(bits) < 3 else ', '.join(bits)}, {rt.address_as}.{tail}")


UNREAD = Skill("unread_counts", "unread counts per source", (
    R("unread.how_many_unread", rf"how many unread {SRC}(?: do i have| are there| have i got| i have| are waiting| are in my inbox)?", _source),
    R("unread.how_many", rf"how many {SRC} (?:do i have|have i got|are there|i have)(?: unread)?", _source),
    R("unread.how_many_generic", r"how many unread(?: things| items)?(?: do i have| are there)?",
      lambda g, c: {"source": "all"}),
    R("unread.count_a", rf"unread {SRC}(?: count)?", _source),
    R("unread.count_b", rf"{SRC} unread count", _source),
    R("unread.count_c", r"unread count", lambda g, c: {"source": "all"}),
    R("unread.do_i_have", rf"do i have any unread {SRC}", _source),
    R("unread.hinglish_a", rf"kitne unread {SRC} (?:hain|hai)", _source),
    R("unread.hinglish_b", rf"kitne {SRC} (?:aaye hain|aaye hai|aaye)", _source),
    R("unread.hinglish_c", rf"{SRC} mein kitne unread (?:hain|hai)", _source),
    R("unread.hinglish_d", r"kitne unread (?:hain|hai)", lambda g, c: {"source": "all"}),
), _unread_counts)


async def _important_mail(match: Match, rt: SkillRuntime) -> SkillResult:
    if rt.briefing is None:
        raise SkillUnavailable("no briefing engine")
    try:
        await rt.briefing.ensure_fresh()
    except Exception:
        pass
    b = rt.briefing.briefing()
    flagged = (f" {b.mail_flagged} {_say(b.mail_flagged, 'email was', 'emails were')} filtered as unsafe."
               if b.mail_flagged else "")
    if not b.priority:
        return SkillResult(f"Nothing important in your inbox, {rt.address_as}. {b.total_unread} unread in total.{flagged}")
    names = []
    for s in b.priority[:3]:
        if s.suspicious:
            names.append("a flagged message")
        else:
            names.append(f"{clean_text(display_name(s.item.sender, 28), 28)} about {clean_text(s.item.title, 50)}")
    k = len(b.priority)
    return SkillResult(f"{k} important {_say(k, 'email', 'emails')}, {rt.address_as}: " + "; ".join(names)
                       + (f"; plus {k - 3} more" if k > 3 else "") + f".{flagged}", tainted=True)


IMPORTANT = Skill("important_mail", "any important mail / what's new", (
    R("imp.whats_new", rf"{WHAT} new(?: today)?|anything new|any news in my inbox"),
    R("imp.any_important", r"(?:do i have |did i get |have i got )?any(?:thing)? important(?: mail| emails?| messages?)?(?: today| come in| came in)?"),
    R("imp.is_there", r"is there (?:any|anything) (?:important|urgent|new)(?: mail| emails?| in my inbox)?"),
    R("imp.any_new_mail", r"(?:do i have |did i get |have i got )?any new (?:mail|emails?)(?: today)?"),
    R("imp.important_mail", r"(?:any )?important (?:mail|emails?)(?: today)?"),
    R("imp.hinglish_a", r"koi (?:important|zaroori|urgent) (?:mail|email|message) aaya(?: hai)?"),
    R("imp.hinglish_b", r"koi (?:naya|nayi) (?:mail|email|message) aaya(?: hai)?"),
    R("imp.hinglish_c", r"kuch (?:important|zaroori|naya) aaya(?: hai)?"),
), _important_mail, untrusted=True)


# ----------------------------------------------------------------------------- apps

_BUILTIN_APPS = {
    "safari": "Safari", "chrome": "Google Chrome", "google chrome": "Google Chrome", "firefox": "Firefox",
    "arc": "Arc", "brave": "Brave Browser", "notes": "Notes", "terminal": "Terminal", "iterm": "iTerm",
    "iterm2": "iTerm", "ghostty": "Ghostty", "finder": "Finder", "mail": "Mail", "messages": "Messages",
    "calendar": "Calendar", "music": "Music", "spotify": "Spotify", "slack": "Slack", "discord": "Discord",
    "zoom": "zoom.us", "vs code": "Visual Studio Code", "vscode": "Visual Studio Code",
    "visual studio code": "Visual Studio Code", "code": "Visual Studio Code", "xcode": "Xcode", "notion": "Notion",
    "whatsapp": "WhatsApp", "telegram": "Telegram", "preview": "Preview", "photos": "Photos",
    "reminders": "Reminders", "maps": "Maps", "facetime": "FaceTime", "system settings": "System Settings",
    "settings": "System Settings", "activity monitor": "Activity Monitor", "obsidian": "Obsidian", "figma": "Figma",
    "postman": "Postman", "docker": "Docker", "pages": "Pages", "numbers": "Numbers", "keynote": "Keynote",
    "calculator": "Calculator", "textedit": "TextEdit", "books": "Books", "podcasts": "Podcasts",
}
#: Apps the router will never CLOSE: Vesper's own host, and the ones where quitting is surprising.
_NEVER_CLOSE = {"terminal", "iterm", "ghostty", "finder", "vesper", "python", "code", "visual studio code", "xcode"}
#: The same list as canonical app names, so an alias ("vscode", "iterm2") cannot slip past the spoken-word check.
_NEVER_CLOSE_CANON = {"terminal", "iterm", "ghostty", "finder", "visual studio code", "xcode"}
_NOT_AN_APP = {"it", "that", "this", "everything", "all", "something", "anything", "app", "apps", "all apps", "other",
               "others", "the app", "my computer", "the door", "doors", "window", "windows", "all windows", "tabs",
               "everything else", "music player", "last one", "previous"}


def installed_apps(roots: Sequence[str] = ("/Applications", "/System/Applications", "~/Applications")) -> List[str]:
    out = []
    for root in roots:
        try:
            out += [p.stem for p in Path(os.path.expanduser(root)).glob("*.app")]
        except OSError:
            continue
    return out


class AppResolver:
    def __init__(self, installed: Iterable[str] = ()):
        self._names = dict(_BUILTIN_APPS)
        for name in installed:
            key = normalize(name)[0]
            if key:
                self._names.setdefault(key, name)

    def __call__(self, spoken: str) -> Optional[str]:
        key = normalize(spoken)[0]
        key = re.sub(r"\s+app$", "", key)
        if not key or key in _NOT_AN_APP:
            return None
        return self._names.get(key)


def _app_validator(closing: bool = False):
    def validate(groups: Dict[str, str], ctx: RouterContext) -> Optional[Dict[str, Any]]:
        spoken = groups.get("app", "")
        if closing and re.sub(r"\s+app$", "", spoken) in _NEVER_CLOSE:
            return None
        name = ctx.resolve_app(spoken)
        if closing and name and name.lower() in _NEVER_CLOSE_CANON:
            return None
        return {"app_name": name} if name else None
    return validate


APP = r"(?P<app>[a-z0-9][a-z0-9 .+\-]{0,30}?)"


async def _open_app(match: Match, rt: SkillRuntime) -> SkillResult:
    await rt.call("open_app", {"app_name": match.args["app_name"]})
    return SkillResult(f"Opening {match.args['app_name']}.")


async def _focus_app(match: Match, rt: SkillRuntime) -> SkillResult:
    await rt.call("focus_app", {"app_name": match.args["app_name"]})
    return SkillResult(f"Switching to {match.args['app_name']}.")


async def _close_app(match: Match, rt: SkillRuntime) -> SkillResult:
    await rt.call("close_app", {"app_name": match.args["app_name"]})
    return SkillResult(f"Closed {match.args['app_name']}.")


OPEN_APP = Skill("open_app", "open an application by name", (
    R("open.en", rf"(?:open|launch|start)(?: up)?(?: the)?(?: app)? {APP}(?: app)?", _app_validator()),
    R("open.hinglish_a", rf"{APP}(?: app)?(?: ko)? (?:kholo|khol do|khol de|kholna|chalu karo|chalao|open karo|start karo)", _app_validator()),
), _open_app, mutates=True)

FOCUS_APP = Skill("focus_app", "switch to an application", (
    R("focus.switch", rf"(?:switch|go|jump|flip|alt tab) to(?: the)? {APP}(?: app)?", _app_validator()),
    R("focus.focus", rf"(?:focus|bring up|bring forward)(?: on)?(?: the)? {APP}(?: app)?", _app_validator()),
    R("focus.hinglish", rf"{APP}(?: app)? (?:pe|par) (?:jao|chalo|switch karo|focus karo)", _app_validator()),
), _focus_app, mutates=True)

CLOSE_APP = Skill("close_app", "quit one named application", (
    R("close.en", rf"(?:close|quit|exit)(?: the)?(?: app)? {APP}(?: app)?", _app_validator(closing=True)),
    R("close.hinglish", rf"{APP}(?: app)?(?: ko)? (?:band karo|band kar do|band kardo|close karo|quit karo)", _app_validator(closing=True)),
), _close_app, tier="confirm", mutates=True)      # closing can discard unsaved work: the Guardian asks every time


# ----------------------------------------------------------------------------- volume

def _level(groups: Dict[str, str], ctx: RouterContext) -> Optional[Dict[str, Any]]:
    n = int(groups["n"])
    return {"level": n} if 0 <= n <= 100 else None


_VOL_STEP = 10


async def _volume_set(match: Match, rt: SkillRuntime) -> SkillResult:
    level = int(match.args["level"])
    await rt.call("set_volume", {"level": level})
    return SkillResult(f"Volume set to {level} percent.")


async def _volume_step(match: Match, rt: SkillRuntime) -> SkillResult:
    out = await rt.call("get_volume", {})
    found = re.search(r"\b(\d{1,3})\b", str(out))
    if not found or not 0 <= int(found.group(1)) <= 100:
        raise SkillUnavailable("could not read the current volume")      # nothing was changed: let the planner handle it
    cur = int(found.group(1))
    new = max(0, min(100, cur + (_VOL_STEP if match.args["direction"] == "up" else -_VOL_STEP)))
    await rt.call("set_volume", {"level": new})
    return SkillResult(f"Volume {'up' if new >= cur else 'down'} to {new} percent.")


VOLUME_SET = Skill("volume_set", "set the volume to an exact level", (
    R("vol.set_to", r"(?:set|change|put|make)(?: the)? volume (?:to )?(?P<n>\d{1,3})(?: percent| %)?", _level),
    R("vol.volume_n", r"volume (?:to )?(?P<n>\d{1,3})(?: percent| %)?", _level),
    R("vol.n_percent", r"(?P<n>\d{1,3})(?: percent| %) volume", _level),
    R("vol.hinglish", r"volume (?P<n>\d{1,3})(?: percent)? (?:kar do|kardo|karo|pe rakho|rakh do)", _level),
), _volume_set, mutates=True)

VOLUME_STEP = Skill("volume_step", "volume up / down by a step", (
    R("vol.up_en", r"(?:turn |bring |pump |crank )?(?:the )?volume up(?: a bit| a little| a notch| some)?|"
                   r"(?:turn up|increase|raise|boost|bump up) (?:the )?volume(?: a bit| a little)?|"
                   r"(?:make it |a bit |a little )?louder|volume up",
      lambda g, c: {"direction": "up"}),
    R("vol.down_en", r"(?:turn |bring |take )?(?:the )?volume down(?: a bit| a little| a notch| some)?|"
                     r"(?:turn down|decrease|lower|reduce|drop) (?:the )?volume(?: a bit| a little)?|"
                     r"(?:make it |a bit |a little )?(?:quieter|softer)|volume down",
      lambda g, c: {"direction": "down"}),
    R("vol.up_hi", r"(?:thoda sa |thoda )?(?:volume|awaaz|awaz) (?:badhao|badha do|badhana|zyada karo|increase karo|upar karo)",
      lambda g, c: {"direction": "up"}),
    R("vol.down_hi", r"(?:thoda sa |thoda )?(?:volume|awaaz|awaz) (?:kam karo|kam kar do|kam kardo|ghatao|ghata do|decrease karo|neeche karo)",
      lambda g, c: {"direction": "down"}),
), _volume_step, mutates=True)


# ----------------------------------------------------------------------------- assembly

def build_skills(briefing_phrases: Iterable[str] = ()) -> List[Skill]:
    return [briefing_skill(briefing_phrases), TIME, DATE, CALENDAR_TODAY, NEXT_MEETING, UNREAD, IMPORTANT,
            OPEN_APP, FOCUS_APP, CLOSE_APP, VOLUME_SET, VOLUME_STEP]


def build_router(*, briefing_cfg: Any = None, installed: Optional[Iterable[str]] = None,
                 clock: Callable[[], datetime] = datetime.now, disabled: Sequence[str] = ()) -> SkillRouter:
    """The router with every built-in skill. `installed` = names of installed apps (default: scan /Applications)."""
    phrases = list(getattr(briefing_cfg, "fast_path_phrases", []) or ["good morning", "morning", "morning briefing",
                   "daily briefing", "brief me", "give me my briefing", "give me my morning briefing"])
    until = int(getattr(briefing_cfg, "morning_until_hour", 12))
    resolver = AppResolver(installed_apps() if installed is None else installed)

    def ctx() -> RouterContext:
        return RouterContext(now=clock(), resolve_app=resolver, briefing_phrases=phrases, morning_until_hour=until)

    return SkillRouter(build_skills(phrases), ctx, disabled=disabled)
