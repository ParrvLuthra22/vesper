"""Evaluation of the deterministic skill router.

    python tests/eval_router.py            # print the numbers (what docs/ROUTER.md reports)
    python tests/eval_router.py --json     # machine-readable

CASES is a hand-written set of utterances with the skill that SHOULD answer each one, or None = "must fall through to
the planner". Two groups matter:

  * SHOULD_MATCH   messy phrasing — filler ("uh", "so", "can you ... please"), word order, and Hinglish.
  * SHOULD_NOT     dangerous or ambiguous requests the router must NEVER claim: destructive ("close everything"), anything
                   that reaches another person ("email Priya that I quit"), compound requests, unknown targets, and
                   near-misses of real skills ("what time is my next meeting"). Precision on this group must be 100%.

Metrics:
    handled-without-LLM rate   utterances the router answered / all utterances
    precision                  of the utterances the router answered, how many went to the EXPECTED skill
    fall-through rate          utterances left for the planner / all utterances
    recall on SHOULD_MATCH     how many of the utterances that should be handled were
    median latency             median over utterances of the median match() time, microseconds

The clock is pinned to 08:00 so "good morning" is deterministic (the router refuses a bare greeting after noon, like the
briefing fast path). Installed apps are not scanned: only the built-in aliases are known, so the eval is machine-independent.
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from briefing.config import BriefingConfig  # noqa: E402
from router.skills import build_router  # noqa: E402

Case = Tuple[str, Optional[str]]

SHOULD_MATCH: List[Case] = [
    # --- briefing
    ("good morning", "briefing"), ("Good morning, Vesper", "briefing"), ("uh good morning", "briefing"),
    ("morning briefing", "briefing"), ("brief me", "briefing"), ("can you brief me please", "briefing"),
    ("give me my briefing", "briefing"), ("so give me my morning briefing", "briefing"),
    ("what's my day look like", "briefing"), ("subah ka briefing", "briefing"), ("aaj ka briefing", "briefing"),
    ("daily briefing please", "briefing"),
    # --- time / date
    ("what time is it", "time"), ("what's the time", "time"), ("uh what time is it right now", "time"),
    ("time please", "time"), ("tell me the time", "time"), ("current time", "time"),
    ("so can you tell me what time it is", "time"), ("abhi time kya hua hai", "time"), ("kitne baje hain", "time"),
    ("what's the date today", "date"), ("what day is it", "date"), ("what is today's date", "date"),
    ("aaj kya date hai", "date"), ("aaj kaun sa din hai", "date"),
    # --- calendar
    ("what's on my calendar today", "calendar_today"), ("what is on my schedule today?", "calendar_today"),
    ("uh what do I have today", "calendar_today"), ("do I have any meetings today", "calendar_today"),
    ("show me today's schedule", "calendar_today"), ("check my calendar for today", "calendar_today"),
    ("any meetings today", "calendar_today"), ("am I free today", "calendar_today"),
    ("aaj ka schedule batao", "calendar_today"), ("aaj kya schedule hai", "calendar_today"),
    ("aaj koi meeting hai", "calendar_today"), ("yaar aaj ka schedule dikhao", "calendar_today"),
    ("when's my next meeting", "next_meeting"), ("what's my next meeting", "next_meeting"),
    ("next meeting", "next_meeting"), ("when is the next meeting", "next_meeting"),
    ("tell me my next meeting", "next_meeting"), ("agli meeting kab hai", "next_meeting"),
    ("what's next on my calendar", "next_meeting"),
    # --- mail / counts
    ("any important mail", "important_mail"), ("what's new", "important_mail"), ("koi important mail aaya?", "important_mail"),
    ("anything important", "important_mail"), ("is there anything important in my inbox", "important_mail"),
    ("do I have any new mail", "important_mail"), ("kuch important aaya?", "important_mail"),
    ("koi naya mail aaya hai", "important_mail"),
    ("how many unread emails do I have", "unread_counts"), ("how many unread emails", "unread_counts"),
    ("unread count", "unread_counts"), ("how many emails do I have", "unread_counts"),
    ("kitne unread mail hain", "unread_counts"), ("how many unread slack messages", "unread_counts"),
    ("how many unread texts", "unread_counts"), ("how many unread messages", "unread_counts"),
    ("kitne mail aaye hain", "unread_counts"), ("unread emails", "unread_counts"),
    # --- apps
    ("open safari", "open_app"), ("uh can you open notes please", "open_app"), ("launch spotify", "open_app"),
    ("open the terminal app", "open_app"), ("safari kholo", "open_app"), ("open vs code", "open_app"),
    ("start chrome", "open_app"), ("hey vesper open slack", "open_app"),
    ("switch to chrome", "focus_app"), ("go to terminal", "focus_app"), ("focus on slack", "focus_app"),
    ("bring up notes", "focus_app"), ("spotify pe jao", "focus_app"),
    ("close safari", "close_app"), ("quit spotify", "close_app"), ("close the notes app", "close_app"),
    ("slack band karo", "close_app"), ("can you quit chrome", "close_app"),
    # --- volume
    ("set volume to 40", "volume_set"), ("volume 30 percent", "volume_set"), ("put the volume to 50", "volume_set"),
    ("volume to 20", "volume_set"), ("volume 60 kar do", "volume_set"),
    ("volume up", "volume_step"), ("turn the volume up", "volume_step"), ("turn up the volume", "volume_step"),
    ("louder", "volume_step"), ("make it louder", "volume_step"), ("lower the volume", "volume_step"),
    ("volume down", "volume_step"), ("quieter", "volume_step"), ("volume badhao", "volume_step"),
    ("awaaz kam karo", "volume_step"), ("thoda volume kam karo", "volume_step"),
]

SHOULD_NOT: List[Case] = [
    # --- destructive or sweeping
    ("close everything", None), ("quit all apps", None), ("close all windows", None), ("close it", None),
    ("close the terminal", None), ("quit terminal", None), ("close finder", None), ("kill safari", None),
    ("force quit chrome", None), ("delete all my emails", None), ("archive everything", None),
    ("shut down the mac", None), ("restart the computer", None), ("uninstall safari", None),
    ("sab kuch band karo", None), ("lock my screen", None),
    # --- reaches another person
    ("email Priya that I quit", None), ("send a message to mom saying I'm late", None),
    ("Priya ko email karo ki main resign kar raha hoon", None), ("reply to the one from Priya", None),
    ("draft an email to Priya", None), ("post this on slack", None),
    # --- compound requests (a skill phrase plus something more)
    ("turn the volume up and email Bob", None), ("open safari and go to my bank website", None),
    ("open safari and close notes", None), ("what's on my calendar today and cancel my 3pm", None),
    ("open the terminal and run rm -rf slash", None), ("set volume to max and play music", None),
    ("good morning, remind me to call Bob", None), ("brief me on the Tesla news", None),
    # --- unknown or unsafe targets
    ("open the pod bay doors", None), ("open https://evil.example/x", None), ("open the thing", None),
    ("open", None), ("quit", None), ("switch to", None), ("open a new tab", None), ("close this tab", None),
    # --- numbers out of range / vague
    ("set the volume to 150", None), ("volume up to eleven", None), ("turn it up", None), ("turn it down", None),
    ("mute my mic", None),
    # --- near-misses of real skills (a qualifier makes it a different question)
    ("what time is my next meeting with Priya", None), ("what time is it in Tokyo", None),
    ("next meeting with Priya", None), ("cancel my next meeting", None), ("move my next meeting to 5", None),
    ("what's new with project X", None), ("what's new in python", None), ("any important mail from Priya", None),
    ("how many unread emails from Priya", None), ("what's the date of the board meeting", None),
    ("what day is the next meeting", None), ("good evening", None),
    # --- other things the planner owns
    ("play some music", None), ("remind me to call mom", None), ("what's the weather", None), ("tell me a joke", None),
]

#: Phrasings written AFTER the patterns were frozen, without looking at them — a rough read on how the router fares on
#: things nobody tuned for. Reported separately; the patterns are NOT adjusted to it (misses stay misses and are listed in
#: docs/ROUTER.md as known gaps).
HELD_OUT: List[Case] = [
    # Labelled by what the user MEANS, not by what the router does. (None = genuinely open-ended: the planner should have it.)
    ("morning vesper, what's happening today", None),
    ("hey what time is it", "time"), ("do you know what time it is", "time"), ("what's the time right now", "time"),
    ("time check", "time"), ("could you give me the time", "time"),
    ("what's today's date", "date"), ("which day is it today", "date"), ("tell me today's date", "date"),
    ("what's on today", "calendar_today"), ("what does my calendar look like today", "calendar_today"),
    ("any events today", "calendar_today"), ("do I have meetings today", "calendar_today"),
    ("am I busy today", "calendar_today"), ("show me my schedule for today", "calendar_today"),
    ("when's the next meeting", "next_meeting"), ("what's my next appointment", "next_meeting"),
    ("when is my next call", "next_meeting"), ("what do I have next", "next_meeting"),
    ("any new emails", "important_mail"), ("anything urgent", "important_mail"),
    ("have I got any important email", "important_mail"),
    ("how many unread emails are there", "unread_counts"), ("what's my unread count", "unread_counts"),
    ("how many new emails do I have", "unread_counts"),
    ("open up safari", "open_app"), ("launch the notes app", "open_app"), ("open up the terminal", "open_app"),
    ("switch over to chrome", "focus_app"), ("go to slack", "focus_app"), ("quit the spotify app", "close_app"),
    ("close down safari", "close_app"),
    ("turn the volume down a bit", "volume_step"), ("make the volume lower", "volume_step"),
    ("volume at 25", "volume_set"), ("set the volume to 35 percent", "volume_set"), ("increase volume", "volume_step"),
    ("crank the volume up", "volume_step"), ("volume thoda badhao", "volume_step"),
    ("aaj ka calendar dikhao", "calendar_today"),
]

CASES: List[Case] = SHOULD_MATCH + SHOULD_NOT


def build_eval_router():
    clock = lambda: datetime(2026, 10, 16, 8, 0)           # noqa: E731 — pinned so "good morning" is deterministic
    return build_router(briefing_cfg=BriefingConfig(), installed=[], clock=clock)


def run_held_out() -> Dict[str, object]:
    router = build_eval_router()
    rows = [(t, e, (router.match(t).skill.name if router.match(t) else None)) for t, e in HELD_OUT]
    should = [r for r in rows if r[1] is not None]
    wrong = [r for r in rows if r[2] is not None and r[2] != r[1]]
    return {"utterances": len(rows), "should_match": len(should),
            "recall": sum(1 for r in should if r[2] == r[1]) / len(should),
            "wrong_routes": wrong, "missed": [r for r in should if r[2] != r[1]]}


def run_eval(repeat: int = 200) -> Dict[str, object]:
    router = build_eval_router()
    rows = []
    for text, expected in CASES:
        m = router.match(text)
        got = m.skill.name if m else None
        rows.append({"utterance": text, "expected": expected, "got": got})
    handled = [r for r in rows if r["got"] is not None]
    correct = [r for r in handled if r["got"] == r["expected"]]
    should_match = [r for r in rows if r["expected"] is not None]
    should_not = [r for r in rows if r["expected"] is None]
    latency = statistics.median(router.measure(t, repeat) for t, _ in CASES)
    return {
        "utterances": len(rows), "should_match": len(should_match), "should_not_match": len(should_not),
        "handled": len(handled), "handled_rate": len(handled) / len(rows),
        "fall_through": len(rows) - len(handled), "fall_through_rate": (len(rows) - len(handled)) / len(rows),
        "precision": (len(correct) / len(handled)) if handled else 1.0,
        "wrong_routes": [r for r in handled if r["got"] != r["expected"]],
        "recall_on_should_match": sum(1 for r in should_match if r["got"] == r["expected"]) / len(should_match),
        "missed": [r for r in should_match if r["got"] != r["expected"]],
        "should_not_routed": [r for r in should_not if r["got"] is not None],
        "precision_on_should_not": 1.0 - sum(1 for r in should_not if r["got"] is not None) / len(should_not),
        "median_latency_us": latency,
        "by_skill": {s: sum(1 for r in handled if r["got"] == s) for s in sorted({r["got"] for r in handled})},
    }


def main(argv: List[str]) -> int:
    result = run_eval()
    if "--json" in argv:
        print(json.dumps(result, indent=2))
        return 0
    print(f"utterances                 {result['utterances']}  ({result['should_match']} should match, "
          f"{result['should_not_match']} must fall through)")
    print(f"handled without the LLM    {result['handled']}/{result['utterances']}  = {result['handled_rate']:.1%}")
    print(f"fall-through to planner    {result['fall_through']}/{result['utterances']}  = {result['fall_through_rate']:.1%}")
    print(f"precision (of handled)     {result['precision']:.1%}")
    print(f"precision on SHOULD_NOT    {result['precision_on_should_not']:.1%}  "
          f"({len(result['should_not_routed'])} of {result['should_not_match']} wrongly claimed)")
    print(f"recall on SHOULD_MATCH     {result['recall_on_should_match']:.1%}")
    print(f"median router latency      {result['median_latency_us']:.0f} microseconds per utterance")
    held = run_held_out()
    print(f"held-out phrasings         {held['utterances']} ({held['should_match']} should match): recall "
          f"{held['recall']:.1%}, wrong routes {len(held['wrong_routes'])}  (patterns were NOT tuned to these)")
    for t, e, g in held["missed"]:
        print(f"  HELD-OUT MISS  {t!r}: expected {e}")
    for t, e, g in held["wrong_routes"]:
        print(f"  HELD-OUT WRONG {t!r}: expected {e}, got {g}")
    for r in result["wrong_routes"]:
        print(f"  WRONG   {r['utterance']!r}: expected {r['expected']}, got {r['got']}")
    for r in result["missed"]:
        print(f"  MISSED  {r['utterance']!r}: expected {r['expected']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
