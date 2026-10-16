# Deterministic skill router

The common voice commands are answered **without calling the LLM**: faster, free, and they work offline. Everything the
router does not recognise — or recognises ambiguously — falls through to the planner exactly as before.

```
you say ─► Brain.handle_user_text
              ├─ turn carries forwarded / third-party text? ──► planner (the router is skipped)
              ├─ router.enabled and exactly one skill claims it? ─► skill handler ─► Guardian ─► existing tools ─► spoken reply
              └─ otherwise ─────────────────────────────────────► planner (unchanged)
```

## What it handles

| Skill | Examples (English / Hinglish) | Reads / does | Tier floor |
|---|---|---|---|
| `briefing` | "good morning" (before noon), "brief me", "what's my day look like" / "subah ka briefing" | briefing cache script | safe |
| `time`, `date` | "what time is it", "tell me the time" / "abhi time kya hua hai", "kitne baje hain" | the clock | safe |
| `calendar_today` | "what's on my calendar today", "do I have any meetings today" / "aaj ka schedule batao", "aaj koi meeting hai" | `today_events` | safe |
| `next_meeting` | "when's my next meeting", "next meeting" / "agli meeting kab hai" | `upcoming` | safe |
| `important_mail` | "any important mail", "what's new" / "koi important mail aaya?", "kuch important aaya?" | briefing cache | safe |
| `unread_counts` | "how many unread emails", "how many unread slack messages", "unread count" / "kitne unread mail hain" | briefing cache | safe |
| `open_app`, `focus_app` | "open Safari", "switch to Chrome" / "Safari kholo", "Spotify pe jao" | `open_app`, `focus_app` | safe |
| `close_app` | "close Safari", "quit Spotify" / "Slack band karo" | `close_app` | **confirm** (asks every time) |
| `volume_set`, `volume_step` | "set volume to 40", "volume up", "louder" / "volume badhao", "awaaz kam karo" | `get_volume`, `set_volume` | safe |

`vesper route "<utterance>"` explains a decision (nothing is executed); add `--list` for every skill and rule.

## How it decides — and why it is conservative

1. **Normalise**: lowercase, no punctuation or apostrophes, and filler dropped from both ends ("uh", "so", "can you",
   "please", "yaar", "sir"...). Nothing that carries meaning is dropped.
2. **Whole-phrase patterns**: every skill's patterns are tried with `re.fullmatch`. Any word a pattern does not expect
   ("... and email Bob", "... in Tokyo", "... with Priya") makes it fall through.
3. **Exactly one skill may claim an utterance.** Zero or more than one → the planner.
4. **Arguments are validated before a skill may claim anything**: an app must be one the Mac has (built-in aliases plus a
   scan of `/Applications`), a volume must be 0–100. `open the pod bay doors`, `volume 150` → planner.
5. **It never closes its own host or sweeps**: Terminal, iTerm, Ghostty, Finder, VS Code and Xcode are never closed by the
   router (checked on the *resolved* app name, so aliases like "vscode" cannot slip past), and "everything", "all",
   "it", "that" are not app names.
6. **Every skill goes through the Guardian** at `max(skill tier, tool tier)`. `close_app` is raised to *confirm* even
   though the tool itself is *safe*. A denial is reported, not retried.
7. **Failure policy**: a read-only skill that fails (calendar permission, unreadable data) falls through to the planner;
   a skill that changes something and fails says so honestly instead.
8. Turns that carry third-party text (a forwarded Telegram message) never take the shortcut. Skill replies that quote
   third-party text (event titles, senders) mark the turn *tainted*, and hostile titles are replaced by "a meeting with a
   flagged title".

Config: `router.enabled` and `router.disabled_skills` in `config/settings.yaml`.

## Measured results

`python tests/eval_router.py` — 156 hand-written utterances (97 that should be handled, 59 that must fall through),
clock pinned to 08:00, only built-in app aliases (so it is machine-independent). Measured on the dev Mac (M3, 8 GB):

| Metric | Result |
|---|---|
| Handled without the LLM | **97 / 156 = 62.2 %** of the whole set (the rest is meant to reach the planner) |
| Fall-through to the planner | 59 / 156 = 37.8 % |
| Precision (of what it handled) | **100 %** — no utterance went to the wrong skill |
| Precision on the 59 dangerous / ambiguous utterances | **100 %** — 0 of 59 wrongly claimed (a hard test) |
| Recall on the 97 utterances it was built for | 100 % — *optimistic: this set was written alongside the patterns* |
| **Recall on 39 unseen phrasings** (held-out) | **53.8 %** (21 / 39), **0 wrong routes** — patterns were not tuned to this set |
| Median router latency | **11 µs** per utterance (median over the 156; measured, not estimated) |

The held-out figure is the honest one for "what would a person actually say": about half of unfamiliar phrasings are
handled and the other half simply take the normal planner path. It misses by falling through, never by misrouting.

**Known gaps (held-out misses, deliberately not tuned for):** "do you know what time it is", "time check",
"which day is it today", "what's on today", "what does my calendar look like today", "do I have meetings today",
"am I busy today", "show me my schedule for today", "when's the next meeting", "what do I have next", "anything urgent",
"what's my unread count", "how many new emails do I have", "switch over to chrome", "close down safari",
"make the volume lower", "volume at 25", "volume thoda badhao". Each is a one-line pattern if you want it; the point of
the held-out set is to say how the router does *before* that kind of tuning.

What this does **not** measure: the latency of the tools a skill calls (calendar, volume — those are the same as the
planner's), cost savings in tokens (each handled utterance avoids one planner turn; the exact saving depends on the
Groq prompt size and was not measured here), or accuracy on a real microphone transcript (speech-to-text errors are not
modelled; the eval is text).

## Tests

`tests/test_router.py` (matching, ambiguity, eval thresholds, runner + Guardian gating, Brain integration) and
`tests/eval_router.py` (the metrics). The security-relevant checks were mutation-tested: removing the full-phrase
anchor, the ambiguity rule, the `close_app` tier floor, the never-close list (including its alias check), the Guardian
call, the volume range check or the greeting-hour check each makes tests fail.
