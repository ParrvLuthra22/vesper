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
6. **Every skill goes through the Guardian** at `max(skill tier, tool tier)`. `close_app` is *confirm* in the tool
   registry already; the skill repeats that as its own floor, so lowering the tool's tier by mistake would not make the
   router quietly stop asking (a test registers `close_app` as *safe* and checks the router still asks). A denial is
   reported, not retried.
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
anchor, the ambiguity rule, the `close_app` tier floor (against a tool registered as *safe*), the never-close list (including its alias check), the Guardian
call, the volume range check or the greeting-hour check each makes tests fail.

## Updates (2026-10-17)

* "good morning / afternoon / evening / night", "brief me", "what's new" trigger the **briefing at any hour**; the spoken greeting
  follows the real time of day in `briefing.speech.timezone` (Asia/Kolkata), not the phrase. "anything new" is the inbox skill.
* `web_search` skill: "search for / google / look up X" hands off to the planner with **only** the `web_search` tool
  (DuckDuckGo, Tavily if `TAVILY_API_KEY` is set; 5 results, ~600 tokens, site names only, hostile results dropped, tainted).
* App names tolerate recogniser near-misses ("Sofari", "Message"); closing an app still needs the exact name.
* Calendar: "what's on my calendar" without "today", and "calender", are claimed.

## Tool-schema budget (category selection)

Every planner call carries the tool schemas (45 tools ≈ 4,500 tokens live; Groq's free tier allows 8k tokens a minute). Requests that
name a category — calendar, mail, messages, drafts, files, web — get only that category's read tools plus the core three
(`tools/selection.py`, `select_by_category`): typically 5-10 tools, 400-900 tokens. `confirm`, `dangerous` and outbound tools are
never added by a category (only by naming them, as before). Unclear requests fall back to the group logic and then to the full set.
`restrict_tools` (router hand-off, e.g. web_search) offers exactly one tool.

## Honest actions (2026-10-17 live test)

* **Grounded results.** `open_app`/`focus_app` check with System Events that the app really is frontmost and return `frontmost: yes|no|unknown`
  plus `can_claim:` / `cannot_claim:` lines (`utils/grounding.py`). The claim verifier flags reply sentences about specifics no result
  supports — a conversation window for a person, "ready for you to type", text placed in a message box, "in front" — and replaces them
  with a sentence built from the results ("I opened Messages, but I can't open a specific conversation.").
* **Messaging people, read-only to send.** `contacts_lookup` (fuzzy; asks which one when several match; the model never sees a number)
  → `messages_compose` (confirm tier; `open imessage://…`, optional pre-filled text). There is no send path — a source-scan test proves
  it; the user presses send. Both tools are `private`: logs, audit, live events and the confirmation card carry counts, never names.
  `vesper doctor` reports the Contacts permission and names the terminal app that needs it.
* **STT name vocabulary** (`voice.stt.*`): first names of people you write to (+ Contacts if already allowed), capped at 60, passed to
  faster-whisper as `initial_prompt`; never logged. `vesper stt-test "<phrase>" --names A,B` compares with/without (Kokoro, no mic;
  only the names you type).
* **LLM failures** say the class (rate limit, timeout, network, auth, server); Retry-After is honoured once up to 20 s
  (`llm.rate_limit.retry_after_max_seconds`); a reply that was half-streamed when the call failed is discarded (`STREAM_RESET`).

## Notifications, sounds, Contacts permission (2026-10-18)

* `show_notification` never claims display: `osascript` exits 0 with no banner when the terminal app lacks notification permission or a Focus mode
  is on. Run `vesper notify-test` (posts one, asks "Did you see it? (y/n)", stores the answer in `data/notifications.json`); `vesper doctor`
  reports notifications as confirmed / unconfirmed / failed and names the app and the path (System Settings > Notifications > <app>).
  Delivery: the HUD always; spoken as well when voice mode is on and it is not quiet hours; the macOS banner best-effort on top.
* `play_sound` (safe tier): one of the macOS system sounds via `afplay` (allow-list, argv, no shell). "make a beep" is a router skill. The result
  says it ran, never that it was heard.
* `vesper contacts-auth`: the one command that makes macOS ask for Contacts access (for the terminal app it runs from); `vesper doctor` shows the
  status and says when the running stack was started from a different app.

## Echo control and HUD-native banners (2026-10-18)

* **The mic hears the speaker.** `voice.echo.*`: (1) half-duplex gate — no utterance capture while Vesper speaks and for `tail_ms` (700) after, also in
  follow-up windows; the wake word keeps running for barge-in but needs `wake_threshold_while_speaking` (0.7); (2) gateway echo filter — a voice
  transcript that is mostly one of the last 5 replies never reaches the planner (log: a count and `reason=echo_match`); (3) loop breaker — 3 failed turns
  in 60 s, or 3 consecutive turns within 3 s of speech ending, pause voice input for 30 s with one HUD notice (nothing spoken). Real acoustic echo
  cancellation is assessed, not built: it needs the played audio as a reference, and Vesper's speech is played by a different process from the one listening.
* **Banners come from the HUD** (app identity "Vesper HUD") when the stack's HUD is connected; `osascript` (terminal identity) is only the fallback.
  `vesper doctor` and `vesper notify-test` say which path is active and for which app the confirmation counts.
