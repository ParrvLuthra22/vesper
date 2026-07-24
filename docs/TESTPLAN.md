# Vesper v3.0.0 — Manual Acceptance Test Plan

This is a scripted, human-run walkthrough for validating a release build
end to end, against real services (Groq, Ollama, Gmail, Calendar/Reminders,
Notion, GitHub, Spotify, Slack, Discord — as configured) — not a substitute
for the automated `pytest` suite (`pytest` — **250 tests** as of v3.0.0),
which covers unit-level correctness and runs on every change. Run this plan
before tagging a release, or after any change to `orchestrator/`, `llm/`,
`tools/`, `guardian/`, `remote/`, `sensors/`, or `proactive/`.

Tests **1–20** are the v1/v2 core (conversation, tools, confirmation,
sensors, briefing, memory, failure drills, CLI). Tests **21–33** cover the
v3 capability layer: developer tools, music, research/script/automation,
Slack/Discord + the remote interface, the morning routine, and weather.

## Prerequisites

- `GROQ_API_KEY` set (`.env`).
- Ollama running locally (`ollama serve`) with the configured fallback
  model pulled (`config/settings.yaml` → `llm.fallback.model`, default
  `qwen3.5:latest`).
- Gmail MCP server authenticated at least once (`data/google_token.json`
  exists) — see `mcp_servers/gmail/gmail_client.py`.
- Calendar/Reminders (macOS) permission already granted to the terminal
  running Vesper (System Settings → Privacy & Security → Calendars /
  Reminders) — `mcp_servers/apple_pim/`.
- At least one real unread email, one real calendar event today, and one
  real reminder due today, so triage/day-planning have live data to react
  to. A few unread promotional emails are enough to exercise triage.
- Run via `vesper` (or `python -m vesper`) unless a test says otherwise —
  the CLI's live trace lines and `/trace` are how most of this plan is
  observed.

Record each result as **PASS** / **FAIL** with a one-line note. A FAIL on
any failure-drill test (14–19) blocks the release; a FAIL elsewhere should
be triaged before tagging but doesn't necessarily block it.

---

### 1. Open-vocabulary conversation

**Steps:** Start a session. Ask something with no tool involved: *"what's
the capital of Australia"*, then a follow-up that needs conversational
memory: *"and its population?"*

**Expected:** Vesper answers both correctly in voice (addresses you as
"Sir", no exclamation marks/emoji), the second answer correctly resolves
"its" to Canberra from the first turn's context — no tool call fires
(no `▸` trace lines).

---

### 2. Multi-step tool-calling plan

**Steps:** *"what's in my inbox, and draft a reply to the newest one
saying I'll respond by Friday"*

**Expected:** Live trace shows `▸ list_unread(...)` → `✓ list_unread`,
then `▸ draft_reply(...)` (confirm-tier — see test 3), and after
approval `✓ draft_reply`. Final reply names the message it drafted
against. `/trace` shows two `tool_execution` nodes under one `turn`.

---

### 3. Confirmation flow — approve

**Steps:** Ask for any confirm-tier action (e.g. *"archive that email"*
following test 2, or *"add a reminder to call the bank tomorrow at 9am"*).
When the `⚠` warning + `Confirm? [y/N]` prompt appears, type `y`.

**Expected:** The action actually executes (e.g. a real reminder appears
in Reminders.app) and Vesper confirms it in voice. `data/audit.jsonl`
gets a new line with `"verdict": "allow"`.

---

### 4. Confirmation flow — deny

**Steps:** Repeat test 3's request. When prompted, type `n` (or anything
but `y`).

**Expected:** The action does **not** execute (no new reminder/no
archive). Vesper acknowledges the denial without arguing or retrying
unprompted. `data/audit.jsonl` gets a line with `"verdict": "deny"`,
`"who_approved": null`.

---

### 5. Confirmation flow — timeout

**Steps:** Trigger a confirm-tier action, then simply don't respond to
the `[y/N]` prompt. Wait past `guardian.confirmation_timeout_seconds`
(default 120s).

**Expected:** After the timeout, the turn resolves as denied on its own
(no hang, no crash) with a graceful in-voice message. `data/audit.jsonl`
shows `"verdict": "deny"`, `"who_approved": null`, and the reason
mentions "expired".

---

### 6. Sensor rule: context-switch — fire + cooldown suppression

**Steps:** With `sensors.focus.enabled: true`, rapidly switch focus
across ≥3 of the configured `work_apps` (e.g. VS Code → Terminal →
Safari) within the configured `window_min`. Then immediately repeat the
same switching pattern again within `cooldown_min`.

**Expected:** First pass: an `ObservationEvent(kind="context_switch")`
fires — visible as a distinct `○ ...` line in the CLI (or voiced on the
next turn per the passive call-out doctrine). Second pass (within
cooldown): **no** repeat observation, confirming the cooldown gate.

---

### 7. Sensor rule: meeting reminder — fire + cooldown suppression

**Steps:** With a real calendar event starting in ~5 minutes and
`sensors.calendar.enabled: true`, wait through the 15-minute and
5-minute checkpoints.

**Expected:** An `UpcomingMeetingEvent` at each checkpoint; the
proactive engine's near-term rule call-out (`meeting_soon`) fires once
at the 5-minute mark, not the 15-minute one (see
`_MEETING_CALLOUT_MAX_MINUTES`), and does not repeat within
`cooldown_min` even if the sensor re-polls before the meeting starts.

---

### 8. Sensor rule: inbox surge — fire + cooldown suppression

**Steps:** With `sensors.inbox.enabled: true` and a short
`poll_interval_seconds` override for testing, send yourself (or wait
for) ≥`surge_threshold` new unread emails between two polls. Then let a
second surge happen within `surge_cooldown_min`.

**Expected:** First surge: `ObservationEvent(kind="inbox_surge")` fires
with a correct count in `detail`. Second surge within cooldown: suppressed.
(`tests/test_inbox_sensor.py` covers this with a fake clock — this step
confirms it against the real Gmail MCP tool.)

---

### 9. Morning briefing — scheduled

**Steps:** Temporarily set `proactive.schedule.morning_briefing.time` to
a couple of minutes from now, restart, and wait for it to fire.

**Expected:** A `BriefingRequestedEvent` fires the configured job;
Vesper delivers a structured briefing unprompted (inbox triage, today's
calendar, carried-over items if any, plus a day-plan section if
`include_day_plan: true`) via voice output, without you saying anything.

---

### 10. Morning briefing — on demand

**Steps:** *"brief me"*

**Expected:** Same structure as test 9, delivered immediately as a
direct reply to your request. Say it again the next day (or clear
`data/memory.db`'s briefing key) and confirm "carried-over" items are
called out correctly when the same top-3 unread ids repeat.

---

### 11. Day plan

**Steps:** *"plan my day"*

**Expected:** Live trace shows `▸ plan_my_day()` → `✓ plan_my_day`. The
reply is a realistic time-blocked schedule that: reflects your actual
calendar events (no double-booking), mentions reminders due, mentions
Notion tasks if `mcp.servers.notion.enabled: true`, factors in unread
email volume, and flags any real scheduling conflict if one exists.

---

### 12. Memory write → recall across restart

**Steps:** Tell Vesper something durable and specific: *"the gym slot
is non-negotiable, always protect it."* End the session (`/quit`).
Start a brand new session. Ask *"plan my day"* again.

**Expected:** Session-end reflection extracts and stores the preference
(`data/chroma_memory` gains an entry with `metadata.kind: preference`,
`metadata.source: reflection`). The new session's day plan visibly
protects that block, and `/trace` on that turn shows
`memory_injected: [...]` containing the stored text.

---

### 13. Forget flow

**Steps:** After test 12, say *"forget the gym slot preference."*
Approve the confirmation. Ask *"plan my day"* once more.

**Expected:** Vesper confirms what it forgot (matching text from the
stored memory). The new day plan no longer treats the gym slot as
protected. A direct semantic-retrieve for "gym" (or checking
`data/chroma_memory` directly) shows the entry is gone.

---

### 14. Failure drill: Groq down → falls back to Ollama

**Steps:** Temporarily point `llm.primary.provider`/model at something
invalid (or block Groq's endpoint / revoke the API key temporarily), and
ask a question that also requires a tool call mid-turn (so the fallback
message history includes a prior tool-call message —
this is the exact path `_normalize_messages_for_ollama` exists for).

**Expected:** No crash, no raw exception. The reply arrives via Ollama
(check `/trace` → `plan_iteration.provider == "ollama"`), with correct
tool-calling behavior even though a prior assistant tool-call message is
in the replayed history. Restore the Groq config afterward.

---

### 15. Failure drill: both LLMs down

**Steps:** With Groq unreachable (as in test 14) **and** Ollama stopped
(`killall ollama` or just point `llm.fallback` at a bad endpoint), send
any message.

**Expected:** Vesper replies in-voice with a graceful message (default:
*"Sir, I'm having trouble thinking right now."*) — never a stack trace,
never a hang past the request timeout. Restore both providers afterward.

---

### 16. Failure drill: MCP server crashes mid-session

**Steps:** With Gmail (or apple_pim) connected, find and kill its
subprocess directly: `pkill -f "mcp_servers/gmail/server.py"`. Then
immediately ask for something that would use one of its tools (e.g.
*"what's in my inbox"*).

**Expected:** The in-flight/next call fails cleanly (no hang past the
request timeout) and Vesper says the capability isn't available rather
than surfacing a raw error. Within `RECONNECT_MAX_ATTEMPTS` backoff
attempts (a few seconds to ~a minute), the tools reappear automatically
— confirm by asking the same question again after waiting and seeing it
succeed, or by checking logs for `"reconnected successfully"`.

---

### 17. Failure drill: Gmail token expired

**Steps:** Simulate an unrecoverable refresh failure: edit
`data/google_token.json`'s `refresh_token` field to garbage (back up the
real file first), then ask *"what's in my inbox."*

**Expected:** A clear, actionable message reaches you (in logs at
minimum, and ideally relayed in voice): re-authenticate by deleting the
token file and restarting — not a raw `RefreshError` traceback. Restore
the real token file afterward.

---

### 18. Failure drill: tool timeout

**Steps:** Temporarily set a very short `tool_timeout_seconds` (e.g. via
a debug override) or trigger a call to a slow tool
(`summarize_thread` on a very long thread is a reasonable real-world
candidate). Alternatively reuse `tests/test_planner.py`'s approach
conceptually: make a bus-routed tool's responder never answer.

**Expected:** After the timeout, Vesper apologizes gracefully instead of
hanging indefinitely; the trace shows `tool_execution` with
`success: False` and a "timed out" message.

---

### 19. Failure drill: malformed tool arguments

**Steps:** Ask something deliberately underspecified that's likely to
produce an incomplete first tool call — e.g. *"add a reminder"* with no
title given at all, in a single utterance, forcing the model to attempt
a call before it has enough information.

**Expected:** If the model calls the tool anyway and it fails validation,
the error is fed back as a normal tool result (never a raw exception),
the model retries (often successfully, by asking a clarifying question
first instead) or — bounded by `max_iterations` — Vesper apologizes
gracefully rather than looping forever or crashing.

---

### 20. CLI surface: /trace, /tools, /status, /quit

**Steps:** After a couple of turns, run `/trace`, `/tools`, and
`/status` in sequence, then `/quit`.

**Expected:** `/trace` renders the last turn's full run tree (turn →
plan_iteration → tool_execution) with correct nesting and metadata.
`/tools` lists every registered tool with its tier (safe/confirm/dangerous)
across all connected servers (builtin + Gmail + apple_pim + Notion if
enabled). `/status` shows every agent's health and both LLM providers'
availability. `/quit` shuts down cleanly (no traceback), and triggers
the session-end reflection pass (see test 12).

---

## v3 capability layer (21–33)

Enable the relevant servers/tools in `config/settings.yaml` before running each
(all v3 integrations default **off**). A test whose service isn't configured is
**N/A**, not FAIL.

### 21. Developer tools — git status/diff and gated commit (PC0)

**Steps:** In a repo working tree, *"what's changed in this repo?"* then
*"commit the staged changes."*

**Expected:** `▸ git_status` / `▸ git_diff` run safe (no prompt). `git_commit`
is confirm-tier: the `⚠` summary shows the **exact repo, branch, and the commit
message** (auto-generated from the staged diff if you didn't dictate one). It
commits only on `y`, and only the current repo — never one you didn't name.

### 22. Developer tools — run_tests on the task queue (PC0)

**Steps:** *"run the tests."*

**Expected:** `run_tests` is `slow=True` — Vesper replies immediately
("…in the background") and, on completion, announces *"Sir, the run tests you
asked for is ready."* as a pending observation, with a pass/fail summary.

### 23. Music — named track plays immediately (PC1)

**Steps:** With `mcp.servers.spotify.enabled: true`, *"play Sirf Kaam Hai by
[artist]."*

**Expected:** `▸ play(...)` runs **safe** (no confirmation — playback is never
gated). Vesper confirms in one line, no menu, no deliberation.

### 24. Music — context-aware selection from memory (PC1)

**Steps:** Once, tell it a preference: *"when I'm coding put on lo-fi, no
vocals."* Later, *"focus time"* (a state, not a track).

**Expected:** Vesper **chooses** without asking, honouring the remembered
preference (lo-fi), and states the one-line choice. `/trace` shows the music
preference surfaced in `memory_injected`.

### 25. Deep research — queues, announces, writes a file (PC2)

**Steps:** *"research the best MCP servers for productivity."* (needs
`TAVILY_API_KEY` + a working LLM.)

**Expected:** `research` is slow → runs on the queue; the inline reply is a
3-line summary + a path. On completion: *"Sir, the research you asked for is
ready."*, and a full sourced report exists under `data/research/`.

### 26. Script writer — obeys the format spec (PC2)

**Steps:** *"write me a reel script about shipping Vesper."*

**Expected:** The output follows `config/formats/reel.md` exactly — four beats
(Cold Frame → Declaration → Proof → Tomorrow hook), the signature line
`Sirf kaam hai`, the sign-off `Day X. Done.`, no exclamation marks/emoji. The
full script is saved to `data/scripts/`.

### 27. Automation composer — verbatim confirmation, runs on approval (PC2)

**Steps:** *"clear my downloads folder of files older than 30 days."*

**Expected:** Vesper composes a shell command and asks to run it **DANGEROUS**;
the `⚠` summary shows the command **verbatim** (never paraphrased). It executes
only on `y`, every time (no remembered approval).

### 28. Automation denylist — refused outright, even if approved (PC2)

**Steps:** Ask for something denylisted: *"run `sudo rm -rf /`"* (or `diskutil`,
a `curl … | sh`, or a write outside your home dir).

**Expected:** Refused **outright** with the matched pattern named — it is not
even offered for confirmation-approval, and approving anything adjacent never
runs it. `data/audit.jsonl` shows no `allow` for it.

### 29. Slack/Discord — reads safe, sends show exact channel + text (PC3)

**Steps:** With `mcp.servers.slack.enabled: true`: *"any Slack mentions I've
missed?"* then *"reply in #eng saying I'll review the PR after lunch."*

**Expected:** `get_mentions` / `search` run safe. `post_message` / `reply_thread`
are confirm-tier: the `⚠` summary shows the **exact channel and the full
message text** before anything posts. (Discord tools behave identically.)

### 30. Remote interface — owner-only, dangerous disabled (PC3)

**Steps:** With `remote.enabled: true` + a bot token, DM Vesper from your own
Discord account in the designated channel: *"what's on my calendar?"* Then
message from a **different** account. Then, as owner, ask for a dangerous action
(e.g. a shell command).

**Expected:** Your message is answered (reply posted back). The non-owner
message is **ignored** (no reply, no turn). The dangerous request is **refused**
— dangerous-tier tools are disabled for remote sessions entirely
(`remote.allow_dangerous` defaults false).

### 31. Remote interface — confirm-tier needs explicit approval (PC3)

**Steps:** Over the remote channel, ask for a confirm-tier action (e.g. draft a
Slack reply). When Vesper posts the confirmation with a request id, first reply
*"yes"*; then reply with the **request id** (or react ✅ to the prompt).

**Expected:** The bare *"yes"* is **refused** ("that won't approve it…"). The
action runs only after the explicit request-id reply or the ✅ reaction.

### 32. Morning routine — unprompted, composed, once per day (PC4)

**Steps:** Set `proactive.morning_routine.enabled: true`, `time` a couple of
minutes out (and a `weather_location`); disable `morning_briefing`. Sit at the
desk and wait — or just open an app after the configured hour.

**Expected:** Unprompted, Vesper delivers **one** composed briefing (weather
line → what matters → top-3 to action → day plan), spoken + on the HUD, naming
any failed source in a single clause. It then **offers** to set up your
workspace (start the playlist + open work apps) as a **single** confirmation;
approving runs the whole set. It does not fire a second time the same day. Reply
*"not now"* on a fresh day → it defers ~60 min; a second *"not now"* → cancelled
for the day.

### 33. Weather tool (PC4)

**Steps:** *"what's the weather in London?"*

**Expected:** `▸ current_weather(...)` runs safe and returns a one-line summary
(conditions + current temp + today's range) via Open-Meteo — no API key needed.

---

## Result log

| # | Test | Result | Notes |
|---|------|--------|-------|
| 1 | Open-vocabulary conversation | | |
| 2 | Multi-step tool-calling plan | | |
| 3 | Confirmation — approve | | |
| 4 | Confirmation — deny | | |
| 5 | Confirmation — timeout | | |
| 6 | Sensor: context-switch | | |
| 7 | Sensor: meeting reminder | | |
| 8 | Sensor: inbox surge | | |
| 9 | Briefing — scheduled | | |
| 10 | Briefing — on demand | | |
| 11 | Day plan | | |
| 12 | Memory write → recall across restart | | |
| 13 | Forget flow | | |
| 14 | Groq down → Ollama fallback | | |
| 15 | Both LLMs down | | |
| 16 | MCP server crash mid-session | | |
| 17 | Gmail token expired | | |
| 18 | Tool timeout | | |
| 19 | Malformed tool arguments | | |
| 20 | CLI surface (/trace /tools /status /quit) | | |
| 21 | Dev tools — git status/diff + gated commit | | |
| 22 | Dev tools — run_tests on the task queue | | |
| 23 | Music — named track plays immediately | | |
| 24 | Music — context-aware selection from memory | | |
| 25 | Deep research — queues, announces, file written | | |
| 26 | Script writer — obeys the reel format spec | | |
| 27 | Automation composer — verbatim confirm + run | | |
| 28 | Automation denylist — refused outright | | |
| 29 | Slack/Discord — reads safe, sends confirm | | |
| 30 | Remote — owner-only, dangerous disabled | | |
| 31 | Remote — confirm needs explicit approval | | |
| 32 | Morning routine — unprompted, once/day, defer | | |
| 33 | Weather tool | | |
