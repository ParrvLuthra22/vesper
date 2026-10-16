# The quiet proactive watcher

Vesper speaks up only when it matters, under a strict budget. **Off by default** (`proactive.watcher.enabled: false` in
`config/settings.yaml`) — turn it on when you want it, and `vesper dnd on` silences it instantly.

It is separate from the older `ProactiveEngine` rules (focus-based call-outs that ride along on your *next* turn, see
[ARCHITECTURE.md](ARCHITECTURE.md) §5): the watcher interrupts on its own, so it has a budget, quiet hours and a DND switch.

## What can interrupt you

| Event | When | Says |
|---|---|---|
| `meeting_30` | a meeting starts in 25–30 minutes | "Meeting in 29 minutes" |
| `meeting_5` | a meeting starts in 0–5 minutes | "Meeting in 4 minutes" |
| `priority_item` | a new unread mail / Slack / Messages item scores ≥ `threshold` (default 60, higher than the briefing's 30), arrived within 30 minutes, is not a bot, not flagged | "New message from Priya Sharma" |
| `stale_collector` | a source that has run before has not synced for over 2 hours | "Email hasn't synced for a while" |

A late watcher does not send a late reminder: the windows are hard (a watcher started 20 minutes before a meeting does not
fire the 30-minute reminder). All-day events never remind.

## The gates (in order — and a held-back interruption is dropped, never queued)

1. **Dedupe** — the same item / meeting-stage / source never fires twice, across restarts (`data/proactive_state.json`:
   ids, kinds, counts and timestamps only).
2. **DND** — `vesper dnd on` (optionally `--for 90` minutes), `off`, `status`. A flag file; checked every tick.
3. **Quiet hours** — default 23:00–08:00 `Asia/Kolkata`, wrap-around handled, evaluated in the configured timezone, not the Mac's.
4. **Budget** — `max_per_day` (default 6) per *local* day, a hard ceiling. When it is tight, meetings go first, then
   priority items by score, then stale-collector notices.
5. **Cooldown** — at most one interruption per kind-group per window (meetings 2 min, priority items 15 min, stale notices 6 h).

"Dropped, not queued" is deliberate: quiet means quiet. Nothing arrives late when DND ends or quiet hours finish; the
morning briefing is where the backlog goes. `vesper dnd status` shows today's delivered / held-back counts by reason
(`dnd`, `quiet_hours`, `budget`, `cooldown`, `unsafe_text`, `unsafe_sender`) — counts only.

## What an interruption may contain

`build_payload` is the only way text gets into an interruption, and it has no field for message text:

* a source label ("Email", "Slack", "Messages", "Calendar"),
* a sender **display name** — sanitised (no addresses, no markup), shortened, and **withheld entirely** if it (or the
  message) reads like instructions to an AI; a phone number becomes "an unknown number",
* the minutes until a meeting. Meeting **titles are not included**.

No subject, no snippet, no body, no title. A test feeds canary subjects, snippets, numbers, addresses and meeting titles
through every channel and asserts each emitted string matches a fixed whitelist of shapes.

## Delivery

* **HUD**: a `ProactiveNotificationEvent` on the bus, forwarded to the HUD as an ordinary observation call-out.
* **macOS notification**: a banner via `osascript` with the title and body passed as **argv** (the script is constant;
  nothing is interpolated).
* **Spoken**: only if voice mode is on (`voice.input.enabled`) *and* it is not quiet hours; then the sentence goes through
  the normal voice-output path (which also shows it in the HUD, so the HUD event is not sent twice).

Each channel can be switched off under `delivery:`. The watcher reads the briefing cache only — no network, nothing sent,
posted or written to any external account.

## Running it

Set `proactive.watcher.enabled: true` (see `config/settings.yaml` for every knob) and run `vesper up`: the gateway's Brain
starts the watcher next to the briefing engine and stops it with it. A bad watcher config logs a warning and does not stop
the assistant.

## Limits worth knowing

* Meeting reminders come from the **briefing cache**, which refreshes about every 10 minutes: a meeting cancelled in the
  last few minutes can still produce a reminder.
* The reminder windows assume the watcher ticks (default every 30 s). If the Mac sleeps through a window, the reminder is
  simply missed — it is not replayed on wake.
* `voice mode` is read from config, not detected from a live microphone.
* Not verified live in this repo: a real macOS banner and a real HUD call-out (tests use a fake notifier and a fake bus;
  no real notification is ever fired by the test suite).

## Tests

`tests/test_proactive_watcher.py` (55 tests, fake clock + fake notifier): windows, dedupe (and across restarts), DND
(including expiry and the CLI), quiet-hours boundaries and timezones, the budget and the local-day reset, cooldowns,
stale detection and recovery, hostile text/sender handling, the no-content-leak whitelist, delivery routing, the argv-only
banner, and that the Brain starts the watcher only when enabled. Each gate was mutation-tested: switching off DND, quiet
hours, the budget, the cooldown, dedupe, the sender/text safety checks, stale re-arming, the local-day roll or midnight
wrap-around each makes a test fail.
