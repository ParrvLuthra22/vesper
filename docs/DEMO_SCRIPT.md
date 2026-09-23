# Demo Script — 75 seconds

A shot-by-shot script for the signature reel: **wake word → a multi-tool task → a
Guardian confirmation → the proactive call-out → HUD traces → idle.** For
machine pre-flight (freeing memory, checking Groq's rate-limit window, warming
models), run [`docs/DEMO.md`](DEMO.md) first — this document is only the shot
list and what to say.

---

## Before you hit record

1. Run the [`docs/DEMO.md`](DEMO.md) pre-flight checklist in full (`doctor.py`,
   `smoke_rate_limit.py`, model warm-up).
2. **Restart `python -m gateway.server` fresh.** The proactive engine's
   per-rule cooldowns live only in memory (`proactive/engine.py`,
   `self._last_fired`) — if you've tested this flow already today, the
   call-out step below won't fire until a cooldown clears. A fresh process
   clears it instantly.
3. **Arm the call-out with a pre-roll, off camera.** The context-switch rule
   fires once you've focused **3 distinct configured work apps within a
   30-minute window** (`config/settings.yaml` →
   `proactive.rules.context_switch`: `threshold: 3`, `window_min: 30`; the
   configured set is VS Code, Terminal, Xcode, Slack, Safari, Chrome, Mail).
   Before rolling, spend a minute actually switching between three of
   those — e.g. Safari → Slack → Terminal — so the observation is pending
   and ready to surface on the next qualifying focus change during the take.

---

## What to hide

- **Notifications.** Turn on Do Not Disturb / a Focus mode. Slack, Mail, and
  iMessage banners will otherwise show real message content on camera the
  moment they arrive mid-take.
- **Slack.** Land on a channel with nothing sensitive visible, or a channel
  used only for testing. Don't leave the DM list open with real names in the
  sidebar.
- **Mail.** Don't have the inbox list on screen with real subject lines —
  keep Mail backgrounded; the inbox sensor doesn't need the Mail window open
  to work.
- **Calendar.** If a calendar-reading step is in the take, use a test event
  with a placeholder title, not a real personal one.
- **Browser.** Close tabs and bookmarks that expose a logged-in personal
  account; start from a clean window.
- **Terminal.** `clear` the scrollback before rolling. Backgrounded for this
  take — it's not on camera, but keep it clean in case you tab to it.
- **Desktop.** No visible desktop icons or files if the HUD window doesn't
  cover the full frame.

## Apps to have open

| App | State at 0:00 | Why |
|---|---|---|
| HUD (`cd hud && npm run tauri dev`) | Foregrounded, full frame | This is the shot |
| `gateway.server` | Running, backgrounded terminal | The brain — not on camera |
| `voice.input` / `voice.output` | Running, backgrounded | Wake word, STT, TTS |
| Slack | Backgrounded, clean channel | Target of the Guardian confirm step |
| Safari or VS Code | Backgrounded | Used only in the pre-roll to arm the call-out |

---

## Shot list

| Time | Shot | Say | On screen |
|---|---|---|---|
| 0:00–0:05 | Idle | *(silence)* | HUD at rest: dim star, clock, last line faded low. Let it sit — restraint is the point of the opening beat. |
| 0:05–0:08 | Wake | **"Hey Jarvis"** *(pretrained wake model — see Roadmap in the README for the custom-phrase status)* | Star flares gold, ~650ms bloom, settles. |
| 0:08–0:13 | Greeting | *(Vesper speaks first)* | Time-appropriate greeting spoken in `bm_lewis` and typed in serif. |
| 0:13–0:33 | Multi-tool task | **"What time is it and what's my battery level?"** | Two mono trace lines stream (`get_time`, `get_battery`), collapse to `▸ 2 steps`, reply spoken and typed. |
| 0:33–0:53 | Guardian confirmation | **"Close Slack for me."** | `close_app` is `confirm`-tier: an inline card appears with the summary, a gold **Approve** and a hairline **Deny**, and a 120s countdown bar. Click **Approve** on camera. |
| 0:53–1:08 | Proactive call-out | *(silence — let it land unprompted)* | The gold call-out from the pre-roll surfaces on its own: *"That's 3 different apps in the last 30 minutes…"* It is the only gold body text in the whole take. |
| 1:08–1:15 | Fade | *(silence)* | Back to idle; the last line fades low. This is the closing beat, not a cut. |

---

## Notes

- **The Guardian confirm is a click, not a spoken "yes."** There is no
  voice-approval path (`hud/src/render.ts` only wires the Approve/Deny
  buttons) — that's deliberate, not a gap: a confirm-tier action gets a
  second, distinct input, not a rushed spoken word in the same breath as the
  request.
- **If the star cools to dim gold at any point**, the turn fell to the local
  Ollama rescue — cut, wait 60s per `docs/DEMO.md`, and retake. A cold local
  load runs ~38s and will visibly stall the shot.
- **The call-out only fires once.** If it doesn't land in the 0:53–1:08
  window, the pre-roll switch sequence didn't register as 3 distinct apps
  inside the 30-minute window, or a prior test inside the last 90 minutes
  already consumed the cooldown — restart `gateway.server` and redo the
  pre-roll rather than waiting mid-take.
- **`close_app` actually quits Slack.** Reopen it before the next take if
  you're doing multiple passes.

TODO(parrv): record the take and link the final file/video here — this
document is the shot list, not the recording.
