# Running Vesper: `vesper up`

```
vesper up [--strict]     # start everything, supervised, in the foreground (Ctrl+C stops it)
vesper status [--json]   # components, state, pid, uptime, restarts; live gateway check
vesper down              # clean ordered shutdown from another terminal
vesper logs [component]  # tail logs/launcher/<component>.log
```
`python -m launcher up` is the same without the console script; `python main.py` and `scripts/run_voice.sh` are
now thin redirects to it. The terminal text REPL (`vesper`, no subcommand) is unchanged and uses no microphone.

## What it starts, in order — each only after the previous is *ready*

| # | Component | Process | Ready when | If it fails |
|---|---|---|---|---|
| 1 | gateway (Brain, planner, MCP) | `python -m gateway.server` | `GET /status` answers 200 with the session token | **required** — the whole stack stops |
| 2 | voice output (Kokoro TTS) | `python -m voice.output` | logs "voice output connected to gateway" | reported loudly; stack continues |
| 3 | HUD | `hud/src-tauri/target/release/hud` | gateway's connected-client count rises | reported loudly; stack continues |
| 4 | voice input (wake word + VAD + STT) | `python -m voice.input` | logs "voice input ready" (wake model loaded, mic streaming) | reported loudly; stack continues |
| 5 | Telegram channel *(optional — only when `channels.telegram.enabled: true`)* | `python -m channels.telegram` | logs "telegram channel ready" (config + token valid; polling started) | reported loudly; stack continues. Exit 69 (disabled, empty allowlist, no/bad token, token rejected) is permanent, not restarted. See [CHANNELS.md](CHANNELS.md) |

"Wake word" and "speech-to-text" are one process (they share the microphone stream). Voice input starts last so a
reply has somewhere to be spoken and shown the moment the wake word can fire. A shared random bearer token is
generated per run (or taken from `VESPER_GATEWAY_TOKEN`) and handed to every child; it is written to
`data/run/token` (mode 0600) for `vesper status`.

## Supervision

* **Crash** (non-zero exit / signal): restart after 1 s, 2 s, 4 s … capped at 30 s. More than 5 crashes in 120 s,
  or 3 failures in a row before ever becoming ready, → **FAILED** (no restart loop). Staying healthy for 60 s
  forgives earlier crashes.
* **Exit 0** → the component turned itself off (e.g. `voice.output.enabled: false`); not restarted.
* **Exit 69** from voice input *or voice output* = "unavailable this session" (mic blocked/in use, wake model or
  package missing; Kokoro files missing or failing to load) → **FAILED immediately, not restarted**, with the child's
  last log lines and a hint. Voice input also exits 69 if the **microphone delivers no audio within 15 s**
  (`voice.input.first_frame_timeout_seconds`) — a blocked permission prompt or a device held by another app makes the
  read hang forever rather than raise.
* A FAILED *required* component (gateway) shuts everything down and exits non-zero. A FAILED optional one is printed
  as a WARNING and listed in `vesper status` — never silent. `--strict` turns any optional failure/skip into an abort.
* **Shutdown** (Ctrl+C, SIGTERM or `vesper down`): reverse order — voice input, HUD, voice output, gateway — SIGTERM,
  8 s grace, then SIGKILL. Children run in their own session so Ctrl+C reaches only the supervisor.
* Refuses to start if a supervisor is already running, or if the gateway port is already in use.
* Config: the `launcher:` block in `config/settings.yaml` (components on/off, backoff, timeouts, HUD binary path).

## `vesper doctor`

A read-only environment check: microphone devices (and, with `--probe-mic`, permission), Telegram token *presence* in
the Keychain/environment (the value is never read or printed), model files, the gateway port, briefing-cache
freshness, and the channel of the last audit entry. One PASS/WARN/FAIL line each, with the fix under every WARN/FAIL.
It sends nothing and exits 1 only on FAIL. The full end-to-end checklist is [LIVE_VERIFICATION.md](LIVE_VERIFICATION.md).

## Manual test checklist (the voice round trip is not automated)

1. `vesper up` → four READY lines; `vesper status` agrees; the HUD star appears on the wake word, not before.
2. **Wake word**: say "Hey Jarvis" (the stock model; "vesper" was never trained) → HUD flares and the greeting is
   spoken. Check it does **not** fire on TV/background speech.
3. **Full turn**: "Hey Jarvis … what time is it" → spoken answer, HUD shows the reply. Try one with a tool
   ("what's on my calendar today") and one that needs approval ("close Safari") — approve it from the HUD card.
4. **Barge-in**: start a long answer ("brief me"), say "Hey Jarvis" mid-sentence → speech stops.
5. **First reply after boot**: confirm it is actually spoken (a cold Kokoro load once crashed and lost it; see RESOURCES.md).
6. **Crash recovery**: `kill -9` the `voice.output` pid from `vesper status` → restarted within ~1–2 s, next reply spoken.
   Repeat with the HUD and with voice input. `kill -9` the **gateway** → it restarts; HUD/voice reconnect on their own.
7. **Microphone denied**: revoke Microphone for your terminal, `vesper up` → voice input FAILED with a clear hint,
   everything else still up, no restart loop.
8. **Mic busy**: start another app holding the mic, then `vesper up` → same loud failure.
9. `vesper down` from a second terminal → ordered stop, no leftover `gateway.server` / `voice.` / `hud` processes (`pgrep -fl`).
10. Ctrl+C in the `vesper up` terminal → same clean stop. Then `vesper up` again immediately (port free, no stale state).
11. `python main.py` → prints the redirect notice and starts the launcher; **no** microphone indicator before the stack is up.
12. Privacy: with `tracing.langsmith_enabled` unchanged (false), confirm nothing appears in your LangSmith project after a turn.
13. Memory pressure: with Chrome etc. open, run 5 turns in a row; note any crash/restart in `vesper status` (RESTARTS column).
