# Running Vesper: `vesper up`

```
vesper up [--strict]     # start everything, supervised, in the foreground (Ctrl+C stops it)
vesper status [--json]   # components, state, pid, uptime, restarts; live gateway check
vesper down              # clean ordered shutdown from another terminal
vesper logs [component] [--tail N] [--redact-transcripts]   # tail logs/launcher/<component>.log;
                                                            # the flag replaces anything said (or said to you) with <text:N>
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

## Which microphone voice input listens to, and how to check it

Voice input is **pinned**: `voice.input.device` (default `"builtin"`) is resolved explicitly at startup to the MacBook's own
microphone — matched by CoreAudio transport type when readable and by name (`MacBook Pro Microphone`, `MacBook Air
Microphone`, `Built-in Microphone` and the localized forms) — never by "whatever macOS has as default" (AirPods, a virtual
device...). You can instead give an exact device name, a sounddevice index, or `"default"` (only if you ask for it). If the
chosen mic is missing, voice input logs the input devices it found (names only) and exits **69**, the permanent-failure
code, so the launcher reports it once instead of restarting; it never falls back silently. The startup line logs the chosen
device's name, index and native rate. A mic that only offers 44.1/48 kHz is opened at that rate and converted to the 16 kHz
mono the wake word and Whisper need (an unconverted stream would be slowed 3x with no error).

* `vesper mic-test` — 10 s on the pinned device: level bar (RMS/peak), wake score every 0.5 s with the max so far, then
  `no signal` / `signal but wake word never above threshold` / `wake word fired`. Records nothing, no speech-to-text,
  sends nothing. Exit 3 = no signal, 69 = device unavailable.
* `vesper doctor` — shows which input device voice input would use (name only), **WARN** if the system default input is not
  the built-in mic (voice input is unaffected, other apps are), **FAIL** if the built-in mic is not found.
* Sensitivity: `voice.wake.threshold` (default 0.5, unchanged). Lower = hears you more easily and mis-fires more. Voice
  input logs the max wake score per 10 s window at DEBUG (scores only, no audio) so you can tune it from real use.
* `vesper logs voice_input --tail 40 --redact-transcripts` — the log with what you said replaced by `<text:N>` (the
  pipeline logs the transcript on its `injected` line).

## Pasting logs safely: `vesper logs <component> --tail N --redact-transcripts`

Component logs can contain what you said ("Payload: {'text': '…'}" on a voice event) and what was said back. With
`--redact-transcripts` every recognised transcript or message text becomes a `<text:N>` placeholder (the same text gets the
same number within one run, and the number says nothing about the content); event names, ids, sources, tool names and
timings stay, so the log is still useful. Payload fields are masked unless their key is on a short allow-list of identifiers
and flags (unknown keys are treated as text), and a payload that cannot be parsed is masked whole. It recognises the shapes
Vesper itself writes (`launcher/logsafe.py`); a new log line that interpolates text some other way would not be caught, so read
the output once before pasting. The log files themselves are never modified.

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
6. **Crash recovery**: in your own terminal run `kill -9 $(pgrep -f "voice.output")` (the process is `python -m voice.output`; `vesper status` shows the component as `voice_output`, so `pkill -f voice_output` matches nothing) → restarted within ~1–2 s, `RESTARTS` goes to 1, next reply spoken.
   Repeat with the HUD and with voice input. `kill -9` the **gateway** → it restarts; HUD/voice reconnect on their own.
6b. **Which microphone**: `vesper mic-test` first (see above), then check the `Voice input starting | … mic=<name> (index N, native R Hz)` line in `vesper logs voice_input` names the built-in mic.
7. **Microphone denied**: revoke Microphone for your terminal, `vesper up` → voice input FAILED with a clear hint,
   everything else still up, no restart loop.
8. **Mic busy**: start another app holding the mic, then `vesper up` → same loud failure.
9. `vesper down` from a second terminal → ordered stop, no leftover `gateway.server` / `voice.` / `hud` processes (`pgrep -fl`).
10. Ctrl+C in the `vesper up` terminal → same clean stop. Then `vesper up` again immediately (port free, no stale state).
11. `python main.py` → prints the redirect notice and starts the launcher; **no** microphone indicator before the stack is up.
12. Privacy: with `tracing.langsmith_enabled` unchanged (false), confirm nothing appears in your LangSmith project after a turn.
13. Memory pressure: with Chrome etc. open, run 5 turns in a row; note any crash/restart in `vesper status` (RESTARTS column).
