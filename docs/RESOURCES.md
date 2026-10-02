# Resource profile — the whole stack on an 8 GB M3

Measured 2026-10-02 on the target machine (MacBook, M3, 8 GB, macOS 26.1) with the real stack started by
`vesper up`: gateway (Brain + planner + Gmail/apple_pim MCP servers), voice output (Kokoro), the Tauri HUD,
voice input (openWakeWord + VAD + faster-whisper). Run from a scratch copy of the repo so no real memory/trace
data was touched.

## How to read these numbers

* **Use `phys_footprint` (the `footprint` tool), not RSS.** macOS compresses and swaps under pressure, and `ps`
  RSS then collapses: the gateway showed **24 MB RSS** while its footprint was **505 MB**. The RSS time series
  in the raw data is therefore meaningless on this machine; footprints are what count.
* CPU is `%CPU` of one core (8 cores total), sampled each 0.5–1 s.
* The machine was **already under heavy pressure** when I measured: 3.2 GB wired, 2.9 GB held by the compressor,
  1.1 GB free, and **8.1 GB of swap in use** (9.2 GB swap file) before Vesper started. Cold-start times and the
  crash below are measured *under that pressure*; on a freshly booted Mac they will be better.
* "Voice turn" caveat: I could **not** drive the microphone. I tried an acoustic loop (the Mac's own speakers saying
  "Hey Jarvis, what time is it" into its mic) and the wake word did not trigger, so that path is **UNVERIFIED
  end to end**. What I measured instead: the same `POST /wake` + `POST /message` calls `voice.input`'s sink makes
  (gateway → planner → Groq → tool → reply → Kokoro speaks → HUD renders), and the **real STT stage separately**
  on a real audio file (below).

## Per-component footprint

| Component | Idle | During / after one voice turn | Peak seen | Idle CPU | Turn CPU peak |
|---|---|---|---|---|---|
| **gateway** (Brain, planner, 2 MCP servers) | **505 MB** | 695 MB (stays; MiniLM embedder loads on first turn, +190 MB) | 696 MB | 0.3 % | 81 % (embedder load) |
| **voice output** (Kokoro TTS) | **35 MB** | 494 MB resident once Kokoro has loaded | **694 MB** (first load) | 0 % | **364 %** (≈3.6 cores while synthesizing) |
| **HUD** (Tauri app + WebKit helpers) | **62 MB** | 134 MB after the first render | 140 MB | 0.1 % | 12 % |
| **voice input** (wake word + VAD) | **150 MB** | 151 MB until someone speaks | 151 MB | **10 %** (continuous: 13.8 % max) | – |
| **voice input + Whisper** (first utterance) | – | **398 MB** (+263 MB; measured in a separate process) | 400 MB | – | – |
| **Total** | **752 MB** | **1,474 MB** measured steady after a turn (**≈ 1.7 GB** once Whisper is also resident) | **1,605 MB** measured (+ Whisper ≈ 1.9 GB) | ≈ 11 % of one core | |

STT (real speech, 4.8 s utterance, `base.en` int8): **3.1 s the first time** (includes loading the model),
**0.41 s warm** (0.09× realtime); transcript was accurate. Wake-model load: 1.1 s.

## Latency of a turn (injected message, warm network)

| Step | Time |
|---|---|
| Planner call, simple question (Groq `gpt-oss-120b`) | 0.5–1.4 s each; 2–3 calls per tool turn |
| Cold first turn after boot | **4.5 s** for the first planning call (embedder load + connection setup) |
| Kokoro: first sentence after boot | seconds (model load; see the crash below) |
| STT after the user stops talking | 0.4 s warm |

## Update: Kokoro is now preloaded at boot (voice output)

`voice.output.preload: true` loads Kokoro **and runs one silent synthesis before connecting to the gateway**, so a
first-use failure surfaces at boot (the launcher reports it) instead of eating the first reply. Cost: voice output
is now **≈390–415 MB resident from startup** (measured `phys_footprint`, 6 consecutive preloads, 1.8–2.9 s each) instead
of 35 MB until the first reply — add ≈ +380 MB to the *idle* rows above (idle total ≈ 1.1 GB). Set
`preload: false` to get the old behaviour back. Result under 8.3–8.6 GB of swap: **0 crashes in 6 standalone
preloads and 2 full-stack boots (with spoken turns afterwards, 0 restarts)**. The original SIGABRT was seen once and
never reproduced on demand, so this is **not proof it is fixed** — only that the failure is now moved to boot.

## A real failure the supervisor caught

On the **first** voice-output turn, `voice_output` was killed by **SIGABRT (signal 6)** while Kokoro was loading
(footprint had just peaked at 694 MB, with 8 GB of swap in use). The launcher logged
`voice_output: died (signal SIGABRT (-6)); restarting in 1s`, restarted it, it reconnected, and the *second* turn
was spoken normally. The first reply was **lost** (never spoken). Root cause is **UNVERIFIED** (the child printed no
traceback; memory pressure during the first load is the likely trigger, not proven). Mitigation worth doing:
pre-load Kokoro at start instead of on the first reply, so a failure happens at boot, where the launcher reports it.

## Headroom on an 8 GB M3

The always-on Vesper stack costs **0.75 GB idle, ≈1.5 GB after the first turn, ≈1.7–1.9 GB at its worst**
(Whisper resident while Kokoro is also loaded). Against 8 GB that leaves roughly **6 GB for macOS and your apps**,
of which macOS itself already holds ≈3.2 GB wired + ≈2.9 GB compressed on this machine. In practice, with a
browser and a few apps open the Mac is **already swapping before Vesper starts** (8–10 GB of swap in use during
these runs); Vesper's ~1.7 GB is enough to keep it there but it works (every turn above completed).

### What a 3–4 B local model at Q4 adds

| Model | Added | Evidence |
|---|---|---|
| `llama3.2:3b` Q4_K_M, `num_ctx=2048` (the configured rescue) | **+2.3 GB** (Ollama `ps` SIZE), cold load **6.6 s**, **37 tok/s** generation | *Measured* on this machine; swap rose 8.9 → 9.9 GB while it loaded |
| a 4 B Q4 model | **≈ +2.9–3.3 GB** | *Estimate* (≈ 2.5 GB file × the 1.15 ratio measured for the 3B). **UNVERIFIED** |

With the 3B resident the Vesper-attributable total is **≈ 4 GB (3B) to ≈ 4.7 GB (4B)** — half to 60 % of RAM,
on top of macOS's ≈3.2 GB wired. **It does not fit comfortably alongside Whisper + Kokoro + the HUD**; it only
works if you keep almost everything else closed, and then it works by swapping. That is why the config keeps the
local model a *rescue* (`keep_alive: 30s`, reflection only) and why the router tells voice output to hold off
while it is loaded (`LocalModelStateEvent`).

Levers, biggest first (none implemented): unload Kokoro after N idle seconds (−440 MB, costs a reload before the
next reply); replace the MiniLM embedder + torch with Ollama's `nomic-embed-text` (−190 MB in the gateway); keep
Whisper `base.en` (it is already small); drop the HUD (−62 MB) when memory-starved. A 4 B planner is not on the
table on this machine; a 3 B is a rescue, not a daily driver.

## Reproducing

```
vesper up                      # in one terminal
footprint -p <pid>             # per process (pids: vesper status --json)
```
Raw samplers used for this report live in the session scratchpad, not in the repo.
