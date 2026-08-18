# Recording the reel

The signature clip: **say the wake phrase → the star flares → the panel wakes →
Vesper speaks and types the greeting → give a spoken command → trace lines
stream → the result is spoken.**

Everything below exists because of one constraint: this is an **8GB M3**, and
the difference between a composed take and a visibly stalling one is whether
the LLM call stays on Groq. Groq answers in well under a second; the local
Ollama rescue needs ~38s to cold-load when memory is tight. On camera that is
the whole reel.

---

## Pre-flight (in order)

### 1. Free the memory

```bash
.venv/bin/python scripts/doctor.py
```

Look for **memory headroom** — it must not say `WARN`. Below 3GB free, loading
Kokoro (~520MB) next to anything else pushes the machine into swap, and the
lag is systemic, not Vesper's.

Close **Chrome first** — it is usually the single largest consumer. Then Slack,
Docker, any simulator. Re-run `doctor.py` until memory headroom is `PASS`.

Every other line should be `PASS` too:

| check | why it matters on camera |
|---|---|
| memory headroom | swap = a laggy take |
| ollama server + `llama3.2:3b` | only the rescue path; must exist but stay unused |
| PortAudio, `sounddevice` | no mic, no take |
| `openwakeword`, `faster_whisper` | the wake and STT stages |
| `kokoro_onnx`, `soundfile` | Vesper's voice |
| MCP venv | inbox/calendar tools in the trace |

### 2. Confirm Groq has TPM headroom

**This is the one that ruins takes.** The free tier is 8,000 tokens/minute.
A planning call is ~850 tokens after the PF3 trim, so a multi-step turn is
~3 calls. If you have been testing all morning, the rolling window may already
be spent — and the reel falls to a laggy Ollama moment mid-shot.

```bash
.venv/bin/python scripts/smoke_rate_limit.py
```

Wants `turns completed: 4/4`, `calls on local: 0`. If you see any `local`, or
pacing waits over ~2s, **wait 60 seconds** for the window to drain and re-run.
Do not start recording on a half-spent minute.

While recording, the star is your instrument: it **cools to a dim gold** the
moment a turn falls to the local model (`panel.local`). If you see it cool,
stop and re-take — that clip has a stall in it.

### 3. Warm the models

Cold loads are silent killers: the wake model, Whisper, and Kokoro each pay a
one-time load. Warm all three so the take starts on a hot path:

```bash
.venv/bin/python - <<'EOF'
from voice.input.config import VoiceInputConfig
from voice.input.stages import WakeDetector, Transcriber
from voice.output.config import VoiceOutputConfig
from voice.output.tts import KokoroTTS
import numpy as np
vc = VoiceInputConfig.from_app_config({"voice":{"input":{"enabled":True}}})
WakeDetector(vc).load();  print("wake  warm")
t = Transcriber(vc); t.load(); t.transcribe(np.zeros(16000, dtype=np.int16)); print("stt   warm")
oc = VoiceOutputConfig.from_app_config({"voice":{"output":{"enabled":True}}})
k = KokoroTTS(oc); k.synthesize("Good evening, Sir."); print("tts   warm")
EOF
```

Expect roughly: wake 0.4s, STT 3.5s, TTS 1.9s. After this they are resident and
the take is smooth.

### 4. Quiet room, one voice

openWakeWord scores ~0.997 on a clean utterance against a 0.5 threshold — it is
not fragile, but a TV or a second speaker in the room will cost takes. Close the
window. Sit ~40–60cm from the mic.

---

## Running the take

Three processes. Use three terminals so you can watch the logs, or
`./scripts/run_voice.sh` for the first two with a shared token.

```bash
python -m gateway.server                     # the brain
cd hud && npm run tauri dev                  # the panel
python -m voice.input                        # wake word + STT
python -m voice.output                       # Kokoro
```

Wait for the star to **breathe** — that is the gateway link, and it is the
signal that everything is attached. A dim ember means the HUD is not connected;
fix that before rolling.

**The sequence:**

1. Let it sit idle a beat. The panel is near-empty on purpose — star, clock,
   the last line faded low. Restraint is the aesthetic, and the contrast with
   the flare is the shot.
2. Say the wake phrase (`"Hey Jarvis"` until the custom model is trained — see
   `voice/TRAINING.md`).
3. The star **flares** gold (~650ms bloom, settles), the panel warms, and
   Vesper **speaks and types** the time-appropriate greeting.
4. Give a command with visible work in it — something that calls two tools, so
   the mono trace lines stream and then collapse to `▸ 2 steps`:
   > *"What time is it and what's my battery level?"*
5. The reply is spoken in `bm_lewis` and typed in serif.
6. Let it fall idle again on camera. The fade-down is the closing beat.

**For a call-out in the shot** (the only gold body text), have a pending
observation ready — switch apps a few times before rolling so the focus sensor
has something to raise.

---

## If it goes wrong mid-take

| symptom | cause | fix |
|---|---|---|
| star cools to dim gold | fell to the Ollama rescue | stop; wait 60s for the TPM window; re-take |
| star is a dim ember | HUD not connected to the gateway | check the gateway is up and the token matches |
| long pause before speech | a model was not warmed | re-run the warm step |
| wake does not fire | noise, distance, or the wrong phrase | quieter room; check `voice.input.wake_model` |
| whole Mac lags | swap — something big is open | close Chrome; re-run `doctor.py` |
| Vesper speaks late | speech deferred behind the local model | that is the 8GB guard working; see the star |

---

## Reduced motion

If the recording machine has **Reduce Motion** on (System Settings →
Accessibility → Display), the flare and the panel warm are suppressed by design
and the greeting merely types. Correct behaviour, wrong reel — turn it off
before shooting.
