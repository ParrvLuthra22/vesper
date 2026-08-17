# Voice output (PV4) — Kokoro-82M TTS

Vesper's voice. Runs as its own client of the gateway: it subscribes, and speaks
the **serif voice lines** the HUD shows (wire `reply`, and proactive
`observation`). Mono trace lines are **never** spoken. Barge-in: a `wake` event
stops playback at once. Off by default (`voice.output.enabled`).

## Setup

```bash
pip install kokoro-onnx soundfile      # onnxruntime-based; no PyTorch
brew install espeak-ng                 # phonemizer data (KokoroTTS finds it automatically)
```

Download the Kokoro model files into `voice/models/` (once — ~337MB total,
gitignored):

```bash
cd voice/models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx   # 310MB
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin    # 27MB
```

Verify:

```bash
python -c "
from voice.output.config import VoiceOutputConfig
from voice.output.tts import KokoroTTS
c = VoiceOutputConfig.from_app_config({'voice':{'output':{'enabled':True}}})
t = KokoroTTS(c); t.load()
s, sr = t.synthesize('Good evening, Sir.')
print(f'{len(s)/sr:.2f}s @ {sr}Hz — {len(t._kokoro.get_voices())} voices available')"
```

## Voice

Default is **`bm_lewis`**, chosen by measurement rather than taste. Median
fundamental frequency across the candidates, same line, same speed:

| voice | median F0 | |
|---|---|---|
| `am_onyx` | 88 Hz | lowest overall, but American |
| **`bm_lewis`** | **100 Hz** | **default — lowest British, unhurried delivery** |
| `am_michael` | 122 Hz | |
| `bm_fable` | 123 Hz | |
| `bm_daniel` | 126 Hz | |
| `bm_george` | 142 Hz | the previous default |
| `am_eric` | 161 Hz | brightest, least butler |

A lower register reads as composed, which is the restraint the persona asks
for; British fits "a British butler in temperament". Set
`voice.output.tts_voice` and restart. Kokoro outputs 24 kHz and all 54 voices
are available — `am_onyx` if you want it lower still and don't mind American.

Measured performance on an M3 (8GB): model load **1.9s**, synthesis **0.55×
realtime** (faster than it speaks), resident **~520MB**. That last number is
why the 8GB guard below exists.

## 8GB guard

Kokoro resident (~520MB) plus the Ollama rescue model (~2.5GB) plus the app is
what pushes an 8GB machine into swap. While the gateway reports the local LLM
as active (wire type `local_model`), synthesis is **queued** rather than run,
and the queue drains as soon as it goes idle:

```yaml
voice:
  output:
    defer_while_local_llm: true
    defer_timeout_seconds: 20.0   # speak anyway if the signal never clears
```

This only engages on the Ollama rescue path — with Groq primary (PF3) no local
model is loaded, so nothing is ever deferred. The timeout matters because the
signal comes from another process: if the gateway dies mid-call, a watchdog
flushes the backlog instead of muting Vesper for the session.

## Run

```yaml
# config/settings.yaml
voice:
  output:
    enabled: true
    tts_voice: "bm_george"
    streaming: true      # speak sentence-by-sentence — no long wait for the first words
```

```bash
python -m gateway.server     # the Brain + gateway (emits VoiceOutputEvent)
python -m voice.output       # speaks Vesper's lines as they arrive
```

With `voice.input` also running, saying the wake word triggers the cinematic
reveal: the HUD star flares, the panel wakes, and Vesper **speaks + shows** the
time-appropriate greeting, then listens.
