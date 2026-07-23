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

Download the Kokoro model files into `voice/models/` (once):

```bash
cd voice/models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

## Voice

Default is **`bm_george`** — a measured British male that fits the butler
restraint. Other calm options: `bm_lewis`, `bm_daniel`, `am_michael`. Set
`voice.output.tts_voice` and restart. Kokoro outputs 24 kHz.

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
