# VESPER

VESPER is an operator with judgment — a personal AI chief of staff for macOS, built as a modular, event-driven assistant with voice control, local intent parsing, and automation tooling, designed for local-first usage on Apple Silicon with optional cloud integrations.

## Status: Under Reconstruction

This codebase is being rebuilt from its previous incarnation (FRIDAY). The startup face-verification gate and image generation have been removed, VisionAgent is disabled by default (screen OCR returns in v2), and identity strings have been renamed throughout. Expect breaking changes while the rebuild is in progress.

## Architecture Overview

- `orchestrator/`: lifecycle + routing + startup orchestration (LangGraph-based)
- `bus/event_bus.py`: pub/sub backbone
- `agents/*`: domain-specific capability modules
- `schemas/events.py`: typed event contracts
- `ui/hud_overlay.py`: floating HUD
- `config/settings.yaml`: runtime configuration

## Quick Start

## 1) Create venv and install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional API keys:
- `OPENROUTER_API_KEY` for Grok/OpenRouter-based extraction and summarization
- `TAVILY_API_KEY` for source-backed live web retrieval

## 2) Install required macOS dependencies

```bash
brew install portaudio tesseract brightness
```

If you use Whisper.cpp binary mode, ensure `voice.whisper.binary_path` points to a valid binary.

## 3) Ensure Kokoro model files exist

Expected files in repo root:
- `kokoro-v0_19.onnx`
- `voices.bin`

## 4) Run Vesper

```bash
./.venv/bin/python main.py
```

Vesper starts unconditionally — there is no startup identity gate blocking command handling.

## Configuration You'll Likely Edit

Primary file: `config/settings.yaml`

- `general.assistant_name`: currently `VESPER`
- `voice.wake_word`: currently `vesper`
- `voice.tts.voice_id`: currently `af_heart`
- `voice.address_user_as_sir`: true
- `vision.enabled`: `false` by default (screen OCR returns in v2)
- `intent.provider`: `ollama` (recommended local) or `pattern`
- `web_search.llm_provider`: `auto`, `openrouter`, `gemini`, or `local`
- `web_search.openrouter.model`: set to your preferred Grok/OpenRouter model

## Manual Test Plan

Use these spoken commands (or keyboard input path, depending on your run mode):

1. Voice response baseline
- "Vesper, what time is it?"

2. App control
- "Open Safari"
- "Search for Python context managers"

3. System control
- "Set volume to 40"
- "What's my battery?"

4. Finder/utility checks
- "Open downloads"

5. Search acknowledgement
- "Search for best Python async tutorial"

## Troubleshooting

### No speech audio
- Check selected audio output device.
- Confirm Kokoro files exist.
- Vesper falls back to `pyttsx3` on Kokoro failure.

### OCR not working (if you re-enable `vision.enabled: true`)
- Confirm `tesseract` is installed (`brew install tesseract`).
- Ensure `pytesseract` is installed in the same venv.

## Project Status

This is an actively evolving assistant stack under reconstruction. Some flows are experimental and improving in each iteration. Report exactly which command failed + what Vesper said/logged for the fastest fix cycle.
