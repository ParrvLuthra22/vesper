# VESPER Open-Source Stack (Apple Silicon, 8GB RAM)

This guide is tuned for local-first performance on Apple Silicon with 8GB RAM.

## 1) Core local runtime

```bash
brew install ollama
ollama serve
```

## 2) Recommended local models

Use one at a time on 8GB RAM for best responsiveness.

### General reasoning / planner fallback

```bash
ollama pull qwen2.5:7b-instruct
```

### Coding help

```bash
ollama pull qwen2.5-coder:7b
```

### Fast lightweight fallback

```bash
ollama pull llama3.2:3b
```

### Optional lightweight vision captioning (non-OCR)

```bash
ollama pull moondream
```

## 3) VESPER settings

`config/settings.yaml` is already configured for local-first operation. The
LLM router uses Groq as primary with Ollama as the local fallback, and vision
(disabled by default) prefers local OCR over cloud Gemini:

- `llm.fallback.provider: "ollama"` — local model used when Groq is unavailable
- `vision.enabled: false` — vision is off by default (see the v2 roadmap)
- `vision.use_gemini: false` — no cloud vision; local OCR only
- `vision.local_ocr_enabled: true`

## 4) Voice stack (v2)

Voice is not part of the v1 primary interface (the CLI is); it exists in the
codebase and runs via `python main.py`. For a local-first voice setup:

- Wake word: `vesper`
- TTS: `kokoro` (falls back to macOS system TTS when unavailable)

## 5) Performance notes for 8GB RAM

- Keep only one 7B model loaded at a time.
- Prefer quantized GGUF models via Ollama defaults.
- Close heavy apps while running local inference.
- Use OCR for text extraction; avoid large multimodal models for continuous screen parsing.
