# What leaves the machine

Short version: **audio never leaves the machine by default.** Wake word, VAD, speech-to-text and text-to-speech
all run locally. The text of what you say, and the results of your tools, **do** go to the LLM provider (Groq).

Statements here are from reading the code and from config; I did **not** packet-capture a run, so "never" means
"no code path does this unless the setting below is on".

## Default (`vesper up`, shipped config)

| Data | Goes to | When | Controlled by |
|---|---|---|---|
| Your request text, the persona + context block (time, retrieved memories, observations), tool schemas, **and every tool result** (email subjects/bodies, calendar entries, web text) | **Groq** (`api.groq.com`) | every planner call (several per turn) | `llm.primary` — the one unavoidable cloud dependency |
| Gmail: message reads, drafts, archive/mark-read | **Google APIs** | when you use mail tools | `mcp.servers.gmail.enabled`, OAuth token in `data/google_token.json` |
| Weather location string | **Open-Meteo** | `current_weather` tool | no key; tool-level |
| What the briefing engine collects: sender, subject and a ≤200-char snippet of **unread mail**, and calendar titles/times (never notes) | nowhere — a local SQLite file, `data/briefing.db` (git-ignored) | every ~10 min | `config/briefing.yaml` (`refresh.enabled: false` stops the timer) |
| Of that, what the LLM sees: ≤600 tokens — at most 5 priority mail items (sender name, subject, 90-char preview), 3 meetings, counts | **Groq**, only when you ask for a briefing (the "good morning" fast path sends nothing — it is spoken from the cache) | on demand | – |
| Memory text | nowhere — Chroma + SQLite on disk, embeddings computed locally (MiniLM) | – | – |
| **Microphone audio** | **nowhere** — openWakeWord, webrtcvad, faster-whisper and Kokoro are local models | – | – |
| Traces (`data/traces/traces.jsonl`) | **nowhere** — local file only | – | `tracing.enabled` |
| Hugging Face Hub (model metadata) | **blocked** — the launcher sets `HF_HUB_OFFLINE=1` for its processes | – | `launcher.offline_models` (set false for a first-run model download) |

Only if the matching keys are set (none are by default): Tavily (web search/research queries), OpenRouter
(`WebSearchAgent` answer synthesis), Slack/Discord/GitHub/Notion/Spotify MCP servers (they talk to those
services), the Discord remote (messages via Discord).

## Opt-in: LangSmith tracing (`tracing.langsmith_enabled: true` **and** `LANGSMITH_API_KEY`)

Off by default. A key in `.env` alone does nothing — the tracer logs "key is set but
tracing.langsmith_enabled=false — traces stay on this machine". When on, each turn sends to LangSmith's cloud:

* the user's text and the assistant's final reply,
* the **full message list of every planner iteration** (system prompt incl. persona, retrieved memories,
  observations, prior turns),
* every tool call's name, **arguments and result** — so email and calendar text, file/diff text, command output,
* latency and the Guardian's verdict.

The local JSONL trace is written in every mode.

## Opt-in: cloud speech recognition (`voice.recognition.allow_cloud_stt: true`)

Only the **legacy** VoiceAgent has this, and the legacy agent is not started by `vesper up`, `main.py` (now a
redirect), or the Brain by default. If you re-enable it (`Brain(enable_voice_agent=True)`) *and* set this flag
*and* whisper.cpp is absent, it uploads your microphone audio to **Google's web speech API**
(`recognize_google`). Without the flag it refuses and logs why. Source scan test: `recognize_google` appears only
in `utils/stt.py`, behind that gate.

## Getting closer to fully local

Point `llm.primary` at a local model (see `docs/RESOURCES.md` for what that costs on 8 GB and the routing-quality
caveat in `config/settings.yaml`), disable the Gmail MCP server, and leave the opt-ins above off. Calendar and
Reminders (EventKit), memory, voice, HUD and tracing are already local.
