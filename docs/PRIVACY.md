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
| Derived from your SENT mail: the **known-correspondent set**, i.e. recipient addresses and domains with message counts, from header addresses only (no body, subject or snippet) | nowhere; inside `data/briefing.db` (list it with `vesper briefing --known`) | refreshed daily | `collect.sent_days`, `collect.sent_max_messages` |
| Your `--mark` feedback rules: a sender address or domain and an action, nothing else | nowhere; `data/briefing_rules.json` (plain JSON, git-ignored) | when you run `--mark` | `vesper briefing --rules` |
| Memory text | nowhere — Chroma + SQLite on disk, embeddings computed locally (MiniLM) | – | – |
| **Microphone audio** | **nowhere** — openWakeWord, webrtcvad, faster-whisper and Kokoro are local models | – | – |
| Traces (`data/traces/traces.jsonl`) | **nowhere** — local file only | – | `tracing.enabled` |
| Hugging Face Hub (model metadata) | **blocked** — the launcher sets `HF_HUB_OFFLINE=1` for its processes | – | `launcher.offline_models` (set false for a first-run model download) |

Only if the matching keys are set (none are by default): Tavily (web search/research queries), OpenRouter
(`WebSearchAgent` answer synthesis), Slack/Discord/GitHub/Notion/Spotify MCP servers (they talk to those
services), the Discord remote (messages via Discord), the Telegram channel (messages via Telegram — see below).

## Opt-in: LangSmith tracing (`tracing.langsmith_enabled: true` **and** `LANGSMITH_API_KEY`)

Off by default. A key in `.env` alone does nothing — the tracer logs "key is set but
tracing.langsmith_enabled=false — traces stay on this machine". When on, each turn sends to LangSmith's cloud:

* the user's text and the assistant's final reply,
* the **full message list of every planner iteration** (system prompt incl. persona, retrieved memories,
  observations, prior turns),
* every tool call's name, **arguments and result** — so email and calendar text, file/diff text, command output,
* latency and the Guardian's verdict.

The local JSONL trace is written in every mode.

## Opt-in: Telegram channel (`channels.telegram.enabled: true`)

Off by default, and nothing is contacted unless it is enabled **and** a bot token is configured. When on,
**everything you say to Vesper through Telegram, and everything it says back, passes through Telegram's servers.**
Telegram bot chats are ordinary cloud chats, **not end-to-end encrypted**: assume Telegram (and anyone who
compromises it or your account) can read them. Do not send secrets through it.

| Data | Goes to | Notes |
|---|---|---|
| The text of your messages to the bot | **Telegram** (you send it from your phone) | Then to Vesper by outbound HTTPS long polling to `api.telegram.org`; there is no webhook and no open port. |
| Every reply Vesper sends, including **confirmation cards** (the tool name and its arguments, e.g. a draft's recipient and text) | **Telegram** | Replies can contain mail/calendar content Vesper read for you. |
| A **voice note you send** | **Telegram** (your phone uploads the audio) | Vesper downloads it over HTTPS, transcribes it with the **local** faster-whisper model, processes the text, and deletes the file (also on errors; a stale-file sweep runs at start). **No cloud speech-to-text is used.** A forwarded voice note is never downloaded. |
| Photos / files / stickers you send | **Telegram** | Vesper never downloads them (it is only told that an attachment of that kind arrived). |
| Metadata: your numeric user ID, the bot, timestamps, IP of your phone/Vesper | **Telegram** | Inherent to using Telegram. |
| The text of a request you make this way | **Groq**, as for any turn | Telegram being involved does not change what the LLM provider sees. |
| The bot token | stays on this machine (Keychain or `.env`), sent only to Telegram as part of API URLs | Never logged, never committed; revoke it in BotFather if exposed. |
| Audit log entries about the channel | nowhere — `data/audit.jsonl`, local | Ids and reasons only, never message text. |

What the channel **does not** change: audio from the microphone still never leaves the machine; the briefing cache
is still local. Safety properties (allowlist, restricted session, taint, nonce-bound confirmations) and the threat
model are in [CHANNELS.md](CHANNELS.md).

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
