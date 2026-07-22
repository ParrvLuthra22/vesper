# Vesper HUD (PV1 — shell)

A frameless, always-on-top desktop panel for Vesper, built with **Tauri 2** and
a **vanilla-TS** frontend (tiny bundle, no framework). This phase is the SHELL
only: the window, the design system, the breathing evening star, and a live
gateway link that proves data flows. Feature rendering is PV2.

## What's here

- **Frameless / transparent / always-on-top** slim panel (~360×520), pinned to
  the top-right of the primary display, draggable by its header. Does not steal
  focus (`focus: false`).
- **Design system** as CSS variables (`src/styles.css`): `#0A0A0B` panel,
  `#2A2823` hairline, brand gold `#C9A96A`, Fraunces (italic) + JetBrains Mono
  **bundled locally** (`src/fonts/*.woff2`) — no CDN, works offline.
- **The evening star** — a 7px gold dot beside a letter-spaced `VESPER` and a
  live clock. It breathes (opacity .45→1, gentle scale, ~3.2s) when connected
  and dims to a near-dead ember when the gateway is gone. Connection state *is*
  the star's liveness. Respects `prefers-reduced-motion`.
- **Live gateway link** — opens `ws://host:port/ws?token=…`, reconnects with
  exponential backoff, and streams incoming **event types** into a mono status
  line as proof of flow.
- **Tray** — Show / Hide Panel, Quit. Also exposed as Tauri commands
  (`toggle_panel`, `quit_app`).

## Configure the gateway target

Precedence: **defaults → config file → environment** (env wins).

Environment (simplest for dev):

```bash
export VESPER_GATEWAY_TOKEN="your-token"   # must match the gateway's token
export VESPER_GATEWAY_PORT=8760            # optional (default 8760)
```

…or copy `vesper-hud.config.example.json` → `vesper-hud.config.json` (gitignored)
and fill in the token. Rust also reads `$APPCONFIG/com.vesper.hud/config.json`
and `$VESPER_HUD_CONFIG`.

## Run

```bash
# 1) start the gateway (from the repo root)
VESPER_GATEWAY_TOKEN=dev-token python -m gateway.server

# 2) start the HUD (from hud/)
cd hud
npm install
VESPER_GATEWAY_TOKEN=dev-token npm run tauri dev
```

The panel appears top-right with the star breathing. Kill the gateway → the star
dies to an ember; restart it → the star revives. Send a turn (e.g. via the CLI
or a websocket client) and watch the event types stream into the status line.
