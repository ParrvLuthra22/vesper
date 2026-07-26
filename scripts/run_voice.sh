#!/bin/bash
# =============================================================================
# VESPER — one-command voice launcher
# =============================================================================
# Starts the gateway (VESPER's brain + a small local web server) AND the voice
# listener together, sharing ONE auto-generated token so you never have to deal
# with tokens by hand. Ctrl+C stops both.
#
#   ./scripts/run_voice.sh
#
# Requirements handled elsewhere (one time):
#   - voice input deps installed:  pip install openwakeword faster-whisper sounddevice webrtcvad-wheels
#   - config/settings.yaml has  voice.input.enabled: true
#   - your terminal app has Microphone permission
#     (System Settings > Privacy & Security > Microphone)
# =============================================================================
set -e

# Go to the repo root (this script lives in scripts/).
cd "$(dirname "$0")/.."

PY=".venv/bin/python"

# 1) One shared password ("token") for this session. Both programs below inherit
#    it automatically from the environment — nothing to copy or paste.
export VESPER_GATEWAY_TOKEN="${VESPER_GATEWAY_TOKEN:-$("$PY" -c 'import secrets; print(secrets.token_hex(16))')}"

echo "──────────────────────────────────────────────"
echo " VESPER voice — starting"
echo " gateway : http://127.0.0.1:8760"
echo " token   : (auto, shared with the voice listener)"
echo " wake    : say  “Hey Jarvis”  then your request"
echo " stop    : press Ctrl+C"
echo "──────────────────────────────────────────────"

# 2) Start the gateway (the brain) in the background.
"$PY" -m gateway.server &
GATEWAY_PID=$!

# Stop the gateway automatically when this script exits.
trap 'echo; echo "stopping VESPER..."; kill "$GATEWAY_PID" 2>/dev/null || true' EXIT

# 3) Give the gateway a few seconds to come up, then start the voice listener
#    in the foreground (this is the window you watch / Ctrl+C to quit).
sleep 5
"$PY" -m voice.input
