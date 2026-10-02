#!/bin/bash
# =============================================================================
# VESPER — one-command launcher (now a thin wrapper)
# =============================================================================
# The logic that used to live here — one shared auto-generated token, gateway
# first, then the voice listener, stop everything on exit — moved into the
# launcher (launcher/), which also starts voice output and the HUD, waits for
# each component to be ready, restarts crashes with backoff, and shuts down in
# order. This script just calls it:
#
#   ./scripts/run_voice.sh        ==        vesper up
#
# One-time requirements (see scripts/doctor.py):
#   - voice deps installed (voice/requirements.txt)
#   - your terminal app has Microphone permission
#     (System Settings > Privacy & Security > Microphone)
# =============================================================================
set -e
cd "$(dirname "$0")/.."
exec .venv/bin/python -m launcher up "$@"
