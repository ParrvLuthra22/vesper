#!/usr/bin/env bash
#
# Start the Ollama server tuned for Vesper's rescue-fallback role on an 8GB Mac.
#
#     ./scripts/ollama_env.sh          # run in the foreground
#     ./scripts/ollama_env.sh &        # background
#
# Why these settings, and why a script rather than "just run ollama serve":
#
#   OLLAMA_KEEP_ALIVE=30s
#       Unload the model 30 seconds after the last call. Ollama's default
#       (5m) keeps ~2GB pinned long after a one-off rescue, which on 8GB is
#       memory the OS and Vesper have to page around. Vesper calls Ollama
#       rarely — only when Groq is rate-limited or unreachable — so paying a
#       3-5s cold load per rescue is the right trade for getting the RAM back.
#
#   OLLAMA_CONTEXT_LENGTH=2048
#       Smaller context = smaller KV cache = less RAM per loaded model. The
#       planner's prompts are trimmed to fit well inside this.
#
#   OLLAMA_MAX_LOADED_MODELS=1
#       Never hold two models resident at once. On 8GB that is an instant swap
#       storm.
#
# Vesper also sends keep_alive and num_ctx on every request (see
# llm/providers/ollama_provider.py), so the unload behavior holds even against
# a server someone started by hand. This script makes it the default.

set -euo pipefail

export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30s}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-2048}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"

if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama not found. Install it:  brew install ollama" >&2
    exit 1
fi

echo "Starting Ollama:"
echo "  OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}          (unload when idle)"
echo "  OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH}      (small KV cache)"
echo "  OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS}    (never two at once)"
echo

exec ollama serve
