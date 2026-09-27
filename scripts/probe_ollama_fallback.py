#!/usr/bin/env python3
"""
Reproduce Vesper's Ollama fallback call outside the router's normal path.

Goal: determine whether the router's 53/53 observed "Ollama network error"
failures under load are (a) a cold-load-vs-client-timeout race, (b) the
wrong model size configured for the fallback tier, (c) contention with a
memory-starved machine, or (d) something specific to how the router invokes
Ollama that a standalone call wouldn't hit.

Uses `OllamaProvider` directly (`llm/providers/ollama_provider.py`) — the
same class `ModelRouter._attempt_tier` calls for the fallback tier — rather
than a bare `ollama` SDK call, so this matches Vesper's real call shape
(message normalization, keep_alive, num_ctx, error wrapping) exactly.

    python scripts/probe_ollama_fallback.py
"""
from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.providers.ollama_provider import OllamaProvider  # noqa: E402

MESSAGES = [{"role": "user", "content": "Reply with just the word: ok"}]

# The model the router is ACTUALLY configured to fall back to
# (config/settings.py's LLMOllamaSettings default, line ~349) vs. the one
# scripts/doctor.py's own health check validates and its comment calls the
# safe ceiling ("keep llm.fallback on a 3B, never a 7B").
CONFIGURED_FALLBACK_MODEL = "qwen3.5:latest"  # 6.6GB per `ollama list`
DOCTOR_DOCUMENTED_SAFE_MODEL = "llama3.2:3b"  # 2.0GB per `ollama list`


def free_ram_gb() -> float:
    """Free + inactive + speculative memory in GB — same formula as
    scripts/doctor.py's `_free_ram_gb`, duplicated here so this probe has no
    import-time dependency on doctor.py's CLI-only module structure."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
    text = out.stdout
    page_size = 4096
    match = re.search(r"page size of (\d+) bytes", text)
    if match:
        page_size = int(match.group(1))
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        found = re.search(rf"{label}:\s+(\d+)", text)
        if found:
            pages += int(found.group(1))
    return (pages * page_size) / (1024**3)


async def timed_call(provider: OllamaProvider, model: str, label: str) -> None:
    start = time.monotonic()
    free_before = free_ram_gb()
    try:
        result = await provider.complete(messages=MESSAGES, model=model, tools=None)
        elapsed = time.monotonic() - start
        print(
            f"{label}: OK in {elapsed:.1f}s (free RAM before call: {free_before:.2f} GB) "
            f"— reply: {result.text[:60]!r}"
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(
            f"{label}: FAILED after {elapsed:.1f}s (free RAM before call: {free_before:.2f} GB) "
            f"— {type(exc).__name__}: {exc}"
        )


async def main() -> None:
    print(f"Total free/reclaimable RAM right now: {free_ram_gb():.2f} GB\n")

    print("=== Test 1: configured fallback model, cold (first call this process) ===")
    provider = OllamaProvider()
    await timed_call(provider, CONFIGURED_FALLBACK_MODEL, "Test 1 (cold, configured model)")

    print("\n=== Test 2: same model, warm burst (5 calls, 0.5s apart) ===")
    for i in range(5):
        await timed_call(provider, CONFIGURED_FALLBACK_MODEL, f"Test 2.{i} (warm burst)")
        await asyncio.sleep(0.5)

    print(
        "\n=== Test 3: sleep past keep_alive (30s) to force unload, then call the "
        "CONFIGURED model cold again ==="
    )
    await asyncio.sleep(32)
    fresh_provider = OllamaProvider()  # fresh client instance, matches a real fallback crossing
    await timed_call(fresh_provider, CONFIGURED_FALLBACK_MODEL, "Test 3 (cold after keep_alive expiry, configured model)")

    print(
        "\n=== Test 4: sleep past keep_alive again, call the doctor.py-documented "
        "SAFE 3B model cold, for direct size comparison ==="
    )
    await asyncio.sleep(32)
    fresh_provider_2 = OllamaProvider()
    await timed_call(fresh_provider_2, DOCTOR_DOCUMENTED_SAFE_MODEL, "Test 4 (cold, 3B safe model)")


if __name__ == "__main__":
    asyncio.run(main())
