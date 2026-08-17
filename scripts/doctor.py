#!/usr/bin/env python3
"""
VESPER environment doctor.

Run this when the startup summary reports something as unavailable — most
often voice input/output. It checks the interpreter, PortAudio, and every
optional voice/MCP dependency, then prints a PASS/FAIL line for each with a
concrete fix. Pure stdlib, so it always runs even when everything else is
broken.

    python scripts/doctor.py
    .venv/bin/python scripts/doctor.py
"""

from __future__ import annotations

import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Below this much free RAM, loading even a 3B model will push the machine
#: into swap. 3GB leaves room for llama3.2:3b at Q4 (~2GB) plus its KV cache.
MIN_FREE_GB_FOR_LOCAL_MODEL = 3.0

OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
FALLBACK_MODEL = "llama3.2:3b"

OK = "\033[32mPASS\033[0m"
NO = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def line(status: str, label: str, detail: str = "") -> None:
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    print("\nInterpreter")
    line(OK if v >= (3, 9) else NO, f"Python {v.major}.{v.minor}.{v.micro}",
         "main app targets 3.9+")


def check_portaudio() -> None:
    print("\nPortAudio (system audio backend)")
    found = False
    for probe in ("/opt/homebrew/opt/portaudio", "/usr/local/opt/portaudio"):
        if Path(probe).exists():
            found = True
            line(OK, "portaudio", probe)
            break
    if not found:
        line(NO, "portaudio", "install with:  brew install portaudio")


def check_voice_input() -> bool:
    print("\nVoice INPUT deps (voice/requirements.txt)")
    deps = {
        "sounddevice": "pip install sounddevice",
        "openwakeword": "pip install openwakeword",
        "faster_whisper": "pip install faster-whisper",
        "webrtcvad": "pip install webrtcvad   # or use silero",
    }
    all_ok = True
    for mod, fix in deps.items():
        ok = _has(mod)
        all_ok = all_ok and ok
        line(OK if ok else NO, mod, "" if ok else fix)
    return all_ok


def check_voice_output() -> None:
    print("\nVoice OUTPUT deps (Kokoro TTS — optional)")
    for mod, fix in {
        "kokoro_onnx": "pip install kokoro-onnx",
        "soundfile": "pip install soundfile",
    }.items():
        ok = _has(mod)
        line(OK if ok else WARN, mod, "" if ok else f"{fix}  (falls back to macOS 'say')")
    say = shutil.which("say")
    line(OK if say else WARN, "macOS 'say' fallback", say or "not found")


def check_mcp() -> None:
    print("\nMCP servers (separate 3.10+ venv)")
    venv_py = PROJECT_ROOT / "mcp_servers" / ".venv" / "bin" / "python3"
    if not venv_py.exists():
        line(NO, "mcp_servers/.venv", "create it and `pip install mcp` (SDK needs 3.10+)")
        return
    try:
        out = subprocess.run(
            [str(venv_py), "-c", "import mcp,sys;print(sys.version.split()[0])"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            line(OK, "mcp_servers/.venv", f"Python {out.stdout.strip()}, mcp SDK present")
        else:
            line(NO, "mcp_servers/.venv", "mcp SDK not importable")
    except Exception as exc:  # noqa: BLE001
        line(WARN, "mcp_servers/.venv", f"probe failed: {exc}")


def _total_ram_gb() -> float:
    """Physical RAM in GB, or 0.0 if it can't be read."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        return int(out.stdout.strip()) / (1024**3)
    except Exception:  # noqa: BLE001
        return 0.0


def _free_ram_gb() -> float:
    """
    Free + inactive + speculative memory in GB, per vm_stat.

    Inactive pages count as reclaimable here: macOS hands them back under
    pressure, so treating them as used would understate real headroom badly
    on a machine that has been up for a while.
    """
    try:
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
    except Exception:  # noqa: BLE001
        return 0.0


def check_memory_headroom() -> None:
    """
    Report RAM and current pressure, and warn when there isn't room to load
    a local model without swapping.

    This is the check that explains the "everything froze" symptom on an
    8GB machine: Ollama loading a model into a box with <3GB free pushes
    the whole system into swap, and the lag is systemic, not Vesper's.
    """
    print("\nMemory headroom (for the local LLM fallback)")
    total = _total_ram_gb()
    free = _free_ram_gb()

    if total <= 0:
        line(WARN, "RAM", "could not read hw.memsize")
        return

    line(OK, "total RAM", f"{total:.1f} GB")

    if free <= 0:
        line(WARN, "free memory", "could not read vm_stat")
        return

    detail = f"{free:.1f} GB free/reclaimable of {total:.1f} GB"
    if free >= MIN_FREE_GB_FOR_LOCAL_MODEL:
        line(OK, "memory pressure", detail)
    else:
        line(
            WARN,
            "memory pressure",
            f"{detail} — below {MIN_FREE_GB_FOR_LOCAL_MODEL:.0f} GB. Loading a local "
            "model now will swap and lag the Mac; close something first.",
        )

    if total < 12:
        line(
            OK,
            "fallback sizing",
            f"{total:.0f} GB machine — keep llm.fallback on a 3B ({FALLBACK_MODEL}), never a 7B",
        )


def check_fallback_reachable() -> None:
    """Ping the Ollama server and confirm the configured fallback model is pulled."""
    print("\nLLM fallback (Ollama — the 429/offline rescue path)")

    if not shutil.which("ollama"):
        line(NO, "ollama binary", "install with:  brew install ollama")
        return
    line(OK, "ollama binary", shutil.which("ollama") or "")

    try:
        with urllib.request.urlopen(f"{OLLAMA_ENDPOINT}/api/tags", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        line(NO, "ollama server", f"not reachable at {OLLAMA_ENDPOINT} ({exc}) — start it with ./scripts/ollama_env.sh")
        return

    line(OK, "ollama server", f"reachable at {OLLAMA_ENDPOINT}")

    models = [str(m.get("name", "")) for m in payload.get("models", [])]
    if any(name == FALLBACK_MODEL or name.startswith(f"{FALLBACK_MODEL}") for name in models):
        line(OK, f"model {FALLBACK_MODEL}", "pulled")
    else:
        line(
            NO,
            f"model {FALLBACK_MODEL}",
            f"not pulled — run:  ollama pull {FALLBACK_MODEL}"
            + (f"  (have: {', '.join(models)})" if models else ""),
        )


def main() -> int:
    print("=" * 64)
    print(f"VESPER doctor — {platform.platform()}")
    print("=" * 64)
    check_python()
    check_memory_headroom()
    check_fallback_reachable()
    check_portaudio()
    voice_ok = check_voice_input()
    check_voice_output()
    check_mcp()

    print("\n" + "-" * 64)
    if voice_ok:
        print("Voice input deps look OK. Set voice.input.enabled=true (or restart)\n"
              "to use the rebuilt pipeline; the legacy VoiceAgent will pick the mic\n"
              "up on next launch.")
    else:
        print("Voice input is DISABLED because deps above are missing. Install them:\n"
              "  pip install -r voice/requirements.txt\n"
              "then restart `python main.py` (voice re-enables on a clean launch).")
    print("-" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
