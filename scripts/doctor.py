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
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


def main() -> int:
    print("=" * 64)
    print(f"VESPER doctor — {platform.platform()}")
    print("=" * 64)
    check_python()
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
