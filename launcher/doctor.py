"""`vesper doctor` — a read-only environment check. It sends nothing, retrieves no secret, and by default
does not open the microphone (opening it can raise a permission prompt).

One line per check, PASS / WARN / FAIL, and a `fix:` line under every WARN and FAIL. Output holds names,
counts, ages and paths only — never mail, message text, or a credential. Exit status 1 if anything FAILs.

    vesper doctor              # everything, no prompts
    vesper doctor --probe-mic  # additionally open the mic for ~2 s and count frames (may show the macOS prompt)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"


@dataclass
class Check:
    status: str
    label: str
    detail: str = ""
    fix: str = ""


def _get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return default if cur is None else cur


def _age(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)}s"
    if seconds < 5400:
        return f"{int(seconds // 60)}m"
    if seconds < 172800:
        return f"{seconds / 3600:.0f}h"
    return f"{seconds / 86400:.0f}d"


# ------------------------------------------------------------------ microphone

def check_microphone(probe: bool, devices: Optional[Callable[[], List[Dict[str, Any]]]] = None,
                     frames: Optional[Callable[[float], int]] = None, device: Optional[int] = None) -> List[Check]:
    def default_devices() -> List[Dict[str, Any]]:
        import sounddevice as sd

        return [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]

    def default_frames(seconds: float) -> int:
        import sounddevice as sd

        n = {"frames": 0}

        def cb(indata, frame_count, t, status):  # noqa: ARG001
            n["frames"] += frame_count

        with sd.InputStream(samplerate=16000, channels=1, callback=cb, device=device):   # the PINNED device, not the default
            time.sleep(seconds)
        return n["frames"]

    out: List[Check] = []
    try:
        found = (devices or default_devices)()
    except Exception as exc:  # noqa: BLE001
        return [Check(FAIL, "Microphone", f"cannot list input devices ({type(exc).__name__})",
                      "pip install sounddevice; brew install portaudio")]
    if not found:
        return [Check(FAIL, "Microphone", "no input device found", "connect or enable a microphone")]
    out.append(Check(PASS, "Microphone devices", f"{len(found)} input device(s)"))
    if not probe:
        out.append(Check(WARN, "Microphone permission", "not tested (opening the mic can show a macOS prompt)",
                         "run `vesper doctor --probe-mic`; if blocked: System Settings > Privacy & Security > "
                         "Microphone > enable your terminal"))
        return out
    try:
        n = (frames or default_frames)(2.0)
    except Exception as exc:  # noqa: BLE001
        out.append(Check(FAIL, "Microphone permission", f"could not open the stream ({type(exc).__name__})",
                         "grant Microphone access to your terminal; close apps holding the mic"))
        return out
    out.append(Check(PASS, "Microphone permission", f"{n} frames in 2 s") if n > 0 else Check(
        FAIL, "Microphone permission", "stream opened but delivered no audio (permission prompt pending/denied)",
        "System Settings > Privacy & Security > Microphone > enable your terminal, then retry"))
    return out


def check_input_device(setting: Any, devices: Optional[Callable[[], List[Dict[str, Any]]]] = None,
                       default_index: Optional[int] = None, transports: Optional[Dict[str, str]] = None,
                       use_real_default: bool = True) -> List[Check]:
    """Which input device would voice input use (NAME only), is the system default something else, is the built-in mic there.
    Pure given its inputs; with no stand-ins it reads the real device list and CoreAudio properties (no stream is opened)."""
    from voice.input.devices import (
        DeviceResolutionError, default_input_index, find_builtin, list_input_devices, read_transport_types,
        resolve_input_device)

    try:
        found = list_input_devices(devices)
    except Exception as exc:  # noqa: BLE001
        return [Check(FAIL, "Voice input device", f"cannot list input devices ({type(exc).__name__})",
                      "pip install sounddevice; brew install portaudio")]
    if transports is None:
        transports = read_transport_types() if use_real_default else {}
    if default_index is None and use_real_default:
        default_index = default_input_index()
    out: List[Check] = []
    builtin = find_builtin(found, transports)
    try:
        chosen = resolve_input_device(setting, found, transports, default_index)
        out.append(Check(PASS, "Voice input device", f"would use '{chosen.name}' (voice.input.device = {setting!r})"))
    except DeviceResolutionError as exc:
        out.append(Check(FAIL, "Voice input device", str(exc),
                         "plug in/enable the microphone, or set voice.input.device in settings.yaml; voice input "
                         "never falls back to another device"))
    if builtin is None:
        out.append(Check(FAIL if str(setting).strip().lower() in ("builtin", "built-in", "built_in", "internal", "macbook", "none")
                         else WARN, "Built-in microphone",
                         "not found among: " + (", ".join(f"'{d.name}'" for d in found) or "no input devices"),
                         "System Settings > Sound > Input should list 'MacBook … Microphone'; with a lid closed (clamshell) "
                         "the built-in mic is disabled"))
    else:
        out.append(Check(PASS, "Built-in microphone", f"found '{builtin.name}'"))
        default = next((d for d in found if d.index == default_index), None) if default_index is not None else None
        if default is None:
            out.append(Check(WARN, "System default input", "could not be determined (no default input device reported)",
                             "informational: voice input is pinned (voice.input.device), so it does not depend on the default"))
        else:
            if default.index != builtin.index:
                out.append(Check(WARN, "System default input", f"is '{default.name}', not the built-in mic",
                                 "voice input is pinned (voice.input.device) so this does not affect it; change the macOS "
                                 "default if you want other apps on the built-in mic too"))
            else:
                out.append(Check(PASS, "System default input", f"is the built-in mic ('{default.name}')"))
    return out


def pinned_input_index(setting: Any) -> Optional[int]:
    """Index of the device voice input would use, or None if it cannot be resolved (read-only)."""
    from voice.input.devices import (
        DeviceResolutionError, default_input_index, list_input_devices, read_transport_types, resolve_input_device)

    try:
        return resolve_input_device(setting, list_input_devices(), read_transport_types(), default_input_index()).index
    except (DeviceResolutionError, Exception):  # noqa: BLE001
        return None


# ------------------------------------------------------------------ Telegram token (presence only)

def keychain_has(service: str, account: str = "bot-token") -> bool:
    """`security find-generic-password` WITHOUT -w: it reports whether the item exists and never returns the secret."""
    try:
        return subprocess.run(["security", "find-generic-password", "-s", service, "-a", account],
                              capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def check_telegram(cfg: Dict[str, Any], env: Dict[str, str],
                   keychain: Callable[[str], bool] = keychain_has) -> List[Check]:
    tg = _get(cfg, "channels.telegram", {}) or {}
    if not tg.get("enabled"):
        return [Check(PASS, "Telegram channel", "disabled (channels.telegram.enabled is false) — nothing to check")]
    out: List[Check] = []
    env_name, service = str(tg.get("bot_token_env") or ""), str(tg.get("keychain_service") or "")
    sources = []
    if env_name and env.get(env_name):
        sources.append(f"environment (${env_name})")
    if service and keychain(service):
        sources.append(f"Keychain ({service})")
    if tg.get("bot_token"):
        sources.append("config file")
    if not sources:
        out.append(Check(FAIL, "Telegram bot token", "not found in the environment or the Keychain",
                         f"security add-generic-password -s {service or 'vesper-telegram-bot'} -a bot-token -w"))
    elif sources == ["config file"]:
        out.append(Check(WARN, "Telegram bot token", "present, but only in a git-tracked config file",
                         "move it to the Keychain and clear channels.telegram.bot_token"))
    else:
        out.append(Check(PASS, "Telegram bot token", "present in " + " and ".join(sources) + " (value not read)"))
    ids = tg.get("allowed_user_ids") or []
    out.append(Check(PASS, "Telegram allowlist", f"{len(ids)} numeric user id(s)") if ids else Check(
        FAIL, "Telegram allowlist", "channels.telegram.allowed_user_ids is empty (the channel refuses to start)",
        "set your numeric Telegram user id (see docs/CHANNELS.md)"))
    return out


# ------------------------------------------------------------------ model files

def check_models(cfg: Dict[str, Any], root: Path = PROJECT_ROOT, hf_cache: Path = HF_CACHE) -> List[Check]:
    out: List[Check] = []
    g = lambda p, d: _get(cfg, p, d)  # noqa: E731
    if g("voice.output.enabled", True):
        for key, default in (("model_path", "voice/models/kokoro-v1.0.onnx"), ("voices_path", "voice/models/voices-v1.0.bin")):
            path = Path(str(g(f"voice.output.{key}", default)))
            path = path if path.is_absolute() else root / path
            out.append(Check(PASS, f"Kokoro {key}", path.name) if path.exists() else Check(
                FAIL, f"Kokoro {key}", f"missing: {path}", "download the Kokoro files into voice/models/ (see voice/README)"))
    if g("voice.input.enabled", True):
        whisper = str(g("voice.input.whisper_model", "base.en"))
        found = (hf_cache / f"models--Systran--faster-whisper-{whisper}").exists()
        out.append(Check(PASS, f"Whisper model ({whisper})", "in the Hugging Face cache") if found else Check(
            FAIL, f"Whisper model ({whisper})", "not in the local cache; the launcher runs offline",
            "set launcher.offline_models: false once and run `vesper up` to download it"))
        wake = str(g("voice.input.wake_model", "hey_jarvis"))
        if wake.endswith(".onnx"):
            p = Path(wake) if Path(wake).is_absolute() else root / str(g("voice.input.models_dir", "voice/models")) / Path(wake).name
            ok = p.exists() or Path(wake).exists()
            fb = str(g("voice.input.wake_model_fallback", ""))
            out.append(Check(PASS, "Wake model", Path(wake).name) if ok else Check(
                WARN if fb else FAIL, "Wake model", f"custom model not on disk ({Path(wake).name})",
                f"train it (voice/TRAINING.md); until then the fallback '{fb}' is used" if fb else "train or fetch the model"))
        else:
            out.append(Check(PASS, "Wake model", f"pretrained '{wake}'"))
    if g("channels.telegram.enabled", False):
        out.append(Check(PASS, "Local speech model for voice notes", "same faster-whisper model as voice input (above)"))
    emb = (hf_cache / "models--sentence-transformers--all-MiniLM-L6-v2").exists()
    out.append(Check(PASS, "Embedding model (MiniLM)", "in the Hugging Face cache") if emb else Check(
        WARN, "Embedding model (MiniLM)", "not cached; semantic memory will not work offline",
        "launcher.offline_models: false once, then `vesper up`"))
    return out


# ------------------------------------------------------------------ ports

def port_busy(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


def check_ports(cfg: Dict[str, Any], supervisor_pid: Optional[int], busy: Callable[[str, int], bool] = port_busy) -> List[Check]:
    host, port = str(_get(cfg, "gateway.host", "127.0.0.1")), int(_get(cfg, "gateway.port", 8760))
    if busy(host, port):
        if supervisor_pid:
            return [Check(PASS, f"Gateway port {port}", f"in use by the running stack (supervisor pid {supervisor_pid})")]
        return [Check(FAIL, f"Gateway port {port}", f"in use by something that is not Vesper's stack on {host}",
                      f"find it: lsof -nP -iTCP:{port} -sTCP:LISTEN ; stop it, or change gateway.port")]
    return [Check(PASS, f"Gateway port {port}", f"free on {host}")]


# ------------------------------------------------------------------ briefing cache

def check_briefing(db_path: str, interval_minutes: float = 10.0, now: Optional[float] = None) -> List[Check]:
    now = time.time() if now is None else now
    p = Path(db_path)
    if not p.exists():
        return [Check(WARN, "Briefing cache", "no cache file yet", "it is created by the first refresh: `vesper up`, "
                      "or `vesper briefing --refresh`")]
    try:
        db = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=2)
        rows = db.execute("SELECT source, last_success, consecutive_failures FROM sources").fetchall()
        unread = db.execute("SELECT COUNT(*) FROM items WHERE source='gmail' AND active=1").fetchone()[0]
        db.close()
    except sqlite3.Error as exc:
        return [Check(FAIL, "Briefing cache", f"cannot read it ({type(exc).__name__})", "delete data/briefing.db; it is rebuilt")]
    if not rows:
        return [Check(WARN, "Briefing cache", "exists but no source has synced", "`vesper briefing --refresh`")]
    out = []
    limit = max(interval_minutes * 3 * 60, 1800)
    for source, last_success, failures in rows:
        if not last_success:
            out.append(Check(FAIL, f"Briefing cache: {source}", "never synced successfully",
                             "`vesper briefing --refresh`; check logs/launcher/gateway.log"))
            continue
        age = now - last_success
        extra = f", {failures} consecutive failure(s)" if failures else ""
        status = PASS if age <= limit and not failures else WARN
        out.append(Check(status, f"Briefing cache: {source}", f"last sync {_age(age)} ago{extra}",
                         "" if status == PASS else "`vesper briefing --refresh`; see logs/launcher/gateway.log"))
    out.append(Check(PASS, "Briefing cache: unread mail", f"{unread} cached item(s) (counts only)"))
    return out


# ------------------------------------------------------------------ messages (Full Disk Access)

def check_messages(enabled: bool, db_path: str, state: Optional[Callable[[str], str]] = None) -> List[Check]:
    """Can this terminal open Messages' chat.db? Only the OPEN is attempted (what Full Disk Access gates): no
    byte of the database is read, nothing is copied, no prompt is shown."""
    from briefing.messages import FDA_STEPS, permission_state

    result = (state or permission_state)(db_path)
    on = "enabled" if enabled else "disabled (messages.enabled: false)"
    if result == "ok":
        return [Check(PASS, "Messages: Full Disk Access", f"chat.db can be opened; collector {on} (nothing was read)",
                      "" if enabled else "set messages.enabled: true in config/briefing.local.yaml to include it")]
    if result == "denied":
        return [Check(WARN, "Messages: Full Disk Access", f"not granted; collector {on}", FDA_STEPS)]
    return [Check(WARN if enabled else PASS, "Messages: chat.db", f"no Messages database found; collector {on}",
                  "sign in to Messages on this Mac, or point messages.db_path at the right file")]


# ------------------------------------------------------------------ audit log

def check_audit(path: Path, now: Optional[float] = None) -> List[Check]:
    now = time.time() if now is None else now
    if not path.exists():
        return [Check(WARN, "Audit log", "no audit file yet (nothing has needed the Guardian)")]
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            last = [l for l in f.read().decode("utf-8", "replace").splitlines() if l.strip()][-1:]
    except (OSError, IndexError):
        return [Check(WARN, "Audit log", "empty or unreadable")]
    try:
        entry = json.loads(last[0])
    except (ValueError, IndexError):
        return [Check(FAIL, "Audit log", "the last line is not valid JSON", "inspect data/audit.jsonl (never edit it)")]
    channel = entry.get("channel")
    age = ""
    try:
        from datetime import datetime

        age = f", {_age(now - datetime.fromisoformat(entry['timestamp']).timestamp())} ago"
    except Exception:  # noqa: BLE001
        pass
    if channel:
        return [Check(PASS, "Audit log: last entry channel", f"'{channel}'{age}")]
    return [Check(WARN, "Audit log: last entry channel", f"no channel field{age} (written before channels existed)",
                  "expected until the next Guardian decision; every new entry records one")]


# ------------------------------------------------------------------ calendar permission, wake model

#: TERM_PROGRAM -> the name shown in System Settings.
_TERMINALS = {"Apple_Terminal": "Terminal", "iTerm.app": "iTerm", "ghostty": "Ghostty", "vscode": "Visual Studio Code",
              "WarpTerminal": "Warp", "Hyper": "Hyper", "WezTerm": "WezTerm", "kitty": "kitty", "Alacritty": "Alacritty"}


def terminal_app(env: Dict[str, str]) -> str:
    return _TERMINALS.get(env.get("TERM_PROGRAM", ""), env.get("TERM_PROGRAM") or "the terminal app you start Vesper from")


def calendar_authorization() -> Optional[int]:
    """EventKit authorization status for events WITHOUT asking (no prompt): 0 not determined, 1 restricted, 2 denied,
    3 full access, 4 write-only. None = cannot be determined (not macOS / pyobjc missing)."""
    try:
        import EventKit

        return int(EventKit.EKEventStore.authorizationStatusForEntityType_(EventKit.EKEntityTypeEvent))
    except Exception:  # noqa: BLE001
        return None


def check_calendar_access(env: Optional[Dict[str, str]] = None, status: Optional[Callable[[], Optional[int]]] = None) -> List[Check]:
    """Voice/`what's on my calendar` reads Calendar through EventKit inside the apple_pim MCP server. That python process is
    started by `vesper up`, so macOS attributes the request to the TERMINAL APP that ran it — that is the app that must be
    allowed, not "python" and not the launcher."""
    env = dict(os.environ) if env is None else env
    st = (status or calendar_authorization)()
    app = terminal_app(env)
    where = f"System Settings > Privacy & Security > Calendars > turn on {app} (macOS 14+: set it to Full Calendar Access)"
    if st == 3:
        return [Check(PASS, "Calendar access", f"granted ({app}); the calendar skill and briefing can read events")]
    if st is None:
        return [Check(WARN, "Calendar access", "could not be determined (EventKit unavailable)",
                      "pip install pyobjc-framework-EventKit; then re-run `vesper doctor`")]
    names = {0: "not asked yet", 1: "restricted by a profile", 2: "denied", 4: "write-only"}
    if st == 0:
        return [Check(WARN, "Calendar access", f"{names[0]} for {app}: the first calendar question will show a prompt",
                      f"run `vesper briefing --refresh` once from {app} and click Allow, or enable it in {where}")]
    return [Check(FAIL, "Calendar access", f"{names.get(st, f'status {st}')} for {app}: 'what's on my calendar' cannot work",
                  f"{where}; then restart `vesper up`. (The permission belongs to {app}, the app that launches Vesper.)")]


def check_wake_model(cfg: Dict[str, Any], root: Path = PROJECT_ROOT) -> List[Check]:
    path = str(_get(cfg, "voice.wake.model_path", "") or "").strip()
    if not path:
        return [Check(PASS, "Wake model", f"{_get(cfg, 'voice.input.wake_model', 'hey_jarvis')} (bundled)")]
    p = Path(path)
    for cand in (p, root / p, root / "voice" / "models" / p.name):
        if cand.is_file() and cand.stat().st_size > 0:
            return [Check(PASS, "Wake model", f"custom model {cand.name} ({cand.stat().st_size // 1024} KB)")]
    return [Check(WARN, "Wake model", f"voice.wake.model_path is set but the file does not exist ({p.name}); voice input falls back "
                  f"to {_get(cfg, 'voice.input.wake_model_fallback', 'hey_jarvis')}",
                  "train it with scripts/train_hey_vesper/ and copy it to voice/models/ (docs/WAKEWORD.md), or clear voice.wake.model_path")]


# ------------------------------------------------------------------ run

def run_checks(cfg: Dict[str, Any], *, env: Optional[Dict[str, str]] = None, probe_mic: bool = False,
               root: Path = PROJECT_ROOT, supervisor_pid: Optional[int] = None,
               audit_path: Optional[Path] = None, briefing_db: Optional[str] = None,
               briefing_interval: float = 10.0, messages: Optional[tuple] = None,
               voice_input_device: Optional[tuple] = None, calendar: Optional[Callable[[], Optional[int]]] = None,
               check_calendar: bool = False) -> List[Check]:
    env = dict(os.environ) if env is None else env
    audit = audit_path or Path(env.get("VESPER_AUDIT_LOG") or root / "data" / "audit.jsonl")
    db = env.get("VESPER_BRIEFING_DB") or briefing_db or str(root / "data" / "briefing.db")
    pinned = pinned_input_index(voice_input_device[0]) if (voice_input_device and probe_mic) else None
    return [*check_microphone(probe_mic, device=pinned),
            *(check_input_device(voice_input_device[0]) if voice_input_device else []), *check_telegram(cfg, env), *check_models(cfg, root),
            *check_ports(cfg, supervisor_pid), *check_briefing(db, briefing_interval),
            *(check_messages(*messages) if messages else []), *check_wake_model(cfg, root),
            *(check_calendar_access(env, calendar) if (check_calendar or calendar) else []), *check_audit(audit)]


def render(checks: List[Check], color: bool = False) -> str:
    paint = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}
    lines = []
    for c in checks:
        tag = f"{paint[c.status]}{c.status}\033[0m" if color else c.status
        lines.append(f"[{tag}] {c.label}" + (f" — {c.detail}" if c.detail else ""))
        if c.fix and c.status != PASS:
            lines.append(f"       fix: {c.fix}")
    n = {s: sum(1 for c in checks if c.status == s) for s in (PASS, WARN, FAIL)}
    lines.append(f"\n{n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="vesper doctor", description=__doc__.split("\n\n")[0])
    ap.add_argument("--probe-mic", action="store_true", help="open the microphone for ~2 s (may show the macOS prompt)")
    args = ap.parse_args(argv)
    from launcher.stack import load_app_config
    from launcher.state import RunState

    cfg = load_app_config()
    try:
        pid = RunState(PROJECT_ROOT / "data" / "run").running_supervisor_pid()
    except Exception:  # noqa: BLE001
        pid = None
    from briefing.config import load_briefing_config

    bcfg = load_briefing_config()
    from voice.input.config import VoiceInputConfig

    checks = run_checks(cfg, probe_mic=args.probe_mic, supervisor_pid=pid, briefing_db=bcfg.cache_path,
                        briefing_interval=float(bcfg.interval_minutes),
                        messages=(bcfg.messages_enabled, bcfg.messages_db_path),
                        voice_input_device=(VoiceInputConfig.from_app_config(cfg).device,), check_calendar=True)
    print(render(checks, color=sys.stdout.isatty()))
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
