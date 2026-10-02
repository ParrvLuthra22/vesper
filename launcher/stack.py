"""
Builds the component list for `vesper up` from the app config, and runs it.

Order (each starts only after the previous is READY):

    1. gateway       python -m gateway.server   the Brain, MCP servers, planner  [required]
    2. voice_output  python -m voice.output     Kokoro TTS; speaks the gateway's replies
    3. hud           hud/src-tauri/target/release/hud   the Tauri overlay
    4. voice_input   python -m voice.input      openWakeWord + VAD + faster-whisper
    5. telegram      python -m channels.telegram   OPTIONAL: only when channels.telegram.enabled

Voice input is last on purpose: once the wake word can fire, a reply has
somewhere to be spoken and shown. "Wake word" and "speech-to-text" are one
process (voice.input) — they share the microphone stream and its buffers.

Readiness:
    gateway       GET /status with the session token returns 200
    voice_output prints "voice output connected to gateway"
    hud           the gateway's connected-client count rises (the HUD attached)
    voice_input  prints "voice input ready" (wake model loaded, mic streaming)
    telegram     prints "telegram channel ready" (config + token valid; polling started)
"""

from __future__ import annotations

import asyncio
import os
import secrets
import signal
import socket
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from launcher.state import RunState
from launcher.supervisor import (
    ComponentSpec,
    Probe,
    ProbeContext,
    RestartPolicy,
    Spawn,
    Supervisor,
    line_probe,
    make_spawner,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUD_BINARY = Path("hud/src-tauri/target/release/hud")
#: voice.input exit status meaning "unavailable this session" (voice/input/service.py).
VOICE_INPUT_UNAVAILABLE = 69
#: channels.telegram exit status meaning "cannot work and retrying will not help" (channels/telegram.py).
TELEGRAM_UNAVAILABLE = 69


def _get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = cfg
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return default if value is None else value


def gateway_address(cfg: Dict[str, Any]) -> tuple:
    return str(_get(cfg, "gateway.host", "127.0.0.1")), int(_get(cfg, "gateway.port", 8760))


def port_in_use(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def _http_get_json(url: str, token: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    import json

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def gateway_status(cfg: Dict[str, Any], token: str) -> Optional[Dict[str, Any]]:
    host, port = gateway_address(cfg)
    return _http_get_json(f"http://{host}:{port}/status", token)


def gateway_probe(cfg: Dict[str, Any], token: str) -> Probe:
    async def probe(ctx: ProbeContext) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: gateway_status(cfg, token) is not None)

    return probe


def hud_attached_probe(cfg: Dict[str, Any], token: str) -> Probe:
    """Ready when the gateway reports MORE connected clients than before the HUD
    started (voice output is already connected by then, so it is the baseline)."""

    async def probe(ctx: ProbeContext) -> bool:
        loop = asyncio.get_running_loop()
        status = await loop.run_in_executor(None, lambda: gateway_status(cfg, token))
        if status is None:
            return False
        baseline = ctx.data.setdefault("baseline", status.get("clients", 0))
        return status.get("clients", 0) > baseline

    return probe


def resolve_hud_binary(cfg: Dict[str, Any], root: Path = PROJECT_ROOT) -> Path:
    configured = str(_get(cfg, "launcher.hud_binary", "") or "")
    path = Path(configured) if configured else DEFAULT_HUD_BINARY
    return path if path.is_absolute() else root / path


def build_specs(cfg: Dict[str, Any], token: str, python: str = sys.executable, root: Path = PROJECT_ROOT) -> List[ComponentSpec]:
    host, port = gateway_address(cfg)
    env = {"VESPER_GATEWAY_TOKEN": token, "PYTHONUNBUFFERED": "1"}
    if bool(_get(cfg, "launcher.offline_models", True)):
        env["HF_HUB_OFFLINE"] = "1"

    comp_on = lambda name: bool(_get(cfg, f"launcher.components.{name}", True))  # noqa: E731

    specs: List[ComponentSpec] = [
        ComponentSpec(
            name="gateway",
            description="Brain + planner + MCP servers (python -m gateway.server)",
            argv=[python, "-m", "gateway.server"],
            env=env,
            required=True,
            probe=gateway_probe(cfg, token),
            ready_timeout=float(_get(cfg, "launcher.gateway_ready_timeout_seconds", 90.0)),
            hint="See logs/launcher/gateway.log.",
        )
    ]

    # voice output
    out_off = None
    if not comp_on("voice_output"):
        out_off = "disabled in config (launcher.components.voice_output)"
    elif not bool(_get(cfg, "voice.output.enabled", True)):
        out_off = "voice.output.enabled is false"
    specs.append(ComponentSpec(
        name="voice_output",
        description="Kokoro text-to-speech (python -m voice.output)",
        argv=[python, "-m", "voice.output"],
        env=env,
        probe=line_probe(r"voice output connected to gateway"),
        ready_timeout=float(_get(cfg, "launcher.voice_output_ready_timeout_seconds", 60.0)),
        permanent_exit_codes=frozenset({VOICE_INPUT_UNAVAILABLE}),  # 69: Kokoro cannot load
        skip_reason=out_off,
        hint="Run scripts/doctor.py; see logs/launcher/voice_output.log.",
    ))

    # HUD
    hud_bin = resolve_hud_binary(cfg, root)
    hud_off = None
    if not comp_on("hud"):
        hud_off = "disabled in config (launcher.components.hud)"
    elif not hud_bin.exists():
        hud_off = f"HUD is not built ({hud_bin}); build it: cd hud && npm install && npm run tauri build"
    specs.append(ComponentSpec(
        name="hud",
        description="Tauri overlay",
        argv=[str(hud_bin)],
        env=env,
        probe=hud_attached_probe(cfg, token),
        ready_timeout=float(_get(cfg, "launcher.hud_ready_timeout_seconds", 20.0)),
        skip_reason=hud_off,
        hint="See logs/launcher/hud.log.",
    ))

    # voice input (wake word + VAD + STT)
    in_off = None
    if not comp_on("voice_input"):
        in_off = "disabled in config (launcher.components.voice_input)"
    elif not bool(_get(cfg, "voice.input.enabled", True)):
        in_off = "voice.input.enabled is false"
    specs.append(ComponentSpec(
        name="voice_input",
        description="Wake word (openWakeWord) + VAD + speech-to-text (faster-whisper)",
        argv=[python, "-m", "voice.input"],
        env=env,
        probe=line_probe(r"voice input ready"),
        ready_timeout=float(_get(cfg, "launcher.voice_input_ready_timeout_seconds", 90.0)),
        permanent_exit_codes=frozenset({VOICE_INPUT_UNAVAILABLE}),
        skip_reason=in_off,
        hint=("Check microphone permission for your terminal (System Settings > Privacy & "
              "Security > Microphone) and run scripts/doctor.py."),
    ))

    # Telegram channel — optional. Part of the stack only when it is enabled, so a default install is
    # unchanged; when enabled it is a separate supervised process (a crash loop here can never take the
    # gateway down) and every config problem is exit 69 = "unavailable", not a restart loop.
    if bool(_get(cfg, "channels.telegram.enabled", False)):
        tg_off = None if comp_on("telegram") else "disabled in config (launcher.components.telegram)"
        specs.append(ComponentSpec(
            name="telegram",
            description="Telegram channel — restricted, owner-only (python -m channels.telegram)",
            argv=[python, "-m", "channels.telegram"],
            env=env,
            probe=line_probe(r"telegram channel ready"),
            ready_timeout=float(_get(cfg, "launcher.telegram_ready_timeout_seconds", 20.0)),
            permanent_exit_codes=frozenset({TELEGRAM_UNAVAILABLE}),
            skip_reason=tg_off,
            hint="See logs/launcher/telegram.log and docs/CHANNELS.md (bot token, allowed_user_ids).",
        ))
    return specs


def policy_from_config(cfg: Dict[str, Any]) -> RestartPolicy:
    g = lambda k, d: _get(cfg, f"launcher.restart.{k}", d)  # noqa: E731
    return RestartPolicy(
        initial_backoff=float(g("initial_backoff_seconds", 1.0)),
        max_backoff=float(g("max_backoff_seconds", 30.0)),
        max_crashes=int(g("max_crashes", 5)),
        window_seconds=float(g("window_seconds", 120.0)),
        healthy_reset_seconds=float(g("healthy_reset_seconds", 60.0)),
        max_startup_failures=int(g("max_startup_failures", 3)),
    )


def load_app_config() -> Dict[str, Any]:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    from config.settings import load_config_dict

    return load_config_dict()


async def run_up(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    run_state: Optional[RunState] = None,
    spawn: Optional[Spawn] = None,
    strict: bool = False,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
    install_signal_handlers: bool = True,
) -> int:
    """Start and supervise the stack until Ctrl+C / `vesper down` / a fatal failure."""
    cfg = cfg if cfg is not None else load_app_config()
    state = run_state or RunState(PROJECT_ROOT / "data" / "run")

    existing = state.running_supervisor_pid()
    if existing is not None:
        log(f"[launcher] Vesper is already running (supervisor pid {existing}). "
            "Use `vesper status`, or `vesper down` first.")
        return 1

    host, port = gateway_address(cfg)
    if port_in_use(host, port):
        log(f"[launcher] Port {port} on {host} is already in use — is another gateway running "
            "(or a stale `python -m gateway.server`)? Stop it, or change gateway.port.")
        return 1

    token = os.environ.get("VESPER_GATEWAY_TOKEN") or str(_get(cfg, "gateway.token", "") or "") or secrets.token_hex(16)
    state.write_token(token)

    specs = build_specs(cfg, token)
    supervisor = Supervisor(
        specs,
        policy_from_config(cfg),
        spawn=spawn or make_spawner(PROJECT_ROOT, PROJECT_ROOT / "logs" / "launcher"),
        on_change=state.write,
        log=log,
        shutdown_grace=float(_get(cfg, "launcher.shutdown_grace_seconds", 8.0)),
    )

    if install_signal_handlers:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, supervisor.request_stop)

    log("[launcher] starting Vesper: " + " -> ".join(
        ["gateway", "voice output", "HUD", "voice input"] + [s.name for s in specs if s.name == "telegram"]))
    try:
        ok = await supervisor.start()
        if ok and strict:
            bad = [c for c in supervisor.components if c.state.value in ("failed", "skipped")]
            if bad:
                log("[launcher] --strict: " + ", ".join(f"{c.spec.name} ({c.state.value})" for c in bad)
                    + " — stopping")
                ok = False
        if not ok:
            await supervisor.stop()
            return 1 if supervisor.fatal_reason or strict else 130
        _print_summary(supervisor, log)
        await supervisor.wait_closed()
        reason = supervisor.fatal_reason
        if reason:
            log(f"[launcher] FATAL — {reason}")
        log("[launcher] shutting down ...")
        await supervisor.stop()
        log("[launcher] stopped")
        return 1 if reason else 0
    finally:
        if install_signal_handlers:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)
        state.clear()


def _print_summary(supervisor: Supervisor, log: Callable[[str], None]) -> None:
    log("[launcher] ------------------------------------------------------------")
    for c in supervisor.components:
        pid = f" pid {c.handle.pid}" if c.handle is not None and c.handle.returncode is None else ""
        extra = f" — {c.detail}" if c.detail and c.state.value in ("failed", "skipped") else ""
        log(f"[launcher]   {c.spec.name:<13} {c.state.value.upper():<8}{pid}{extra}")
    voice_in = next((c for c in supervisor.components if c.spec.name == "voice_input"), None)
    if voice_in is not None and voice_in.state.value == "ready":
        log("[launcher] say the wake word (voice.input.wake_model) — Ctrl+C or `vesper down` to stop")
    else:
        log("[launcher] voice input is NOT running (see above) — text and the HUD still work; "
            "Ctrl+C or `vesper down` to stop")
    log("[launcher] ------------------------------------------------------------")
