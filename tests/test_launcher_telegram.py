"""Telegram as an OPTIONAL supervised component of `vesper up`."""

from __future__ import annotations

from typing import Any, Dict

from launcher import stack

BASE: Dict[str, Any] = {
    "gateway": {"host": "127.0.0.1", "port": 8760},
    "voice": {"input": {"enabled": True}, "output": {"enabled": True}},
    "launcher": {"components": {"voice_output": True, "voice_input": True, "hud": True, "telegram": True},
                 "offline_models": True},
}


def cfg(**telegram) -> Dict[str, Any]:
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in BASE.items()}
    out["channels"] = {"telegram": telegram}
    return out


def names(c, tmp_path):
    return [s.name for s in stack.build_specs(c, "tok", python="/py", root=tmp_path)]


def test_default_stack_is_unchanged_when_telegram_is_off(tmp_path):
    assert "telegram" not in names(cfg(enabled=False), tmp_path)
    assert "telegram" not in names({k: v for k, v in BASE.items()}, tmp_path)        # no channels block at all


def test_enabled_telegram_is_an_optional_last_component(tmp_path):
    specs = stack.build_specs(cfg(enabled=True, allowed_user_ids=[1]), "tok123", python="/py", root=tmp_path)
    assert [s.name for s in specs][-1] == "telegram"
    tg = specs[-1]
    assert not tg.required and tg.skip_reason is None
    assert tg.argv == ["/py", "-m", "channels.telegram"]
    assert tg.env["VESPER_GATEWAY_TOKEN"] == "tok123" and tg.env["HF_HUB_OFFLINE"] == "1"
    assert tg.permanent_exit_codes == frozenset({69}) == frozenset({stack.TELEGRAM_UNAVAILABLE})   # config problems are not restarted
    assert not specs[0].env.get("VESPER_TELEGRAM_BOT_TOKEN")                  # the bot token is never put in a child's env by us


def test_enabled_but_switched_off_in_the_launcher_is_skipped_with_a_reason(tmp_path):
    c = cfg(enabled=True, allowed_user_ids=[1])
    c["launcher"]["components"]["telegram"] = False
    tg = [s for s in stack.build_specs(c, "t", python="/py", root=tmp_path) if s.name == "telegram"][0]
    assert "launcher.components.telegram" in tg.skip_reason


def test_ready_probe_and_timeout(tmp_path):
    c = cfg(enabled=True, allowed_user_ids=[1])
    c["launcher"]["telegram_ready_timeout_seconds"] = 7
    tg = stack.build_specs(c, "t", python="/py", root=tmp_path)[-1]
    assert tg.ready_timeout == 7.0 and tg.probe is not None


def test_a_telegram_failure_never_stops_the_required_gateway(tmp_path):
    specs = stack.build_specs(cfg(enabled=True, allowed_user_ids=[1]), "t", python="/py", root=tmp_path)
    assert [s.required for s in specs] == [True, False, False, False, False]
