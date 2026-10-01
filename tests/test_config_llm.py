"""llm.rate_limit and llm.tool_selection in settings.yaml must reach the router
and planner. They were silently dropped by the pydantic loader (found in the
2026-10-02 audit): the YAML's tuned 6s wait ceiling never applied."""

from __future__ import annotations

from pathlib import Path

from config.settings import load_config_dict
from llm.router import ModelRouter
from orchestrator.planner import Planner
from tools.registry import ToolRegistry
from tracing.tracer import Tracer


def _load(monkeypatch, tmp_path: Path, body: str):
    """Load a custom YAML. NOTE: AppSettings only honours the VESPER_CONFIG env
    var (settings_customise_sources); load_config_dict(path) ignores its argument."""
    path = tmp_path / "settings.yaml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("VESPER_CONFIG", str(path))
    return load_config_dict()


def test_repo_settings_yaml_rate_limit_is_loaded():
    llm = load_config_dict()["llm"]
    assert llm["rate_limit"]["tokens_per_minute"] == 8000
    assert llm["rate_limit"]["max_wait_seconds"] == 6.0  # the value in config/settings.yaml
    assert llm["tool_selection"]["enabled"] is True


def test_custom_values_reach_the_router(tmp_path, monkeypatch):
    cfg = _load(monkeypatch, tmp_path, (
        "llm:\n"
        "  rate_limit:\n"
        "    tokens_per_minute: 1234\n"
        "    max_wait_seconds: 9.5\n"
    ))
    router = ModelRouter(config=cfg)
    assert router._max_rate_limit_wait == 9.5
    assert router._get_budget("groq").tokens_per_minute == 1234


def test_defaults_when_block_absent(tmp_path, monkeypatch):
    cfg = _load(monkeypatch, tmp_path, "general:\n  assistant_name: T\n")
    router = ModelRouter(config=cfg)
    assert router._max_rate_limit_wait == ModelRouter.DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS
    assert router._get_budget("groq").tokens_per_minute == ModelRouter.DEFAULT_TOKENS_PER_MINUTE


def test_tool_selection_flag_reaches_the_planner(tmp_path, monkeypatch):
    off = _load(monkeypatch, tmp_path, "llm:\n  tool_selection:\n    enabled: false\n")
    assert off["llm"]["tool_selection"]["enabled"] is False
    planner = Planner(router=None, registry=ToolRegistry(), config=off,
                      tracer=Tracer(config={"tracing": {"enabled": False}}))
    assert planner._tool_selection_enabled is False

    on = _load(monkeypatch, tmp_path, "llm:\n  tool_selection:\n    enabled: true\n")
    planner = Planner(router=None, registry=ToolRegistry(), config=on,
                      tracer=Tracer(config={"tracing": {"enabled": False}}))
    assert planner._tool_selection_enabled is True
