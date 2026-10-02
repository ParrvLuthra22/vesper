"""
The legacy voice path is retired and nothing sends microphone audio off the
machine by default (docs/PRIVACY.md):

  - `python main.py` is a redirect to the launcher; it touches no audio code.
  - The legacy VoiceAgent is not registered by default.
  - With no Vosk model it refuses to listen — loudly — instead of falling back to
    "listen to everything".
  - Google web-speech (recognize_google) needs an explicit opt-in.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from bus.event_bus import EventBus

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_bus():
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


# ------------------------------------------------------------------ main.py

def test_main_py_redirects_to_the_launcher(monkeypatch, capsys):
    import importlib

    calls: List[List[str]] = []
    monkeypatch.setattr("launcher.cli.main", lambda argv: calls.append(list(argv)) or 0)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    sys.modules.pop("main", None)
    sys.path.insert(0, str(REPO))
    main_mod = importlib.import_module("main")
    with pytest.raises(SystemExit) as exc:
        main_mod.run()
    assert exc.value.code == 0 and calls == [["up"]]
    assert "redirect" in capsys.readouterr().out


def test_main_py_help_does_not_start_anything(monkeypatch, capsys):
    import importlib

    started: List[Any] = []
    monkeypatch.setattr("launcher.cli.main", lambda argv: started.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", ["main.py", "--help"])
    sys.modules.pop("main", None)
    main_mod = importlib.import_module("main")
    with pytest.raises(SystemExit) as exc:
        main_mod.run()
    assert exc.value.code == 0 and started == []
    assert "vesper up" in capsys.readouterr().out


def test_main_py_contains_no_audio_or_cloud_speech_code():
    src = (REPO / "main.py").read_text()
    code = re.sub(r'""".*?"""', "", src, flags=re.S)           # ignore the explanatory docstring
    for needle in ("VoiceAgent", "recognize_google", "pyttsx3", "sounddevice", "pyaudio", "speech_recognition", "Brain("):
        assert needle not in code, needle


def test_recognize_google_exists_only_in_the_gated_transcriber():
    offenders = []
    for path in REPO.rglob("*.py"):
        rel = path.relative_to(REPO)
        if any(p in {".venv", "build", "tests", "node_modules", "target", ".git"} for p in rel.parts):
            continue
        if "recognize_google" in path.read_text(encoding="utf-8", errors="replace"):
            offenders.append(str(rel))
    assert offenders == ["utils/stt.py"]


# ------------------------------------------------------- legacy VoiceAgent

def test_brain_does_not_register_the_legacy_voice_agent_by_default():
    import asyncio

    from orchestrator.brain import Brain

    brain = Brain(config={})
    assert brain._enable_voice_agent is False
    asyncio.run(brain._register_default_agents())
    assert "VoiceAgent" not in brain._agents


def test_cloud_stt_transcriber_refuses_without_opt_in(monkeypatch):
    stt = _real_stt()
    monkeypatch.setitem(sys.modules, "speech_recognition", MagicMock())   # even if the package is installed
    logs = _capture(stt.logger)
    assert stt.SpeechRecognitionTranscriber().initialize() is False
    assert "allow_cloud_stt" in logs.text


def test_cloud_stt_transcriber_works_when_explicitly_allowed(monkeypatch):
    SpeechRecognitionTranscriber = _real_stt().SpeechRecognitionTranscriber
    fake_sr = MagicMock()
    monkeypatch.setitem(sys.modules, "speech_recognition", fake_sr)
    assert SpeechRecognitionTranscriber(allow_cloud=True).initialize() is True


def test_cloud_speech_is_off_in_the_shipped_config():
    from config.settings import load_config_dict

    assert load_config_dict()["voice"]["recognition"]["allow_cloud_stt"] is False


def _real_stt():
    """tests/conftest.py replaces the audio/STT classes in utils.stt with dummies;
    load a pristine copy of the module to exercise the real gating code."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("utils_stt_pristine", REPO / "utils" / "stt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Records(logging.Handler):
    """Collects log records straight from a logger (this repo's loggers do not
    always propagate to pytest's caplog, which made caplog order-dependent)."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def text(self) -> str:
        return " ".join(r.getMessage() for r in self.records)


def _capture(logger: logging.Logger) -> _Records:
    handler = _Records()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return handler


def _spy(agent) -> MagicMock:
    """Swap the agent's (wrapper) logger for a mock so the calls can be asserted."""
    agent._logger = MagicMock()
    return agent._logger


def _said(mock: MagicMock, level: str) -> str:
    return " ".join(str(c.args[0]) for c in getattr(mock, level).call_args_list if c.args)


def _legacy_agent(config: Dict[str, Any]):
    from agents.voice_agent import VoiceAgent

    return VoiceAgent(event_bus=EventBus(), config=config)


@pytest.mark.asyncio
async def test_missing_vosk_model_fails_loudly_and_never_listens_without_a_wake_word(tmp_path):
    agent = _legacy_agent({"voice": {"vosk": {"model_path": str(tmp_path / "no-such-model")},
                                     "recognition": {"allow_cloud_stt": False}}})
    log = _spy(agent)
    await agent._initialize_wake_word_detector()
    assert agent._wake_word_detector is None
    assert "no-such-model" in agent._legacy_block_reason
    assert "Legacy voice input cannot start" in _said(log, "error")      # ERROR, not a quiet warning
    assert "vesper up" in _said(log, "error")

    await agent._initialize_components()
    assert agent._transcriber is None                   # the transcriber is never even built
    assert getattr(agent, "_mic_stream", None) is None  # and the microphone is never opened


@pytest.mark.asyncio
async def test_cloud_fallback_is_never_constructed_without_opt_in(monkeypatch):
    import utils.stt as stt

    constructed: List[Any] = []

    class Boom(stt.SpeechRecognitionTranscriber):
        def __init__(self, *a, **k):
            constructed.append((a, k))
            super().__init__(*a, **k)

    monkeypatch.setattr(stt, "SpeechRecognitionTranscriber", Boom)
    monkeypatch.setattr(stt.WhisperTranscriber, "initialize", lambda self: False)
    agent = _legacy_agent({"voice": {"recognition": {"allow_cloud_stt": False}}})
    log = _spy(agent)
    await agent._initialize_transcriber()
    assert agent._transcriber is None and constructed == []
    assert "cloud speech recognition is disabled" in _said(log, "error")


@pytest.mark.asyncio
async def test_cloud_fallback_requires_the_flag_and_says_audio_leaves(monkeypatch):
    import utils.stt as stt

    real = _real_stt()
    monkeypatch.setitem(sys.modules, "speech_recognition", MagicMock())
    monkeypatch.setattr(stt, "SpeechRecognitionTranscriber", real.SpeechRecognitionTranscriber)
    monkeypatch.setattr(stt.WhisperTranscriber, "initialize", lambda self: False)
    agent = _legacy_agent({"voice": {"recognition": {"allow_cloud_stt": True}}})
    log = _spy(agent)
    await agent._initialize_transcriber()
    assert isinstance(agent._transcriber, real.SpeechRecognitionTranscriber)
    assert "IS sent to Google" in _said(log, "warning")


# --------------------------------------------------------- voice input warm-up

def test_pipeline_warm_up_fails_fast_when_the_wake_model_cannot_load(monkeypatch):
    from voice.input.config import VoiceInputConfig
    from voice.input.service import build_pipeline
    from voice.input.stages import VoiceInputUnavailable

    pipeline = build_pipeline(VoiceInputConfig(), sink=MagicMock())
    monkeypatch.setitem(sys.modules, "openwakeword", None)      # import raises ImportError
    with pytest.raises(VoiceInputUnavailable):
        pipeline.warm_up()
