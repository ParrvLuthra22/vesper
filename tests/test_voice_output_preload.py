"""Kokoro preload: pay the first-use cost at startup so a failure surfaces at boot
(where the launcher reports it), not as a lost first reply."""

from __future__ import annotations

import logging
from typing import List
from unittest.mock import MagicMock

import pytest

from voice.output import service as vo_service
from voice.output.config import VoiceOutputConfig
from voice.output.speaker import Speaker, VoiceOutputService
from voice.output.tts import KokoroTTS


def test_warm_up_loads_and_runs_one_silent_synthesis(monkeypatch):
    tts = KokoroTTS(VoiceOutputConfig())
    calls: List[str] = []
    monkeypatch.setattr(tts, "load", lambda: calls.append("load"))
    monkeypatch.setattr(tts, "synthesize", lambda text: calls.append(f"synth:{text}") or (None, 24000))
    tts.warm_up()
    assert calls == ["load", "synth:Ready."]


def test_speaker_and_service_preload_delegate_to_the_tts():
    tts = MagicMock()
    VoiceOutputService(VoiceOutputConfig(), speaker=Speaker(VoiceOutputConfig(), tts=tts, player=MagicMock())).preload()
    tts.warm_up.assert_called_once()


def test_preload_is_on_by_default_and_configurable():
    assert VoiceOutputConfig().preload is True
    assert VoiceOutputConfig.from_app_config({"voice": {"output": {"preload": False}}}).preload is False
    from config.settings import load_config_dict

    assert load_config_dict()["voice"]["output"]["preload"] is True      # not silently dropped by the loader


def _patch_main(monkeypatch, preload: bool, preload_error: Exception = None):
    cfg = VoiceOutputConfig(enabled=True, preload=preload)
    monkeypatch.setattr(vo_service, "load_config", lambda: cfg)
    svc = MagicMock()
    if preload_error:
        svc.preload.side_effect = preload_error
    monkeypatch.setattr(vo_service, "VoiceOutputService", lambda config: svc)
    ran = []
    monkeypatch.setattr(vo_service.asyncio, "run", lambda coro: (ran.append(coro), coro.close()))
    return svc, ran


def test_main_preloads_before_connecting_to_the_gateway(monkeypatch):
    order: List[str] = []
    svc, ran = _patch_main(monkeypatch, preload=True)
    svc.preload.side_effect = lambda: order.append("preload")
    monkeypatch.setattr(vo_service.asyncio, "run", lambda coro: (order.append("connect"), coro.close()))
    assert vo_service.main() == 0
    assert order == ["preload", "connect"]


def test_main_exits_unavailable_and_never_connects_when_kokoro_cannot_load(monkeypatch, caplog):
    svc, ran = _patch_main(monkeypatch, preload=True, preload_error=FileNotFoundError("kokoro-v1.0.onnx"))
    with caplog.at_level(logging.ERROR):
        rc = vo_service.main()
    assert rc == vo_service.EXIT_UNAVAILABLE == 69
    assert ran == []                                                      # never reached the gateway loop
    assert "Kokoro failed to load" in caplog.text and "kokoro-v1.0.onnx" in caplog.text


def test_main_skips_preload_when_disabled(monkeypatch):
    svc, ran = _patch_main(monkeypatch, preload=False)
    assert vo_service.main() == 0
    svc.preload.assert_not_called()
    assert len(ran) == 1


def test_launcher_treats_voice_output_69_as_a_permanent_failure(tmp_path):
    from launcher.stack import VOICE_INPUT_UNAVAILABLE, build_specs

    specs = {s.name: s for s in build_specs({"gateway": {}, "launcher": {"components": {"hud": False}}}, "t",
                                            python="/py", root=tmp_path)}
    assert specs["voice_output"].permanent_exit_codes == frozenset({VOICE_INPUT_UNAVAILABLE}) == frozenset({69})
