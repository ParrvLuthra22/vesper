"""voice input's startup: it resolves the microphone EXPLICITLY, logs its name/index/rate, exits 69 (permanent failure)
with the list of devices that WERE found if the built-in mic is missing or cannot be opened, and never falls back to the
default. Fake device lists and fake pipelines only; no sounddevice call, no stream."""

from __future__ import annotations

import logging

import pytest

from voice.input import service
from voice.input.config import VoiceInputConfig
from voice.input.devices import InputDevice

BUILTIN = InputDevice(2, "MacBook Pro Microphone", 1, 48000.0)
AIRPODS = InputDevice(0, "Parrv's AirPods Pro", 1, 16000.0)
BLACKHOLE = InputDevice(1, "BlackHole 2ch", 1, 44100.0)


class FakePipeline:
    disabled_reason = None

    def __init__(self):
        self.ran_with = None

    def warm_up(self):
        pass

    def run(self, source):
        self.ran_with = source

    def stop(self):
        pass


@pytest.fixture
def wired(monkeypatch):
    """voice input enabled, a fake device list, a fake pipeline, and a recorder for the mic source it would build."""
    cfg = VoiceInputConfig(enabled=True)
    monkeypatch.setattr(service, "load_config", lambda: cfg)
    state = {"devices": [AIRPODS, BLACKHOLE, BUILTIN], "default": 0, "sources": [], "pipeline": FakePipeline()}
    monkeypatch.setattr(service, "list_input_devices", lambda query=None: list(state["devices"]))
    monkeypatch.setattr(service, "read_transport_types", lambda: {})
    monkeypatch.setattr(service, "default_input_index", lambda sd=None: state["default"])
    monkeypatch.setattr(service, "build_pipeline", lambda config, **k: state["pipeline"])

    class RecordingSource:
        resampled, open_rate = False, 48000

        def __init__(self, *args, **kwargs):
            state["sources"].append(args)

        def close(self):
            pass

    monkeypatch.setattr(service, "_AnnouncingMicSource", RecordingSource)
    state["cfg"] = cfg
    return state


def test_it_listens_to_the_builtin_mic_even_though_airpods_are_the_system_default(wired, caplog):
    caplog.set_level(logging.INFO, logger="vesper.voice")
    assert service.main() == 0
    (args,) = wired["sources"]
    assert args[2] == BUILTIN.index                                   # the explicitly resolved index, not None / default
    startup = [r.getMessage() for r in caplog.records if "Voice input starting" in r.getMessage()]
    assert len(startup) == 1
    line = startup[0]
    assert "MacBook Pro Microphone" in line and "index 2" in line and "48000 Hz" in line
    assert "AirPods" not in line and "BlackHole" not in line          # only the chosen device's name is logged


def test_a_virtual_default_device_does_not_matter_either(wired):
    wired["default"] = BLACKHOLE.index
    assert service.main() == 0 and wired["sources"][0][2] == BUILTIN.index


def test_builtin_missing_exits_69_lists_what_was_found_and_never_builds_a_source(wired, caplog):
    wired["devices"] = [AIRPODS, BLACKHOLE]
    caplog.set_level(logging.INFO, logger="vesper.voice")
    assert service.main() == service.EXIT_UNAVAILABLE == 69
    assert wired["sources"] == []                                      # no fallback to the default device: nothing was opened
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "built-in microphone was not found" in text.lower() and "will not fall back" in text
    assert "Parrv's AirPods Pro" in text and "BlackHole 2ch" in text


def test_an_explicit_device_that_does_not_exist_exits_69_too(wired, caplog):
    wired["cfg"].device = "Studio Display Microphone"
    caplog.set_level(logging.INFO, logger="vesper.voice")
    assert service.main() == 69 and wired["sources"] == []
    assert "does not match any input device" in " ".join(r.getMessage() for r in caplog.records)


def test_the_device_list_failing_to_load_exits_69(wired, monkeypatch, caplog):
    def boom(query=None):
        raise OSError("PortAudio error -9999")
    monkeypatch.setattr(service, "list_input_devices", boom)
    caplog.set_level(logging.INFO, logger="vesper.voice")
    assert service.main() == 69 and wired["sources"] == []
    assert "cannot list audio input devices" in " ".join(r.getMessage() for r in caplog.records)


def test_the_builtin_mic_that_cannot_be_opened_exits_69_and_says_which_devices_were_found(wired, caplog):
    class Failing(FakePipeline):
        def run(self, source):
            raise OSError("Error opening InputStream: Internal PortAudio error")
    wired["pipeline"] = Failing()
    caplog.set_level(logging.INFO, logger="vesper.voice")
    assert service.main() == 69
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "could not open 'MacBook Pro Microphone'" in text
    assert "Input devices found:" in text and "Parrv's AirPods Pro" in text and "BlackHole 2ch" in text


def test_disabled_voice_input_never_looks_at_audio_devices(wired, monkeypatch):
    wired["cfg"].enabled = False
    monkeypatch.setattr(service, "list_input_devices", lambda query=None: (_ for _ in ()).throw(AssertionError("no audio calls")))
    assert service.main() == 0


def test_the_ready_marker_line_is_unchanged_for_the_launcher():
    assert service.READY_MARKER == "voice input ready" and service.EXIT_UNAVAILABLE == 69
