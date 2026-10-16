"""Device pinning: voice input uses the microphone it was told to use, never "whatever macOS has as default".
Fake device lists only — no sounddevice call, no CoreAudio call, no stream is opened."""

from __future__ import annotations

import sys

import pytest

from voice.input import devices as dev
from voice.input.config import VoiceInputConfig
from voice.input.devices import (
    DeviceResolutionError, InputDevice, default_input_index, find_builtin, list_input_devices, looks_builtin,
    read_transport_types, resolve_input_device)


def D(i, name, rate=48000.0):
    return InputDevice(index=i, name=name, max_input_channels=1, default_samplerate=rate)


BUILTIN = D(2, "MacBook Pro Microphone")
AIRPODS = D(0, "Parrv's AirPods Pro")
BLACKHOLE = D(1, "BlackHole 2ch", 44100.0)
USB = D(3, "Yeti Stereo Microphone", 48000.0)


# ------------------------------------------------------------------ "builtin"

def test_the_builtin_mic_is_chosen_even_when_airpods_are_the_system_default():
    devices = [AIRPODS, BLACKHOLE, BUILTIN, USB]
    chosen = resolve_input_device("builtin", devices, default_index=AIRPODS.index)
    assert chosen == BUILTIN


def test_the_builtin_mic_is_chosen_even_when_a_virtual_device_is_the_default():
    devices = [BLACKHOLE, BUILTIN]
    assert resolve_input_device("builtin", devices, default_index=BLACKHOLE.index) == BUILTIN


@pytest.mark.parametrize("name", [
    "MacBook Pro Microphone", "MacBook Air Microphone", "Built-in Microphone", "MacBook Microphone", "Internal Microphone",
    # localized
    "MacBook Pro-Mikrofon", "Eingebautes Mikrofon", "Microphone MacBook Pro", "Microphone intégré",
    "Micrófono del MacBook Pro", "Micrófono integrado", "Microfono di MacBook Pro", "Microfono integrato",
    "Microfone do MacBook Pro", "Microfoon van MacBook Pro", "Ingebouwde microfoon", "MacBook Proのマイク", "内蔵マイク",
    "MacBook Pro麦克风", "内置麦克风", "MacBook Pro 麥克風", "內建麥克風", "MacBook Pro 마이크", "내장 마이크",
    "Микрофон MacBook Pro", "Встроенный микрофон",
])
def test_localized_builtin_names_resolve(name):
    chosen = resolve_input_device("builtin", [AIRPODS, D(7, name), USB], default_index=0)
    assert chosen.name == name


@pytest.mark.parametrize("name", [
    "Parrv's AirPods Pro", "Parrv's iPhone Microphone", "BlackHole 2ch", "Zoom Audio Device", "Microsoft Teams Audio",
    "Loopback Audio", "Aggregate Device", "Built-in Output", "USB Microphone", "Yeti Stereo Microphone",
    "MacBook Pro Speakers", "External Microphone", "Built-in Microphone (Virtual)", "Logitech Webcam C920",
])
def test_things_that_are_not_the_laptops_own_mic_never_match_by_name(name):
    assert looks_builtin(name) is False


def test_builtin_missing_fails_and_lists_the_devices_that_were_found_and_never_falls_back():
    devices = [AIRPODS, BLACKHOLE, USB]
    with pytest.raises(DeviceResolutionError) as exc:
        resolve_input_device("builtin", devices, default_index=AIRPODS.index)       # the default exists — still no fallback
    msg = str(exc.value)
    assert "built-in microphone was not found" in msg and "will not fall back" in msg
    assert exc.value.found == ["Parrv's AirPods Pro", "BlackHole 2ch", "Yeti Stereo Microphone"]
    for name in exc.value.found:
        assert name in msg


def test_no_input_devices_at_all():
    with pytest.raises(DeviceResolutionError) as exc:
        resolve_input_device("builtin", [], default_index=None)
    assert "Input devices found: none" in str(exc.value)


def test_the_unset_setting_means_builtin_not_the_system_default():
    assert resolve_input_device(None, [AIRPODS, BUILTIN], default_index=0) == BUILTIN


# ------------------------------------------------------------------ CoreAudio transport type (preferred when readable)

def test_transport_type_finds_a_builtin_mic_with_a_name_nobody_anticipated():
    odd = D(5, "Mikrofon (laptop)")                       # no known name at all
    chosen = resolve_input_device("builtin", [AIRPODS, odd], transports={"Mikrofon (laptop)": "built-in",
                                                                         "Parrv's AirPods Pro": "bluetooth"})
    assert chosen == odd


def test_a_name_that_looks_builtin_is_rejected_when_coreaudio_says_it_is_not():
    impostor = D(6, "MacBook Pro Microphone")             # e.g. a USB/virtual device that copies the name
    with pytest.raises(DeviceResolutionError):
        resolve_input_device("builtin", [impostor], transports={"MacBook Pro Microphone": "usb"})


def test_transport_type_plus_known_name_beats_name_alone():
    a, b = D(1, "Built-in Microphone"), D(4, "MacBook Pro Microphone")
    assert find_builtin([a, b], transports={"MacBook Pro Microphone": "built-in"}) == b


def test_read_transport_types_returns_empty_off_macos_and_never_raises(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert read_transport_types() == {}
    monkeypatch.setattr(sys, "platform", "darwin")
    import ctypes

    def boom(*a, **k):
        raise OSError("no CoreAudio here")
    monkeypatch.setattr(ctypes, "CDLL", boom)             # never touches the real framework
    assert read_transport_types() == {}


# ------------------------------------------------------------------ explicit overrides

def test_exact_and_partial_name_overrides():
    devices = [AIRPODS, BUILTIN, USB]
    assert resolve_input_device("Yeti Stereo Microphone", devices) == USB
    assert resolve_input_device("yeti", devices) == USB                # unique partial, case-insensitive
    with pytest.raises(DeviceResolutionError, match="more than one"):
        resolve_input_device("microphone", devices)                    # matches the built-in AND the Yeti
    with pytest.raises(DeviceResolutionError, match="does not match") as exc:
        resolve_input_device("Studio Display Microphone", devices)
    assert "MacBook Pro Microphone" in str(exc.value)                 # and says what WAS found


def test_index_override():
    devices = [AIRPODS, BUILTIN, USB]
    assert resolve_input_device(3, devices) == USB
    assert resolve_input_device("3", devices) == USB
    with pytest.raises(DeviceResolutionError, match="index 9"):
        resolve_input_device(9, devices)


def test_default_is_only_used_when_asked_for_by_name():
    devices = [AIRPODS, BUILTIN]
    assert resolve_input_device("default", devices, default_index=0) == AIRPODS
    with pytest.raises(DeviceResolutionError, match="no default input"):
        resolve_input_device("default", devices, default_index=None)


# ------------------------------------------------------------------ listing

def test_list_input_devices_keeps_only_input_capable_devices_and_names():
    fake = lambda: [{"name": "MacBook Pro Speakers", "max_input_channels": 0, "index": 0},
                    {"name": "MacBook Pro Microphone", "max_input_channels": 1, "index": 1, "default_samplerate": 48000.0},
                    {"name": "AirPods", "max_input_channels": 1, "index": 2, "default_samplerate": 16000.0}]
    got = list_input_devices(fake)
    assert [(d.index, d.name) for d in got] == [(1, "MacBook Pro Microphone"), (2, "AirPods")]
    assert str(got[0]) == "MacBook Pro Microphone"                    # str() is the name and nothing else


def test_default_input_index_is_read_only_and_tolerant():
    class SD:
        class default:
            device = (4, 7)
    assert default_input_index(SD) == 4
    class Broken:
        class default:
            device = None
    assert default_input_index(Broken) is None


# ------------------------------------------------------------------ configuration

def test_default_device_is_builtin_and_the_threshold_is_unchanged():
    cfg = VoiceInputConfig.from_app_config({})
    assert cfg.device == "builtin" and cfg.wake_threshold == 0.5


def test_device_override_and_legacy_alias():
    assert VoiceInputConfig.from_app_config({"voice": {"input": {"device": "Yeti"}}}).device == "Yeti"
    assert VoiceInputConfig.from_app_config({"voice": {"input": {"device": 4}}}).device == 4
    assert VoiceInputConfig.from_app_config({"voice": {"input": {"mic_device": "Yeti"}}}).device == "Yeti"      # legacy alias
    both = VoiceInputConfig.from_app_config({"voice": {"input": {"device": "builtin", "mic_device": "Yeti"}}})
    assert both.device == "builtin"                                    # the new key wins


def test_voice_wake_threshold_is_the_knob_and_the_old_key_still_works():
    assert VoiceInputConfig.from_app_config({"voice": {"wake": {"threshold": 0.35}}}).wake_threshold == 0.35
    assert VoiceInputConfig.from_app_config({"voice": {"input": {"wake_threshold": 0.7}}}).wake_threshold == 0.7
    both = VoiceInputConfig.from_app_config({"voice": {"wake": {"threshold": 0.35}, "input": {"wake_threshold": 0.7}}})
    assert both.wake_threshold == 0.35


def test_shipped_settings_pin_the_builtin_mic_and_keep_the_default_threshold():
    from config.settings import load_config_dict

    raw = load_config_dict()
    cfg = VoiceInputConfig.from_app_config(raw)
    assert cfg.device == "builtin" and cfg.mic_device is None
    assert cfg.wake_threshold == 0.5                                   # not changed without a measured reason
    assert raw["voice"]["wake"]["threshold"] == 0.5 and raw["voice"]["input"]["device"] == "builtin"
