"""
Which microphone does voice input use? Always the one it was told to — never "whatever macOS has as default today".

    voice.input.device: "builtin"        (default) the MacBook's own microphone
                        "MacBook Pro Microphone"   an exact (or unique partial) device name
                        2                 a sounddevice index
                        "default"         the system default input — only if you ask for it by name

Resolution is a pure function over a device list, so every case is testable with a fake list:

  * "builtin" picks the device whose CoreAudio transport type is built-in when that can be read, and/or whose NAME is
    a known built-in microphone name ("MacBook Pro Microphone", "Built-in Microphone", and the localized variants).
    Names that are obviously not the laptop's own mic (AirPods, an iPhone, BlackHole, Zoom, aggregate/virtual devices) are
    never matched by name.
  * If nothing matches — or the explicit name/index does not exist — resolution FAILS with the list of input devices that
    WERE found (names only). There is no silent fallback to the default device.

Only device NAMES are ever logged or printed; nothing here opens a stream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

#: A device name is the laptop's own microphone if it says MACBOOK or BUILT-IN/INTERNAL *and* names a microphone, in any of
#: the languages macOS localizes device names into ("MacBook Pro Microphone", "Built-in Microphone", "Micrófono del MacBook
#: Pro", "Microphone intégré", "MacBook Pro-Mikrofon", "内蔵マイク", "Встроенный микрофон" ...).
_MIC_WORDS = ("microphone", "micro", "mikro", "microfon", "microfoon", "microfone", "mic", "マイク", "麦克风", "麥克風",
              "마이크", "микрофон")
_MACBOOK_WORDS = ("macbook",)
_BUILTIN_WORDS = ("built-in", "built in", "builtin", "internal", "integrad", "integrat", "intégr", "integr", "eingebaut",
                  "ingebouwd", "inbyggd", "indbygget", "wbudowan", "встроен", "内蔵", "内置", "內建", "内建", "내장")
#: Never a built-in microphone, whatever else the name says.
_NOT_BUILTIN = re.compile(
    r"(?i)airpods|bluetooth|iphone|ipad|watch|blackhole|soundflower|loopback|zoom|teams|virtual|aggregate|multi-?output|"
    r"vb-?cable|obs|usb|display|monitor|webcam|camera|headset|headphone|yeti|scarlett|focusrite|jabra|bose|sony|wh-|wf-")

BUILTIN_SETTINGS = {"builtin", "built-in", "built_in", "internal", "macbook"}
DEFAULT_SETTINGS = {"default", "system", "system-default"}


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    max_input_channels: int = 1
    default_samplerate: float = 48000.0

    def __str__(self) -> str:                      # names only, by construction
        return self.name


class DeviceResolutionError(RuntimeError):
    """The pinned microphone cannot be found. `found` holds the NAMES of the input devices that WERE found."""

    def __init__(self, message: str, found: Sequence[str]):
        self.found = list(found)
        shown = ", ".join(f"'{n}'" for n in self.found) if self.found else "none"
        super().__init__(f"{message} Input devices found: {shown}.")


def list_input_devices(query: Optional[Callable[[], Any]] = None) -> List[InputDevice]:
    """Input-capable devices from sounddevice (or from `query`, a stand-in returning the same dict shape)."""
    if query is None:
        import sounddevice as sd

        query = sd.query_devices
    out: List[InputDevice] = []
    for i, d in enumerate(query()):
        if int(d.get("max_input_channels", 0)) > 0:
            out.append(InputDevice(index=int(d.get("index", i)), name=str(d.get("name", f"device {i}")),
                                   max_input_channels=int(d["max_input_channels"]),
                                   default_samplerate=float(d.get("default_samplerate", 48000.0))))
    return out


def looks_builtin(name: str) -> bool:
    low = name.lower()
    if _NOT_BUILTIN.search(name):
        return False
    has_mic = any(w in low for w in _MIC_WORDS)
    return has_mic and (any(w in low for w in _MACBOOK_WORDS) or any(w in low for w in _BUILTIN_WORDS))


def find_builtin(devices: Sequence[InputDevice], transports: Optional[Dict[str, str]] = None) -> Optional[InputDevice]:
    """The laptop's own microphone, preferring what CoreAudio says about the transport over what the name suggests."""
    transports = transports or {}
    scored = []
    for d in devices:
        by_transport = transports.get(d.name) == "built-in"
        by_name = looks_builtin(d.name)
        if by_transport and not _NOT_BUILTIN.search(d.name):
            scored.append((2 if by_name else 1, d))          # built-in transport (+ a known name is the strongest signal)
        elif by_name and (not transports or d.name not in transports or transports[d.name] == "built-in"):
            scored.append((1, d))
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], t[1].index))
    return scored[0][1]


def resolve_input_device(setting: Any, devices: Sequence[InputDevice], transports: Optional[Dict[str, str]] = None,
                         default_index: Optional[int] = None) -> InputDevice:
    """Turn the config value into ONE input device, or raise DeviceResolutionError naming what was found."""
    names = [d.name for d in devices]
    text = str(setting).strip() if setting is not None else "builtin"
    if setting is None or text.lower() in BUILTIN_SETTINGS:
        dev = find_builtin(devices, transports)
        if dev is None:
            raise DeviceResolutionError("The built-in microphone was not found, and voice input will not fall back to "
                                        "another device.", names)
        return dev
    if text.lower() in DEFAULT_SETTINGS:
        for d in devices:
            if default_index is not None and d.index == default_index:
                return d
        raise DeviceResolutionError("voice.input.device is 'default' but the system has no default input device.", names)
    if isinstance(setting, int) or re.fullmatch(r"\d+", text):
        idx = int(text)
        for d in devices:
            if d.index == idx:
                return d
        raise DeviceResolutionError(f"voice.input.device is index {idx}, which is not an input device.", names)
    exact = [d for d in devices if d.name.lower() == text.lower()]
    if len(exact) == 1:
        return exact[0]
    partial = [d for d in devices if text.lower() in d.name.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1 or len(exact) > 1:
        raise DeviceResolutionError(f"voice.input.device '{text}' matches more than one input device; use the full name "
                                    "or the index.", names)
    raise DeviceResolutionError(f"voice.input.device '{text}' does not match any input device.", names)


def default_input_index(sd_module: Any = None) -> Optional[int]:
    """Index of the system default INPUT device, or None. Read-only."""
    try:
        if sd_module is None:
            import sounddevice as sd_module  # type: ignore[no-redef]
        d = sd_module.default.device
        idx = d[0] if isinstance(d, (list, tuple)) else d
        return int(idx) if idx is not None and int(idx) >= 0 else None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- CoreAudio transport type (best effort)

def read_transport_types() -> Dict[str, str]:
    """{device name: "built-in" | "usb" | "bluetooth" | "virtual" | ...} straight from CoreAudio, or {} if it cannot be read
    (not macOS, framework missing, anything unexpected). Property reads only: no device is opened, nothing is changed.
    This is a PREFERENCE: when it returns {} the name match alone decides."""
    import sys

    if sys.platform != "darwin":
        return {}
    try:
        import ctypes
        import struct

        ca = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

        def fourcc(s: str) -> int:
            return struct.unpack(">I", s.encode("ascii"))[0]

        class Address(ctypes.Structure):
            _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]

        glob = fourcc("glob")
        ca.AudioObjectGetPropertyDataSize.argtypes = [ctypes.c_uint32, ctypes.POINTER(Address), ctypes.c_uint32,
                                                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        ca.AudioObjectGetPropertyData.argtypes = [ctypes.c_uint32, ctypes.POINTER(Address), ctypes.c_uint32,
                                                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFRelease.argtypes = [ctypes.c_void_p]

        addr = Address(fourcc("dev#"), glob, 0)
        size = ctypes.c_uint32(0)
        if ca.AudioObjectGetPropertyDataSize(1, ctypes.byref(addr), 0, None, ctypes.byref(size)) != 0:
            return {}
        n = size.value // 4
        ids = (ctypes.c_uint32 * n)()
        if ca.AudioObjectGetPropertyData(1, ctypes.byref(addr), 0, None, ctypes.byref(size), ids) != 0:
            return {}
        kinds = {"bltn": "built-in", "usb ": "usb", "blue": "bluetooth", "blea": "bluetooth", "virt": "virtual",
                 "grup": "aggregate", "airp": "airplay", "hdmi": "hdmi", "dprt": "displayport", "thun": "thunderbolt",
                 "1394": "firewire", "pci ": "pci"}
        out: Dict[str, str] = {}
        for dev_id in ids:
            t = ctypes.c_uint32(0)
            tsz = ctypes.c_uint32(4)
            ta = Address(fourcc("tran"), glob, 0)
            if ca.AudioObjectGetPropertyData(dev_id, ctypes.byref(ta), 0, None, ctypes.byref(tsz), ctypes.byref(t)) != 0:
                continue
            ref = ctypes.c_void_p(0)
            rsz = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
            na = Address(fourcc("lnam"), glob, 0)
            if ca.AudioObjectGetPropertyData(dev_id, ctypes.byref(na), 0, None, ctypes.byref(rsz), ctypes.byref(ref)) != 0 or not ref.value:
                continue
            buf = ctypes.create_string_buffer(256)
            ok = cf.CFStringGetCString(ref, buf, 256, 0x08000100)       # kCFStringEncodingUTF8
            cf.CFRelease(ref)
            if ok:
                code = struct.pack(">I", t.value).decode("latin-1")
                out[buf.value.decode("utf-8", errors="replace")] = kinds.get(code, code.strip() or "other")
        return out
    except Exception:  # noqa: BLE001 — a preference only; never let it break voice input
        return {}
