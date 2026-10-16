"""`vesper mic-test` — what does the microphone voice input would use actually hear, and does the wake word notice?

Opens the PINNED input device (voice.input.device, the same resolution voice input itself uses) for 10 seconds and prints,
every 0.5 s: a level bar with RMS / peak, the wake-word model's score, and the maximum score seen so far. Then a verdict:

    no signal                                   the peak never rose above the noise floor — wrong/blocked/muted mic
    signal but wake word never above threshold  the mic works; the wake word did not score high enough (say "Hey Jarvis")
    wake word fired                             the wake score reached voice.wake.threshold

It does NOT record or save audio, does NOT run speech-to-text, and sends nothing anywhere: the only things it reads are
audio frames (held for one frame at a time) and the wake model's score. Only device NAMES are printed.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

#: Peak (0..1 of full scale) below which the input is treated as "no signal". A working laptop mic in a quiet room sits
#: well above this (-54 dBFS); digital silence or a muted/blocked device sits at ~0.
NO_SIGNAL_PEAK = 0.002
VERDICT_NO_SIGNAL = "no signal"
VERDICT_QUIET = "signal but wake word never above threshold"
VERDICT_FIRED = "wake word fired"
REPORT_EVERY_MS = 500


def to_dbfs(level: float) -> float:
    return 20.0 * math.log10(level) if level > 1e-9 else -99.0


def meter_bar(level: float, width: int = 20) -> str:
    """A bar for a 0..1 level on a -60..0 dBFS scale."""
    frac = max(0.0, min(1.0, (to_dbfs(level) + 60.0) / 60.0))
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


@dataclass
class MicTestResult:
    device: str
    seconds: float
    peak: float
    max_rms: float
    max_score: float
    threshold: float
    verdict: str
    lines: List[str] = field(default_factory=list)


def verdict_for(peak: float, max_score: float, threshold: float) -> str:
    if max_score >= threshold:
        return VERDICT_FIRED
    if peak < NO_SIGNAL_PEAK:
        return VERDICT_NO_SIGNAL
    return VERDICT_QUIET


def run_mic_test(frames, wake_feed: Callable[[np.ndarray], Sequence[float]], *, threshold: float, device: str = "",
                 seconds: float = 10.0, frame_ms: int = 30, out: Callable[[str], None] = print) -> MicTestResult:
    """Consume `frames` (an iterator of int16 mono frames) for `seconds` of AUDIO TIME (frames x frame_ms, so tests need no
    sleeping). Nothing is stored beyond the running maxima."""
    lines: List[str] = []

    def emit(text: str) -> None:
        lines.append(text)
        out(text)

    elapsed_ms = 0.0
    next_report = float(REPORT_EVERY_MS)
    peak = max_rms = max_score = 0.0
    win_peak = win_sq = 0.0
    win_n = 0
    win_score = 0.0
    for frame in frames:
        x = np.asarray(frame, dtype=np.float32) / 32768.0
        if len(x):
            fp = float(np.max(np.abs(x)))
            peak, win_peak = max(peak, fp), max(win_peak, fp)
            win_sq += float(np.sum(x * x))
            win_n += len(x)
        scores = list(wake_feed(frame))
        if scores:
            win_score = max(win_score, max(scores))
            max_score = max(max_score, max(scores))
        elapsed_ms += frame_ms
        if elapsed_ms >= next_report:
            rms = math.sqrt(win_sq / win_n) if win_n else 0.0
            max_rms = max(max_rms, rms)
            emit(f"{elapsed_ms / 1000.0:5.1f}s {meter_bar(rms)} rms {rms:6.4f} ({to_dbfs(rms):5.1f} dBFS)  "
                 f"peak {win_peak:6.4f}  wake {win_score:5.3f}  max {max_score:5.3f}")
            win_peak = win_sq = 0.0
            win_n = 0
            win_score = 0.0
            next_report += REPORT_EVERY_MS
        if elapsed_ms >= seconds * 1000.0:
            break
    verdict = verdict_for(peak, max_score, threshold)
    emit(f"verdict: {verdict}   (peak {peak:.4f} = {to_dbfs(peak):.1f} dBFS, max wake score {max_score:.3f}, "
         f"threshold {threshold:.2f})")
    return MicTestResult(device, elapsed_ms / 1000.0, peak, max_rms, max_score, threshold, verdict, lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="vesper mic-test", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to listen (default 10)")
    args = ap.parse_args(list(argv) if argv is not None else sys.argv[1:])

    from voice.input.audio import MicSource
    from voice.input.devices import DeviceResolutionError
    from voice.input.service import EXIT_UNAVAILABLE, load_config, resolve_device
    from voice.input.stages import VoiceInputUnavailable, WakeDetector

    config = load_config()
    try:
        device = resolve_device(config)
    except DeviceResolutionError as exc:
        print(f"Voice input would be unavailable: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot list audio input devices: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    print(f"device : {device.name} (index {device.index}, native {int(device.default_samplerate)} Hz, device setting "
          f"{config.device!r})")
    print(f"nothing is recorded or saved; no speech-to-text; nothing leaves this machine. Say \"Hey Jarvis\"…  "
          f"({args.seconds:g} s)")

    wake = WakeDetector(config)
    try:
        wake.load()
    except VoiceInputUnavailable as exc:
        print(f"wake model unavailable: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    source = MicSource(config.sample_rate, config.frame_samples, device.index)
    try:
        result = run_mic_test(source.frames(), wake.feed, threshold=config.wake_threshold, device=device.name,
                              seconds=args.seconds, frame_ms=config.frame_ms)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 — the device could not be opened/read
        print(f"could not read from '{device.name}': {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    finally:
        source.close()
    if source.resampled:
        print(f"(the device runs at {source.open_rate} Hz; audio was converted to {config.sample_rate} Hz for the wake model)")
    return 3 if result.verdict == VERDICT_NO_SIGNAL else 0


if __name__ == "__main__":
    sys.exit(main())
