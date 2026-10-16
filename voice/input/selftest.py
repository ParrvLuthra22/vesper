"""`vesper mic-test --selftest`: does the PIPELINE detect "Hey Jarvis" — with no microphone involved?

The phrase is synthesized with the local Kokoro TTS (several voices), converted to the device's native rate (48 kHz by
default), scaled to three levels, and fed through the SAME code that live mic frames take:

    device-rate int16 blocks -> StreamResampler -> 480-sample 16 kHz frames -> WakeDetector.feed (1280-sample windows) -> openWakeWord

A "direct" control run feeds the same audio straight at 16 kHz, skipping the resampler, to tell a resampling problem from a
model/voice problem. Nothing is played, recorded or saved; no audio stream is opened; nothing is sent anywhere.

The level of "full" is the phrase normalised to a peak of -1 dBFS, which is louder than live speech at a laptop; 0.3x and
0.1x are 10 dB and 20 dB lower. Compare the third with the peak that `vesper mic-test` shows for your real voice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PHRASE = "Hey Jarvis."
LEVELS: Tuple[Tuple[str, float], ...] = (("full", 1.0), ("0.3x", 0.3), ("0.1x", 0.1))
PEAK_FULL = 0.89                                   # about -1 dBFS
LEAD_S = TAIL_S = 1.0
VERDICT_OK = "pipeline OK (synthetic phrase detected)"
VERDICT_SUSPECT = "pipeline suspect (synthetic phrase not detected)"


def _dbfs(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-9 else -99.0


def resample_float(samples: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Offline high-quality resample of a float signal (the synthetic source; not the live path under test)."""
    if src == dst:
        return np.asarray(samples, dtype=np.float64)
    from scipy import signal

    f = Fraction(dst, src).limit_denominator(1000)
    return signal.resample_poly(np.asarray(samples, dtype=np.float64), f.numerator, f.denominator)


def prepare_signal(samples: np.ndarray, src_rate: int, device_rate: int, level: float) -> np.ndarray:
    """float TTS audio -> int16 at `device_rate`, normalised to PEAK_FULL * level, with a second of silence either side."""
    x = resample_float(samples, src_rate, device_rate)
    peak = float(np.max(np.abs(x))) or 1.0
    x = x / peak * PEAK_FULL * level
    pad = np.zeros(int(device_rate * LEAD_S))
    x = np.concatenate([pad, x, np.zeros(int(device_rate * TAIL_S))])
    return np.clip(np.rint(x * 32767.0), -32768, 32767).astype(np.int16)


def frames_via_live_path(device_audio: np.ndarray, device_rate: int, frame_samples: int = 480, target_rate: int = 16000):
    """Yield the 16 kHz int16 frames the wake word would get from a mic running at `device_rate`."""
    if device_rate == target_rate:
        for i in range(0, len(device_audio) - frame_samples + 1, frame_samples):
            yield device_audio[i:i + frame_samples]
        return
    from voice.input.resample import StreamResampler

    rs = StreamResampler(device_rate, target_rate, frame_samples)
    n = rs.block_in
    audio = np.concatenate([device_audio, np.zeros(n * (rs.latency_frames + 1), dtype=np.int16)])   # flush the look-ahead
    for i in range(0, len(audio) - n + 1, n):
        out = rs.push(audio[i:i + n])
        if out is not None:
            yield out


@dataclass
class Row:
    voice: str
    level: str
    path: str                 # "device-rate path" | "direct 16 kHz"
    peak_dbfs: float
    max_score: float


@dataclass
class SelftestResult:
    verdict: str
    rows: List[Row] = field(default_factory=list)
    diagnosis: str = ""
    best_full: float = 0.0
    best_direct_full: float = 0.0


def max_score_for(frames, detector) -> float:
    best = 0.0
    for fr in frames:
        scores = detector.feed(fr)
        if scores:
            best = max(best, max(scores))
    return best


def control_signals(device_rate: int, other_phrase: Optional[Callable[[], Tuple[np.ndarray, int]]] = None
                    ) -> Dict[str, np.ndarray]:
    """Things that must NOT wake it: digital silence, white noise at about -30 dBFS, and (if given) a different sentence."""
    rng = np.random.default_rng(7)
    secs = 3
    controls = {
        "silence": np.zeros(int(device_rate * secs), dtype=np.int16),
        "noise -30 dBFS": (rng.normal(0, 0.0316, int(device_rate * secs)) * 32767).clip(-32768, 32767).astype(np.int16),
    }
    if other_phrase is not None:
        samples, rate = other_phrase()
        controls["other phrase"] = prepare_signal(samples, rate, device_rate, 1.0)
    return controls


def run_selftest(voices: Dict[str, Callable[[], Tuple[np.ndarray, int]]], detector, *, device_rate: int = 48000,
                 threshold: float = 0.5, levels: Sequence[Tuple[str, float]] = LEVELS, frame_samples: int = 480,
                 out: Callable[[str], None] = print,
                 other_phrase: Optional[Callable[[], Tuple[np.ndarray, int]]] = None) -> SelftestResult:
    """`voices`: name -> synth() returning (float samples, rate). `detector`: has feed(int16 frame) -> [scores] and reset()."""
    rows: List[Row] = []
    for vname, synth in voices.items():
        samples, rate = synth()
        for label, level in levels:
            dev_audio = prepare_signal(samples, rate, device_rate, level)
            direct_audio = prepare_signal(samples, rate, 16000, level)
            for path, frames in (("device-rate path", frames_via_live_path(dev_audio, device_rate, frame_samples)),
                                 ("direct 16 kHz", frames_via_live_path(direct_audio, 16000, frame_samples))):
                detector.reset()
                score = max_score_for(frames, detector)
                peak = float(np.max(np.abs(dev_audio))) / 32768.0
                rows.append(Row(vname, label, path, _dbfs(peak), score))
                out(f"  voice {vname:<10} level {label:<5} ({_dbfs(peak):6.1f} dBFS peak)  {path:<16}  max wake score {score:5.3f}")
    fired_controls = []
    for cname, audio in control_signals(device_rate, other_phrase).items():
        detector.reset()
        score = max_score_for(frames_via_live_path(audio, device_rate, frame_samples), detector)
        out(f"  control {cname:<16} (must NOT wake)                       max wake score {score:5.3f}")
        if score >= threshold:
            fired_controls.append(cname)
    full_dev = [r.max_score for r in rows if r.level == levels[0][0] and r.path == "device-rate path"]
    full_dir = [r.max_score for r in rows if r.level == levels[0][0] and r.path == "direct 16 kHz"]
    best, best_dir = (max(full_dev) if full_dev else 0.0), (max(full_dir) if full_dir else 0.0)
    ok = best >= threshold
    if ok:
        diagnosis = ""
        if best_dir < threshold:
            diagnosis = "note: only the device-rate path detected it; the direct 16 kHz control did not."
    elif best_dir >= threshold:
        diagnosis = ("the direct 16 kHz audio IS detected but the device-rate path loses it: look at the resampler / frame "
                     "chunking.")
    else:
        diagnosis = ("not even the direct 16 kHz audio is detected: the fault is not in the mic/resample path (the model, the "
                     "synthetic voices, or the threshold), so a low live score does not point at a pipeline bug.")
    if fired_controls:
        ok = False
        diagnosis = (f"the controls {', '.join(fired_controls)} fired the wake word, so a high score proves nothing: the detector "
                     "is firing on things that are not the phrase.")
    verdict = VERDICT_OK if ok else VERDICT_SUSPECT
    out(f"verdict: {verdict}   (best full-level score {best:.3f} via the device-rate path, {best_dir:.3f} direct; "
        f"threshold {threshold:.2f})")
    if diagnosis:
        out(diagnosis)
    return SelftestResult(verdict, rows, diagnosis, best, best_dir)
