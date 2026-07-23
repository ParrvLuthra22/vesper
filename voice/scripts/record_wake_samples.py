#!/usr/bin/env python3
"""Record ~30 samples of YOUR real voice saying the wake phrase.

These real recordings are mixed with synthetic (Kokoro) positives during
training so the model generalizes to your actual voice + mic. Negatives are a
SEPARATE dataset of clearly different phrases (see voice/TRAINING.md) — do NOT
record similar-sounding phrases here.

    python voice/scripts/record_wake_samples.py --phrase "wake up daddy's home" -n 30

Each prompt: press Enter, say the phrase once, it auto-stops on silence.
"""
from __future__ import annotations

import argparse
import os
import wave

import numpy as np

SAMPLE_RATE = 16000


def record_one(sd, max_seconds: float, silence_rms: float, min_seconds: float) -> np.ndarray:
    frame = int(SAMPLE_RATE * 0.03)
    chunks, started, trailing = [], False, 0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=frame) as stream:
        for _ in range(int(max_seconds / 0.03)):
            data, _ = stream.read(frame)
            arr = np.frombuffer(bytes(data), dtype=np.int16)
            chunks.append(arr)
            rms = float(np.sqrt(np.mean((arr.astype(np.float32)) ** 2)) + 1e-9)
            if rms > silence_rms:
                started, trailing = True, 0
            elif started:
                trailing += 1
            if started and trailing > int(0.6 / 0.03) and len(chunks) * 0.03 > min_seconds:
                break
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


def save_wav(path: str, audio: np.ndarray) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", required=True)
    ap.add_argument("-n", "--count", type=int, default=30)
    ap.add_argument("--out", default="voice/training_data/positives_real")
    ap.add_argument("--silence-rms", type=float, default=300.0)
    args = ap.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is required: pip install -r voice/requirements.txt")
        return 1

    os.makedirs(args.out, exist_ok=True)
    print(f'Recording {args.count} samples of: "{args.phrase}"')
    print("Vary your tone/distance/speed a little between takes for robustness.\n")
    for i in range(args.count):
        input(f"[{i + 1}/{args.count}] Press Enter, then say the phrase… ")
        audio = record_one(sd, max_seconds=4.0, silence_rms=args.silence_rms, min_seconds=0.4)
        path = os.path.join(args.out, f"sample_{i:03d}.wav")
        save_wav(path, audio)
        print(f"  saved {path} ({len(audio) / SAMPLE_RATE:.1f}s)")
    print(f"\nDone. {args.count} samples in {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
