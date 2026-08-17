#!/usr/bin/env python3
"""Train a custom openWakeWord model (e.g. "wake up daddy's home") -> ONNX.

This is the 4-8 hour GPU job you run ONCE. It:
  1. synthesizes many positive clips of the phrase across Kokoro voices,
  2. mixes in your ~30 real recordings (voice/scripts/record_wake_samples.py),
  3. augments with room reverb + background noise,
  4. computes openWakeWord features and trains a small classifier,
  5. exports voice/models/<name>.onnx (then set voice.input.wake_model to it).

Requires the training extras (see voice/TRAINING.md):
    pip install openwakeword[training] kokoro scipy

Negatives are a SEPARATE dataset of clearly-different speech/noise (NOT
similar-sounding phrases) — pass --negatives-dir. Full guidance in TRAINING.md.

    python voice/scripts/train_wake_word.py \
        --phrase "wake up daddy's home" --name wake_up_daddys_home \
        --real-dir voice/training_data/positives_real \
        --negatives-dir ~/data/openwakeword_negatives \
        --background-dir ~/data/audioset_noise
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

SAMPLE_RATE = 16000


def _write_wav(path: str, audio_float, rate: int) -> None:
    import wave

    import numpy as np

    audio16 = np.clip(np.asarray(audio_float) * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio16.tobytes())


def _synthesize_kokoro_onnx(phrase: str, out_dir: str, n_per_voice: int) -> int:
    """
    Preferred synthesizer: the same kokoro-onnx + model files Vesper already
    uses for speech output (voice/models/). No extra dependency, no second
    model download, and it exposes all 54 voices rather than a hand-picked
    seven — more speaker variety is exactly what a wake model wants.
    """
    import random

    import numpy as np
    import scipy.signal as ss

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from voice.output.config import VoiceOutputConfig
    from voice.output.tts import KokoroTTS

    config = VoiceOutputConfig.from_app_config({"voice": {"output": {"enabled": True}}})
    tts = KokoroTTS(config)
    tts.load()

    voices = sorted(tts._kokoro.get_voices())
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for voice in voices:
        for i in range(n_per_voice):
            speed = random.uniform(0.85, 1.15)
            lang = "en-gb" if voice.startswith("b") else "en-us"
            audio, rate = tts._kokoro.create(phrase, voice=voice, speed=speed, lang=lang)
            audio = np.asarray(audio, dtype=np.float32)
            # openWakeWord features are computed at 16 kHz; Kokoro emits 24 kHz.
            if rate != SAMPLE_RATE:
                audio = ss.resample_poly(audio, SAMPLE_RATE, rate)
            _write_wav(os.path.join(out_dir, f"{voice}_{i:03d}.wav"), audio, SAMPLE_RATE)
            count += 1
    return count


def _synthesize_kokoro_torch(phrase: str, out_dir: str, n_per_voice: int) -> int:
    """Fallback: the PyTorch `kokoro` package (pip install kokoro)."""
    import random

    import numpy as np
    from kokoro import KPipeline

    os.makedirs(out_dir, exist_ok=True)
    pipe = KPipeline(lang_code="a")  # American English
    voices = ["af_heart", "af_bella", "af_nicole", "am_adam", "am_michael", "bf_emma", "bm_george"]
    count = 0
    for voice in voices:
        for i in range(n_per_voice):
            speed = random.uniform(0.85, 1.15)
            audio = np.concatenate([seg.audio for seg in pipe(phrase, voice=voice, speed=speed)])
            _write_wav(os.path.join(out_dir, f"{voice}_{i:03d}.wav"), audio, SAMPLE_RATE)
            count += 1
    return count


def synthesize_positives(phrase: str, out_dir: str, n_per_voice: int) -> int:
    """Generate synthetic positives across every Kokoro voice, with small
    speed jitter so the model doesn't overfit one delivery."""
    try:
        return _synthesize_kokoro_onnx(phrase, out_dir, n_per_voice)
    except Exception as exc:
        print(f"kokoro-onnx synthesis unavailable ({exc}); trying the PyTorch kokoro package")
        return _synthesize_kokoro_torch(phrase, out_dir, n_per_voice)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", required=True)
    ap.add_argument("--name", required=True, help="output model name -> voice/models/<name>.onnx")
    ap.add_argument("--real-dir", default="voice/training_data/positives_real")
    ap.add_argument("--negatives-dir", required=True, help="clearly-different speech/noise clips")
    ap.add_argument("--background-dir", required=True, help="background noise for augmentation")
    ap.add_argument("--work-dir", default="voice/training_data")
    ap.add_argument("--n-per-voice", type=int, default=400)
    ap.add_argument("--steps", type=int, default=50000)
    args = ap.parse_args()

    try:
        import openwakeword.train as owt  # noqa: F401
    except Exception:
        print("Training deps missing. See voice/TRAINING.md:\n"
              "  pip install openwakeword[training] kokoro scipy")
        return 1

    synth_dir = os.path.join(args.work_dir, "positives_synth")
    print(f"[1/5] Synthesizing positives for: {args.phrase!r}")
    n = synthesize_positives(args.phrase, synth_dir, args.n_per_voice)
    print(f"      {n} synthetic clips -> {synth_dir}")
    real_n = len([f for f in os.listdir(args.real_dir) if f.endswith(".wav")]) if os.path.isdir(args.real_dir) else 0
    print(f"[2/5] Real recordings: {real_n} in {args.real_dir}")

    # openWakeWord's training expects a config describing positive/negative/
    # background dirs, augmentation, feature extraction, and training steps. We
    # write it out and hand off to the library's trainer.
    config = {
        "model_name": args.name,
        "target_phrase": [args.phrase],
        "positive_dirs": [synth_dir, args.real_dir],
        "negative_dirs": [args.negatives_dir],
        "background_dirs": [args.background_dir],
        "augmentation": {"reverb": True, "background_snr_db": [5, 10, 20], "gain_db": [-6, 0, 6]},
        "steps": args.steps,
        "sample_rate": SAMPLE_RATE,
        "output_dir": args.work_dir,
    }
    cfg_path = os.path.join(args.work_dir, f"{args.name}_train.yaml")
    import yaml

    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"[3/5] Wrote training config -> {cfg_path}")

    print("[4/5] Training (this is the multi-hour GPU step) ...")
    # openWakeWord >=0.6 exposes a training entrypoint driven by the config.
    # (Equivalent to its automatic_model_training notebook.)
    owt.train_model(config)  # produces <output_dir>/<name>.onnx

    produced = os.path.join(args.work_dir, f"{args.name}.onnx")
    dest = os.path.join("voice/models", f"{args.name}.onnx")
    os.makedirs("voice/models", exist_ok=True)
    if os.path.exists(produced):
        shutil.copyfile(produced, dest)
        print(f"[5/5] Done -> {dest}\n      Set voice.input.wake_model: \"{args.name}\" and restart.")
        return 0
    print(f"[5/5] Training finished but {produced} not found — check the trainer logs.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
