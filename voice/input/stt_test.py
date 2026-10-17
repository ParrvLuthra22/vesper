"""`vesper stt-test "<phrase>" [--names Priya,Rahul]` — does the name vocabulary help? No microphone.

The phrase is synthesized with the local Kokoro voice, converted to 16 kHz, and transcribed twice with the configured faster-whisper
model: once plain, once primed with a vocabulary. The vocabulary is ONLY the names passed with --names: this command never reads your
Contacts or your mail headers, so nothing about real people can appear unless you type it. Both transcripts are printed (they are the
text you supplied, as heard). Nothing is recorded, played, saved or sent.
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from voice.input.vocab import build_vocab, prompt_from


def run_stt_test(phrase: str, names: Sequence[str], synth: Callable[[str], Tuple[np.ndarray, int]],
                 transcribe: Callable[[np.ndarray, str], str], out=print) -> Tuple[str, str]:
    """(plain transcript, primed transcript). `synth` -> (float samples, rate); `transcribe(int16 16k audio, prompt) -> text`."""
    from voice.input.selftest import resample_float

    samples, rate = synth(phrase)
    x = resample_float(np.asarray(samples, dtype=np.float64), rate, 16000)
    peak = float(np.max(np.abs(x))) or 1.0
    pcm = np.clip(np.rint(x / peak * 0.5 * 32767), -32768, 32767).astype(np.int16)
    pcm = np.concatenate([np.zeros(8000, dtype=np.int16), pcm, np.zeros(8000, dtype=np.int16)])
    vocab = build_vocab(names, (), cap=60)
    prompt = prompt_from(vocab)
    plain = transcribe(pcm, "")
    primed = transcribe(pcm, prompt) if prompt else plain
    out(f"phrase      : {phrase}")
    out(f"vocabulary  : {len(vocab)} name(s) from --names" + ("" if vocab else "  (none given: pass --names A,B to see the effect)"))
    out(f"without     : {plain}")
    out(f"with vocab  : {primed}" + ("" if prompt else "  (same run: no vocabulary)"))
    out("same" if plain.strip().lower() == primed.strip().lower() else "DIFFERENT — the vocabulary changed what was heard")
    return plain, primed


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="vesper stt-test", description=__doc__.split("\n\n")[0])
    ap.add_argument("phrase", help='what to synthesize and transcribe, e.g. "message Priya that I am late"')
    ap.add_argument("--names", default="", help="comma-separated names to use as the vocabulary (never read from Contacts here)")
    args = ap.parse_args(list(argv) if argv is not None else sys.argv[1:])

    import dataclasses
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")      # the model is on disk (like launcher.offline_models); no Hub request

    from config.settings import load_config_dict
    from voice.input.config import VoiceInputConfig
    from voice.input.stages import Transcriber
    from voice.output.config import VoiceOutputConfig
    from voice.output.tts import KokoroTTS

    cfg = load_config_dict()
    vin = VoiceInputConfig.from_app_config(cfg)
    tts = KokoroTTS(dataclasses.replace(VoiceOutputConfig.from_app_config(cfg), speed=1.0))
    stt = Transcriber(vin)
    stt.load()

    def transcribe(pcm: np.ndarray, prompt: str) -> str:
        kw = {"initial_prompt": prompt} if prompt else {}
        segments, _ = stt._model.transcribe(pcm.astype(np.float32) / 32768.0, language=vin.whisper_language,
                                            beam_size=vin.whisper_beam_size, vad_filter=False, **kw)
        return " ".join(s.text.strip() for s in segments).strip()

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    run_stt_test(args.phrase, names, lambda p: tts.synthesize(p), transcribe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
