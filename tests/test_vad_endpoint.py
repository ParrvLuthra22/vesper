"""End-of-utterance detection (SpeechSegmenter) on SYNTHETIC audio: tone bursts stand in for words, silence for pauses, an
energy gate stands in for the VAD. The point is the timing logic: pauses inside a sentence must not cut it, the first word
must not be clipped, a short blip (wake-word tail / chime) must not become a command, and the caps must hold."""

from __future__ import annotations

import numpy as np

from voice.input.config import VoiceInputConfig
from voice.input.stages import SpeechSegmenter

RATE = 16000
FRAME_MS = 30
FRAME = RATE * FRAME_MS // 1000


class EnergyVad:
    def is_speech(self, frame: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) > 1000.0


def tone(seconds: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (0.3 * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(RATE * seconds), dtype=np.int16)


def cfg(**kw) -> VoiceInputConfig:
    return VoiceInputConfig(frame_ms=FRAME_MS, sample_rate=RATE, **kw)       # the real defaults unless overridden


def drive(segmenter: SpeechSegmenter, signal: np.ndarray):
    """Push 30 ms frames until the segmenter says DONE/TIMEOUT. Returns (verdict, seconds consumed)."""
    n = 0
    for i in range(0, len(signal) - FRAME + 1, FRAME):
        n += 1
        verdict = segmenter.push(signal[i:i + FRAME])
        if verdict != SpeechSegmenter.CONTINUE:
            return verdict, n * FRAME_MS / 1000
    return SpeechSegmenter.CONTINUE, n * FRAME_MS / 1000


def test_defaults_are_1200ms_silence_and_30s_cap():
    c = VoiceInputConfig()
    assert (c.vad_silence_ms, c.max_utterance_ms, c.vad_preroll_ms) == (1200, 30000, 300)


def test_a_one_second_pause_inside_a_sentence_does_not_cut_it():
    sig = np.concatenate([silence(0.3), tone(1.5), silence(1.0), tone(1.5), silence(1.6)])
    seg = SpeechSegmenter(cfg(), EnergyVad())
    verdict, _ = drive(seg, sig)
    assert verdict == SpeechSegmenter.DONE
    assert seg.speech_ms >= 2900                                # BOTH halves were captured (3.0 s of tone)
    assert len(seg.audio) >= int(RATE * 4.0)


def test_the_old_800ms_setting_would_have_cut_the_same_sentence():
    sig = np.concatenate([silence(0.3), tone(1.5), silence(1.0), tone(1.5), silence(1.6)])
    seg = SpeechSegmenter(cfg(vad_silence_ms=800), EnergyVad())
    verdict, _ = drive(seg, sig)
    assert verdict == SpeechSegmenter.DONE and seg.speech_ms < 1600     # cut after the first half: the reported bug


def test_a_pause_longer_than_silence_ms_ends_the_utterance():
    sig = np.concatenate([silence(0.3), tone(1.5), silence(1.5), tone(1.5), silence(1.6)])
    seg = SpeechSegmenter(cfg(), EnergyVad())
    verdict, consumed = drive(seg, sig)
    assert verdict == SpeechSegmenter.DONE
    assert seg.speech_ms < 1600                                 # the second half is a new utterance
    assert 3.0 <= consumed <= 3.3                               # 0.3 + 1.5 + 1.2 s of silence


def test_the_hard_cap_ends_a_never_ending_utterance_at_30s():
    sig = tone(40.0)
    seg = SpeechSegmenter(cfg(), EnergyVad())
    verdict, consumed = drive(seg, sig)
    assert verdict == SpeechSegmenter.DONE and 29.5 <= consumed <= 30.5


def test_max_utterance_is_configurable():
    seg = SpeechSegmenter(cfg(max_utterance_ms=5000), EnergyVad())
    verdict, consumed = drive(seg, tone(20.0))
    assert verdict == SpeechSegmenter.DONE and 4.5 <= consumed <= 5.5


def test_the_first_word_is_not_clipped_thanks_to_the_preroll():
    lead = silence(1.0)
    word = tone(1.0)
    sig = np.concatenate([lead, word, silence(1.6)])
    seg = SpeechSegmenter(cfg(), EnergyVad())
    drive(seg, sig)
    audio = seg.audio
    # the captured clip starts BEFORE the first loud sample (pre-roll), by at least the onset time + part of the pre-roll
    first_loud = int(np.argmax(np.abs(audio) > 1000))
    assert first_loud >= int(RATE * 0.20)
    # ...and the word is entirely inside the clip
    assert int(np.sum(np.abs(audio) > 1000)) >= int(0.95 * np.sum(np.abs(word) > 1000))


def test_a_short_blip_is_discarded_and_the_real_command_still_gets_through():
    blip = tone(0.15)                                           # the tail of "…vis", or a chime
    sig = np.concatenate([blip, silence(1.4), silence(0.6), tone(1.2), silence(1.6)])
    seg = SpeechSegmenter(cfg(), EnergyVad())
    verdict, _ = drive(seg, sig)
    assert verdict == SpeechSegmenter.DONE
    assert seg.discarded_blips == 1
    assert 1100 <= seg.speech_ms <= 1300                        # only the real 1.2 s of speech


def test_without_the_blip_rule_the_wake_tail_alone_would_be_sent_to_stt():
    sig = np.concatenate([tone(0.15), silence(1.6)])
    seg = SpeechSegmenter(cfg(vad_min_speech_ms=30, vad_onset_ms=30), EnergyVad())
    assert drive(seg, sig)[0] == SpeechSegmenter.DONE           # how the old pipeline behaved: a 0.9 s "utterance" of nothing


def test_no_speech_times_out_and_a_custom_timeout_is_honoured():
    seg = SpeechSegmenter(cfg(), EnergyVad())
    verdict, consumed = drive(seg, silence(10.0))
    assert verdict == SpeechSegmenter.TIMEOUT and 5.9 <= consumed <= 6.1             # listen_timeout_ms default 6 s
    seg = SpeechSegmenter(cfg(), EnergyVad(), timeout_ms=8000)                       # the follow-up window
    verdict, consumed = drive(seg, silence(10.0))
    assert verdict == SpeechSegmenter.TIMEOUT and 7.9 <= consumed <= 8.1


def test_a_click_shorter_than_the_onset_is_not_speech():
    seg = SpeechSegmenter(cfg(), EnergyVad())
    sig = np.concatenate([silence(0.5), tone(0.03), silence(0.5), tone(0.03), silence(6.0)])
    assert drive(seg, sig)[0] == SpeechSegmenter.TIMEOUT and len(seg.audio) == 0


def test_a_12_second_utterance_with_one_second_pauses_is_captured_whole():
    parts = []
    for _ in range(5):
        parts += [tone(2.0), silence(1.0)]
    sig = np.concatenate([silence(0.3)] + parts[:-1] + [silence(1.6)])      # 5 phrases of 2 s, 4 pauses of 1 s
    seg = SpeechSegmenter(cfg(), EnergyVad())
    verdict, _ = drive(seg, sig)
    assert verdict == SpeechSegmenter.DONE and seg.speech_ms >= 9800
