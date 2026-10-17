"""Wake-word sensitivity: the threshold is a documented config knob, and the max score per 10 s window is logged (scores only
— never audio) so it can be tuned from real use: at INFO when something wake-like (>= 0.05) was heard, at DEBUG when the
window was quiet. Fake model; nothing is loaded or opened. The patience guard has its own tests in test_wake_patience.py;
these use hits=1 / window=1, which is the single-window behaviour."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from voice.input.config import VoiceInputConfig
from voice.input.stages import WakeDetector

WINDOW = 1280


class FakeModel:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def predict(self, window):
        assert len(window) == WINDOW
        self.calls += 1
        s = self.scores[min(self.calls - 1, len(self.scores) - 1)]
        return {"hey_jarvis": s}

    def reset(self):
        pass


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def detector(scores, threshold=0.5, clock=None, hits=1, window=1):
    d = WakeDetector(VoiceInputConfig(wake_threshold=threshold, wake_patience_hits=hits, wake_patience_window=window),
                     clock=clock or Clock())
    d._model, d._model_key = FakeModel(scores), "hey_jarvis"
    return d


def chunk(windows=1):
    return np.zeros(WINDOW * windows, dtype=np.int16)


def test_feed_returns_a_score_per_80ms_window_without_thresholding_or_resetting():
    d = detector([0.1, 0.9, 0.2])
    assert d.feed(chunk(3)) == [0.1, 0.9, 0.2]              # 0.9 is above the threshold, and feed() still reports all three
    assert d.feed(np.zeros(100, dtype=np.int16)) == []        # a partial window is buffered, not scored


def test_process_still_fires_at_the_threshold_exactly_as_before():
    assert detector([0.49]).process(chunk()) is False
    assert detector([0.5]).process(chunk()) is True
    assert detector([0.2, 0.2, 0.8, 0.1]).process(chunk(4)) is True
    assert detector([0.3], threshold=0.25).process(chunk()) is True       # the configured threshold, not a constant


def test_the_default_threshold_is_035_with_a_two_of_three_patience_guard():
    c = VoiceInputConfig()
    assert (c.wake_threshold, c.wake_patience_hits, c.wake_patience_window) == (0.35, 2, 3)


def test_max_score_per_10s_window_is_logged_at_info_when_wake_like_with_no_audio(caplog):
    caplog.set_level(logging.DEBUG, logger="vesper.voice")
    clock = Clock()
    d = detector([0.05, 0.31, 0.12, 0.2], clock=clock)
    d.process(chunk(2)); clock.t += 4
    d.process(chunk(2)); clock.t += 7                      # 11 s since the window started -> one line
    d.process(chunk(1))
    lines = [r for r in caplog.records if "wake: max score" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    msg = lines[0].getMessage()
    assert "0.310" in msg and "threshold 0.50" in msg and "over the last 11s" in msg
    assert not any(isinstance(a, (bytes, np.ndarray)) for r in caplog.records for a in (r.args or ()))     # scores only, never audio


def test_each_window_reports_its_own_max_then_resets():
    msgs = []
    clock = Clock()
    d = detector([0.4, 0.1, 0.1, 0.1], clock=clock)
    handler = logging.Handler()
    handler.emit = lambda r: msgs.append(r.getMessage())
    log = logging.getLogger("vesper.voice")
    old = log.level
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    try:
        d.process(chunk()); clock.t += 10.5; d.process(chunk())       # window 1: max 0.4 reported at the second call
        clock.t += 10.5; d.process(chunk())                           # window 2: max 0.1
    finally:
        log.removeHandler(handler); log.setLevel(old)
    maxes = [m for m in msgs if "wake: max score" in m]
    assert "0.400" in maxes[0] and "0.100" in maxes[1]


def test_a_quiet_window_is_only_logged_at_debug(caplog):
    caplog.set_level(logging.INFO, logger="vesper.voice")
    clock = Clock()
    d = detector([0.01])
    d._clock = clock
    d.process(chunk()); clock.t += 30; d.process(chunk())
    assert not [r for r in caplog.records if "wake: max score" in r.getMessage()]
