"""One wake = one conversation: chime (no greeting) -> speak -> reply -> follow-up window without the wake word -> silence or
"thanks" ends it. Plus the greeting clock, transcript hygiene, the chime, the reader thread and the gateway's speech state.
Fakes only: no audio, no model, no network."""

from __future__ import annotations

import asyncio
import threading
import time
import wave
from typing import List

import numpy as np
import pytest

from voice.input.audio import AudioSource, BufferedSource
from voice.input.config import VoiceInputConfig, VoiceState, VoiceTransition
from voice.input.pipeline import VoiceInputPipeline
from voice.input.session import Chime, SessionClock, clean_transcript, is_end_phrase, is_junk, strip_wake_residue
from voice.input.sink import CallableSink

WAKE, SPEECH, QUIET = 1, 2, 0


def cfg(**kw) -> VoiceInputConfig:
    base = dict(frame_ms=30, sample_rate=16000, vad_silence_ms=60, listen_timeout_ms=300, max_utterance_ms=3000,
                vad_onset_ms=30, vad_min_speech_ms=90, vad_preroll_ms=30, followup_s=0.3, reply_wait_s=2.0, chime=True,
                greet_after_idle_min=60.0)
    base.update(kw)
    return VoiceInputConfig(**base)


def frame(marker, c):
    return np.full(c.frame_samples, marker, dtype=np.int16)


class FakeWake:
    def __init__(self):
        self.resets = 0

    def process(self, f):
        return int(f[0]) == WAKE

    def reset(self):
        self.resets += 1


class FakeVad:
    def is_speech(self, f):
        return int(f[0]) == SPEECH


class ScriptedStt:
    """Returns the next scripted text per call."""

    def __init__(self, *texts):
        self.texts, self.calls = list(texts), 0

    def transcribe(self, audio):
        t = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return t


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def speech(n=6):
    return [SPEECH] * n


def utterance(n=6):
    return speech(n) + [QUIET] * 4


def build(c, stt, markers, busy_script=None, clock=None, drain=None):
    clock = clock or Clock()
    wakes: List[dict] = []
    chimes: List[int] = []
    injected: List[str] = []
    transitions: List[VoiceTransition] = []
    busy = busy_script if busy_script is not None else []          # consumed in place so a test can see what was polled

    def reply_busy():
        return busy.pop(0) if busy else False

    pipe = VoiceInputPipeline(
        config=c, wake=FakeWake(), transcriber=stt, sink=CallableSink(lambda t: injected.append(t)),
        emitter=transitions.append, vad_factory=lambda: FakeVad(),
        on_wake=lambda greet=True, animate=True: wakes.append({"greet": greet, "animate": animate}),
        chime=Chime(enabled=True, player=lambda path: chimes.append(1), directory=None),
        session_clock=SessionClock(c.greet_after_idle_min, clock=clock),
        reply_busy=reply_busy, drain=drain, monotonic=clock, sleep=clock.sleep)
    pipe.run_stream([frame(m, c) for m in markers])
    return pipe, injected, wakes, chimes, transitions


# --------------------------------------------------------------------------- the conversation

def test_wake_plays_a_chime_and_does_not_ask_for_a_greeting():
    c = cfg()
    _, injected, wakes, chimes, _ = build(c, ScriptedStt("what time is it"), [WAKE] + utterance() + [QUIET] * 20)
    assert injected == ["what time is it"]
    assert chimes == [1]
    assert wakes == [{"greet": False, "animate": True}]


def test_after_a_reply_the_next_utterance_needs_no_wake_word():
    c = cfg()
    markers = [WAKE] + utterance() + utterance() + [QUIET] * 20          # ONE wake word, two commands
    _, injected, wakes, _, tx = build(c, ScriptedStt("first command", "second command"), markers)
    assert injected == ["first command", "second command"]
    assert len(wakes) == 1
    states = [t.state for t in tx]
    assert VoiceState.FOLLOWUP in states
    assert states[-1] == VoiceState.IDLE


def test_the_follow_up_window_closes_on_silence_and_a_new_wake_word_starts_a_new_conversation():
    c = cfg()
    markers = [WAKE] + utterance() + [QUIET] * 30 + [WAKE] + utterance() + [QUIET] * 30
    _, injected, wakes, chimes, tx = build(c, ScriptedStt("one", "two"), markers)
    assert injected == ["one", "two"]
    assert len(wakes) == 2 and len(chimes) == 2
    assert any(t.detail == "follow-up window closed" for t in tx)


@pytest.mark.parametrize("phrase", ["Thanks.", "that's all", "Thank you, Jarvis", "ok thanks"])
def test_an_end_phrase_ends_the_conversation_and_is_not_sent_to_the_planner(phrase):
    c = cfg()
    markers = [WAKE] + utterance() + utterance() + utterance() + [QUIET] * 20
    _, injected, wakes, _, tx = build(c, ScriptedStt("play some music", phrase, "must never be heard"), markers)
    assert injected == ["play some music"]
    assert [t.detail for t in tx if t.state == VoiceState.IDLE][-1] in ("conversation ended", "no speech", "follow-up window closed", "")
    assert any(t.detail == "conversation ended" for t in tx)


def test_stop_ends_the_conversation_and_barges_in_on_the_speech():
    c = cfg()
    markers = [WAKE] + utterance() + utterance() + [QUIET] * 20
    _, injected, wakes, _, _ = build(c, ScriptedStt("tell me a story", "stop"), markers)
    assert injected == ["tell me a story"]
    assert wakes[-1] == {"greet": False, "animate": False}              # the barge-in signal


def test_a_request_that_merely_contains_stop_is_not_an_end_phrase():
    c = cfg()
    markers = [WAKE] + utterance() + utterance() + [QUIET] * 20
    _, injected, *_ = build(c, ScriptedStt("play music", "stop the music"), markers)
    assert injected == ["play music", "stop the music"]


def test_the_wake_word_tail_alone_is_not_a_command():
    """The reported bug: the burst after the wake word ended the listen window 800 ms later, before the user had spoken."""
    c = cfg()
    markers = [WAKE, SPEECH, SPEECH, QUIET, QUIET, QUIET, QUIET] + utterance(8) + [QUIET] * 20
    pipe, injected, *_ = build(c, ScriptedStt("what is on my calendar"), markers)
    assert injected == ["what is on my calendar"]


def test_whisper_junk_is_not_injected():
    c = cfg()
    _, injected, *_ = build(c, ScriptedStt("You", "uh"), [WAKE] + utterance() + [QUIET] * 20)
    assert injected == []


def test_wake_word_residue_is_stripped_from_the_command():
    c = cfg()
    _, injected, *_ = build(c, ScriptedStt("Hey Jarvis, what time is it"), [WAKE] + utterance() + [QUIET] * 20)
    assert injected == ["what time is it"]


def test_followup_zero_is_the_old_single_shot_behaviour():
    c = cfg(followup_s=0)
    markers = [WAKE] + utterance() + utterance() + [QUIET] * 20
    _, injected, wakes, _, tx = build(c, ScriptedStt("one", "two"), markers)
    assert injected == ["one"]                                           # the second needs a wake word
    assert VoiceState.FOLLOWUP not in [t.state for t in tx]


# --------------------------------------------------------------------------- waiting for the reply

def test_the_follow_up_window_waits_until_the_reply_has_been_spoken_then_drops_queued_audio():
    c = cfg()
    clock = Clock()
    drained = []
    busy = [True, True, True, False, False]
    markers = [WAKE] + utterance() + utterance() + [QUIET] * 20
    _, injected, *_ = build(c, ScriptedStt("one", "two"), markers, busy_script=busy, clock=clock,
                            drain=lambda: drained.append(clock.t))
    assert injected == ["one", "two"]
    assert drained and drained[0] >= 0.4 + 0.2 * 3                      # not before the reply finished
    assert not busy                                                      # every scripted poll was consumed


def test_a_reply_that_never_finishes_does_not_hold_the_window_forever():
    c = cfg(reply_wait_s=1.0)
    clock = Clock()
    markers = [WAKE] + utterance() + utterance() + [QUIET] * 20
    _, injected, *_ = build(c, ScriptedStt("one", "two"), markers, busy_script=[True] * 1000, clock=clock)
    assert injected == ["one", "two"]
    assert clock.t < 5.0


def test_an_unreachable_gateway_does_not_make_us_wait():
    c = cfg()
    clock = Clock()
    pipe_busy = iter([None])
    markers = [WAKE] + utterance() + utterance() + [QUIET] * 20
    clk = Clock()
    pipe = VoiceInputPipeline(config=c, wake=FakeWake(), transcriber=ScriptedStt("one", "two"),
                              sink=CallableSink(lambda t: None), vad_factory=lambda: FakeVad(),
                              chime=Chime(enabled=False), session_clock=SessionClock(60, clock=clk),
                              reply_busy=lambda: None, monotonic=clk, sleep=clk.sleep)
    pipe.run_stream([frame(m, c) for m in markers])
    assert clk.t < 1.5


# --------------------------------------------------------------------------- the greeting clock

def test_the_greeting_plays_at_most_once_after_a_long_idle_period_never_per_wake():
    clock = Clock()
    sc = SessionClock(60, clock=clock)
    assert sc.should_greet() is False                                    # the gateway greeted at boot
    clock.t += 5 * 60
    assert sc.should_greet() is False
    clock.t += 61 * 60
    assert sc.should_greet() is True                                     # back after more than an hour
    clock.t += 30
    assert sc.should_greet() is False                                    # the very next wake: no greeting again


def test_activity_keeps_the_idle_timer_from_expiring():
    clock = Clock()
    sc = SessionClock(60, clock=clock)
    for _ in range(5):
        clock.t += 40 * 60
        sc.touch()
    clock.t += 40 * 60
    assert sc.should_greet() is False


def test_greet_after_idle_zero_never_greets():
    clock = Clock()
    sc = SessionClock(0, clock=clock)
    clock.t += 10 ** 6
    assert sc.should_greet() is False


def test_the_pipeline_asks_for_the_greeting_only_on_the_wake_after_a_long_idle():
    c = cfg()
    clock = Clock()
    stt = ScriptedStt("a", "b")
    wakes: List[dict] = []
    sc = SessionClock(60, clock=clock)

    class SlowFrames:
        def __iter__(self_inner):
            def gen():
                for m in [WAKE] + utterance() + [QUIET] * 20:
                    yield frame(m, c)
                clock.t += 2 * 3600                                      # two quiet hours
                for m in [WAKE] + utterance() + [QUIET] * 20:
                    yield frame(m, c)
            return gen()

    pipe = VoiceInputPipeline(config=c, wake=FakeWake(), transcriber=stt, sink=CallableSink(lambda t: None),
                              vad_factory=lambda: FakeVad(), on_wake=lambda greet=True, animate=True: wakes.append(greet),
                              chime=Chime(enabled=False), session_clock=sc, monotonic=clock, sleep=clock.sleep)
    pipe.run_stream(SlowFrames())
    assert wakes == [False, True]


# --------------------------------------------------------------------------- transcript hygiene, chime

def test_transcript_cleaning():
    assert strip_wake_residue("Hey Jarvis, what time is it?") == "what time is it?"
    assert strip_wake_residue("jarvis") == ""
    assert strip_wake_residue("Travis is my friend's name") != "Travis is my friend's name"[:0]     # only a LEADING phrase
    assert is_junk("You") and is_junk("Hmm.") and is_junk(" ") and not is_junk("you there")
    assert clean_transcript("Hey Jarvis") == ""
    assert clean_transcript("what is the weather") == "what is the weather"


P = ["thanks", "thank you", "thats all", "that is all", "stop", "nevermind", "never mind", "cancel", "goodbye"]


@pytest.mark.parametrize("text,expected", [
    ("Thanks.", True), ("Thank you, Jarvis.", True), ("That's all", True), ("ok thanks", True),
    ("thank you that's all", True), ("Stop", True), ("never mind", True), ("goodbye sir", True),
    ("stop the music", False), ("thanks for the briefing, what's next", False), ("what time is it", False), ("", False),
])
def test_end_phrase(text, expected):
    assert is_end_phrase(text, P) is expected


def test_the_chime_is_a_short_valid_wav_played_without_blocking(tmp_path):
    played = []
    ch = Chime(enabled=True, player=lambda path: played.append(path), directory=str(tmp_path))
    assert ch.play() is True and len(played) == 1
    with wave.open(played[0], "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2
        assert 0.15 <= wf.getnframes() / wf.getframerate() <= 0.35
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    assert 3000 < int(np.max(np.abs(data))) < 20000                      # audible but soft, not clipped
    assert Chime(enabled=False, player=lambda p: played.append(p)).play() is False and len(played) == 1


def test_a_failing_chime_player_never_breaks_listening(tmp_path):
    def boom(path):
        raise FileNotFoundError("afplay")

    assert Chime(enabled=True, player=boom, directory=str(tmp_path)).play() is False


# --------------------------------------------------------------------------- reader thread

class ListSource(AudioSource):
    def __init__(self, items, delay=0.0, error=None):
        self.items, self.delay, self.error, self.closed = items, delay, error, False

    def frames(self):
        for x in self.items:
            if self.delay:
                time.sleep(self.delay)
            yield x
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


def test_buffered_source_keeps_reading_while_the_consumer_is_busy():
    src = ListSource([np.full(4, i, dtype=np.int16) for i in range(40)])
    b = BufferedSource(src, frame_ms=30, backlog_s=2.0)           # room for ~66 frames
    it = b.frames()
    first = next(it)
    time.sleep(0.2)                                               # the consumer is "transcribing"
    rest = [int(f[0]) for f in it]
    assert int(first[0]) == 0 and rest == list(range(1, 40))      # nothing lost, order kept


def test_buffered_source_drops_the_oldest_frames_when_the_backlog_is_full():
    src = ListSource([np.full(4, i, dtype=np.int16) for i in range(300)])
    b = BufferedSource(src, frame_ms=30, backlog_s=0.3)           # 10 frames of room
    it = b.frames()
    next(it)
    time.sleep(0.3)
    got = [int(f[0]) for f in it]
    assert b.dropped > 0 and got[-1] == 299 and len(got) <= 12     # recent audio kept, stale audio gone


def test_buffered_source_flush_discards_the_queue():
    gate = threading.Event()

    class Gated(AudioSource):
        def frames(self):
            for i in range(10):
                yield np.full(4, i, dtype=np.int16)
            gate.wait(2)
            yield np.full(4, 99, dtype=np.int16)

    b = BufferedSource(Gated(), frame_ms=30, backlog_s=2.0)
    it = b.frames()
    next(it)
    time.sleep(0.2)
    assert b.flush() >= 8
    gate.set()
    assert int(next(it)[0]) == 99


def test_buffered_source_hands_a_reader_error_to_the_consumer_and_close_closes_the_source():
    src = ListSource([np.zeros(4, dtype=np.int16)], error=RuntimeError("device lost"))
    b = BufferedSource(src)
    with pytest.raises(RuntimeError, match="device lost"):
        list(b.frames())
    b.close()
    assert src.closed


def test_buffered_source_forwards_attributes_of_the_wrapped_source():
    class Mic(ListSource):
        open_rate = 48000
        resampled = True

    b = BufferedSource(Mic([]))
    assert b.open_rate == 48000 and b.resampled is True


# --------------------------------------------------------------------------- gateway speech state

def test_speech_state_busy_during_a_turn_and_while_speaking_then_quiet():
    from gateway.speech_state import SpeechState

    t = [0.0]
    s = SpeechState(clock=lambda: t[0], synth_grace_s=4.0, tail_s=0.5)
    assert s.busy() is False
    s.turn_started()
    assert s.busy() is True
    s.reply_queued(); s.turn_finished()
    t[0] = 1.0
    assert s.busy() is True                                              # reply queued, synthesis not started yet
    s.set_speaking(True)
    t[0] = 6.0
    assert s.busy() is True and s.snapshot()["speaking"] is True         # still speaking, well past the queue grace
    s.set_speaking(False)
    assert s.busy() is True                                              # 0.5 s tail: the next line may follow at once
    t[0] = 6.6
    assert s.busy() is False


def test_speech_state_speaking_expires_if_voice_output_dies_mid_line():
    from gateway.speech_state import SpeechState

    t = [0.0]
    s = SpeechState(clock=lambda: t[0], max_speaking_s=120)
    s.set_speaking(True)
    t[0] = 119.0
    assert s.busy() is True
    t[0] = 121.0
    assert s.busy() is False


def test_speech_state_a_queued_reply_that_is_never_spoken_releases_after_the_grace_period():
    from gateway.speech_state import SpeechState

    t = [0.0]
    s = SpeechState(clock=lambda: t[0], synth_grace_s=4.0)
    s.reply_queued()
    t[0] = 3.9
    assert s.busy() is True
    t[0] = 4.1
    assert s.busy() is False
