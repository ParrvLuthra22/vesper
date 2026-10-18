"""The microphone hears the speaker: half-duplex gating, a wake bar that rises while it speaks, an echo filter for transcripts that are
Vesper's own reply, and a loop breaker. Fake audio events and fake clocks only; nothing is recorded or played."""
from __future__ import annotations

import asyncio
import logging
from typing import List

import httpx
import numpy as np
import pytest

from gateway.echo import EchoFilter, LoopBreaker, coverage, normalize
from gateway.speech_state import SpeechState
from tests.test_gateway import _auth, _running_gateway
from voice.input.config import VoiceInputConfig
from voice.input.echo import EchoGate
from voice.input.pipeline import VoiceInputPipeline
from voice.input.session import Chime, SessionClock
from voice.input.sink import CallableSink
from voice.input.stages import SpeechSegmenter, WakeDetector

WAKE, SPEECH, QUIET = 1, 2, 0


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def cfg(**kw):
    base = dict(frame_ms=30, sample_rate=16000, vad_silence_ms=60, listen_timeout_ms=300, max_utterance_ms=3000, vad_onset_ms=30,
                vad_min_speech_ms=90, vad_preroll_ms=30, followup_s=0, chime=False)
    base.update(kw)
    return VoiceInputConfig(**base)


def frame(m, c):
    return np.full(c.frame_samples, m, dtype=np.int16)


# ------------------------------------------------------------------ the gate

def gate_with(clock, **snap):
    state = {"snap": {"audible": False, "pause_s": 0, **snap}}
    g = EchoGate(700, clock, lambda: state["snap"])
    return g, state


def test_nothing_speaking_means_no_gate():
    c = Clock()
    g, _ = gate_with(c)
    g.update()
    assert g.active() is False and g.speaking() is False and g.paused_for() == 0


def test_gate_is_shut_while_it_speaks_and_for_the_tail_afterwards():
    c = Clock()
    g, st = gate_with(c)
    st["snap"]["audible"] = True
    g.update()
    assert g.active() is True
    c.t += 5
    g.update()
    assert g.active() is True                                      # still speaking, however long
    st["snap"]["audible"] = False
    g.update()
    assert g.active() is True                                      # tail starts now
    c.t += 0.69
    assert g.active() is True
    c.t += 0.02
    assert g.active() is False                                     # 700 ms after the last audible moment


def test_gate_fails_open_when_the_gateway_cannot_be_asked():
    c = Clock()
    g = EchoGate(700, c, lambda: None)
    g.update()
    assert g.active() is False
    g2, st = gate_with(c)
    st["snap"]["audible"] = True
    g2.update()
    g2._source = lambda: None
    g2.update()                                                    # outage: keep the last known state, do not invent speech
    assert g2.active() is True


def test_pause_seconds_come_from_the_gateway_and_run_down():
    c = Clock()
    g, st = gate_with(c, pause_s=30.0)
    g.update()
    assert g.paused_for() == pytest.approx(30.0)
    c.t += 12
    assert g.paused_for() == pytest.approx(18.0)
    st["snap"]["pause_s"] = 0
    g.update()
    assert g.paused_for() == 0


# ------------------------------------------------------------------ the pipeline honours the gate

class FakeGate:
    def __init__(self):
        self.on, self.pause = False, 0.0

    def active(self):
        return self.on

    def speaking(self):
        return self.on

    def paused_for(self):
        return self.pause


class FakeWake:
    def __init__(self):
        self.speaking = None
        self.seen_speaking: List[bool] = []

    def process(self, f):
        self.seen_speaking.append(self.speaking)
        return int(f[0]) == WAKE

    def reset(self):
        pass


class FakeVad:
    def is_speech(self, f):
        return int(f[0]) == SPEECH


class Stt:
    def __init__(self):
        self.lengths = []

    def transcribe(self, audio):
        self.lengths.append(len(audio))
        return "turn on the lights"


def run_pipeline(script, gate=None, c=None):
    """script: list of (marker, gate_on|None). A non-None gate_on is applied BEFORE that frame is delivered."""
    c = c or cfg()
    injected, stt, wake = [], Stt(), FakeWake()

    def frames():
        for marker, g in script:
            if g is not None and gate is not None:
                gate.on = g
            yield frame(marker, c)

    pipe = VoiceInputPipeline(config=c, wake=wake, transcriber=stt, sink=CallableSink(lambda t: injected.append(t)),
                              vad_factory=lambda: FakeVad(), chime=Chime(enabled=False), session_clock=SessionClock(60),
                              gate=gate)
    pipe.run_stream(frames())
    return pipe, injected, stt, wake


def test_its_own_voice_after_the_wake_is_not_captured_as_an_utterance():
    g = FakeGate()
    # wake, then 8 frames of the assistant's own greeting/reply leaking into the mic (gate shut), then it stops and the user speaks
    script = [(WAKE, None)] + [(SPEECH, True)] * 8 + [(QUIET, False)] * 2 + [(SPEECH, None)] * 6 + [(QUIET, None)] * 6
    pipe, injected, stt, _ = run_pipeline(script, g)
    assert injected == ["turn on the lights"] and pipe.gated_frames == 8
    assert len(stt.lengths) == 1 and stt.lengths[0] < 14 * 480            # the 8 echo frames are NOT in the audio sent to STT


def test_without_the_gate_the_same_echo_would_have_become_the_command():
    script = [(WAKE, None)] + [(SPEECH, None)] * 8 + [(QUIET, None)] * 6
    pipe, injected, stt, _ = run_pipeline(script, gate=None)
    assert injected == ["turn on the lights"] and stt.lengths[0] >= 8 * 480    # the old behaviour: the echo is transcribed


def test_the_listen_timeout_does_not_run_while_the_gate_is_shut():
    g = FakeGate()
    script = [(WAKE, None)] + [(QUIET, True)] * 40 + [(SPEECH, False)] * 6 + [(QUIET, None)] * 6      # 40 frames > the 300 ms timeout
    pipe, injected, _, _ = run_pipeline(script, g)
    assert injected == ["turn on the lights"]


def test_no_gate_when_nothing_is_speaking():
    g = FakeGate()
    script = [(WAKE, None), (SPEECH, False)] + [(SPEECH, None)] * 5 + [(QUIET, None)] * 6
    pipe, injected, _, _ = run_pipeline(script, g)
    assert injected == ["turn on the lights"] and pipe.gated_frames == 0


def test_the_wake_detector_is_told_when_it_is_speaking():
    g = FakeGate()
    script = [(QUIET, True), (QUIET, True), (QUIET, False), (WAKE, None), (SPEECH, None)] + [(SPEECH, None)] * 4 + [(QUIET, None)] * 6
    _, _, _, wake = run_pipeline(script, g)
    assert wake.seen_speaking[:3] == [True, True, False]


def test_voice_paused_by_the_loop_breaker_ignores_even_the_wake_word():
    g = FakeGate()
    g.pause = 30.0
    script = [(WAKE, None)] + [(SPEECH, None)] * 6 + [(QUIET, None)] * 6
    pipe, injected, _, wake = run_pipeline(script, g)
    assert injected == [] and wake.seen_speaking == []                    # frames were not even offered to the wake detector


def test_segmenter_hold_forgets_a_half_built_onset_but_not_a_started_utterance():
    c = cfg(vad_onset_ms=90)
    seg = SpeechSegmenter(c, FakeVad())
    seg.push(frame(SPEECH, c)); seg.push(frame(SPEECH, c))
    seg.hold()
    seg.push(frame(SPEECH, c))
    assert seg.audio.size == 0                                              # one frame after the hold is not an onset
    seg2 = SpeechSegmenter(cfg(), FakeVad())
    for _ in range(4):
        seg2.push(frame(SPEECH, cfg()))
    n = seg2.audio.size
    seg2.hold()                                                             # the user is already speaking: leave it alone
    assert seg2.audio.size == n


# ------------------------------------------------------------------ the wake bar rises while speaking

class Model:
    def __init__(self, score):
        self.score = score

    def predict(self, w):
        return {"hey_jarvis": self.score}

    def reset(self):
        pass


def detector(score):
    d = WakeDetector(VoiceInputConfig())                  # defaults: 0.35, 2-of-3, echo bar 0.7
    d._model, d._model_key = Model(score), "hey_jarvis"
    return d


def windows(n):
    return np.zeros(1280 * n, dtype=np.int16)


def test_a_middling_score_wakes_when_silent_but_not_while_it_speaks():
    d = detector(0.5)
    assert d.process(windows(3)) is True
    d = detector(0.5)
    d.speaking = True
    assert d.threshold() == 0.7 and d.process(windows(3)) is False        # its own voice at 0.5 cannot wake it


def test_a_strong_wake_word_still_barges_in_while_it_speaks():
    d = detector(0.85)
    d.speaking = True
    assert d.process(windows(3)) is True


def test_echo_disabled_keeps_the_single_threshold():
    d = WakeDetector(VoiceInputConfig(echo_enabled=False))
    d._model, d._model_key = Model(0.5), "hey_jarvis"
    d.speaking = True
    assert d.threshold() == 0.35 and d.process(windows(3)) is True


def test_config_defaults_and_yaml_keys():
    c = VoiceInputConfig()
    assert (c.echo_enabled, c.echo_tail_ms, c.echo_wake_threshold_while_speaking) == (True, 700, 0.7)
    from config.settings import load_config_dict
    echo = load_config_dict()["voice"]["echo"]
    assert echo["tail_ms"] == 700 and echo["wake_threshold_while_speaking"] == 0.7 and echo["pause_s"] == 30


# ------------------------------------------------------------------ the echo filter

REPLIES = ["Good evening, Sir. You have two meetings left today.", "I've hit the language model's rate limit, Sir. Try again in a minute.",
           "Opening Safari for you now, Sir.", "Nothing important in your inbox, Sir. Four unread in total."]


def flt(*replies, **kw):
    f = EchoFilter(**kw)
    for r in replies:
        f.add_reply(r)
    return f


@pytest.mark.parametrize("heard", [
    "I've hit the language model's rate limit Sir try again in a minute",
    "hit the language model's rate limit sir",                        # the reply, partly heard
    "I have hit the language models rate limit sir. Try again in a minute",   # STT spelling noise
    "good evening sir you have two meetings left today",
])
def test_the_assistants_own_reply_is_dropped(heard):
    assert flt(*REPLIES).check(heard) == "echo_match"


@pytest.mark.parametrize("heard", [
    "open safari for me",                                             # a real command that shares words with a past reply
    "what's on my calendar",
    "you have two meetings, move the second one to friday",
    "thanks", "stop", "good evening",                                 # short: never treated as echo
    "check my email and tell me if anything is important",
    "is there a rate limit on this api",
])
def test_real_commands_are_not_dropped_even_when_similar_to_a_reply(heard):
    assert flt(*REPLIES).check(heard) is None


def test_only_the_last_five_replies_count():
    f = EchoFilter(history=5)
    f.add_reply("this is the very first reply that was said long ago")
    assert f.check("this is the very first reply that was said long ago") == "echo_match"
    for i in range(5):
        f.add_reply(f"completely different later sentence number {i} about other things")
    assert f.check("this is the very first reply that was said long ago") is None


def test_threshold_and_min_words_are_configurable():
    strict = flt("Opening Safari for you now, Sir.", threshold=0.95)
    assert strict.check("opening safari for me") is None                  # 3 of 4 words covered = 0.75 < 0.95
    loose = flt("Opening Safari for you now, Sir.", threshold=0.5, min_words=3)
    assert loose.check("opening safari for") == "echo_match"


def test_coverage_is_in_order_word_coverage():
    assert coverage(normalize("a b c d"), normalize("a b c d e f")) == 1.0
    assert coverage(normalize("d c b a"), normalize("a b c d")) <= 0.5


def test_drops_log_only_a_count_and_a_reason_code(caplog):
    caplog.set_level(logging.INFO)

    async def go():
        async with _running_gateway() as (gw, port, bus, brain):
            gw._echo.add_reply("the quick brown fox jumps over the lazy dog today")
            async with httpx.AsyncClient() as c:
                url = f"http://127.0.0.1:{port}"
                r = await c.post(f"{url}/message", json={"text": "the quick brown fox jumps over the lazy dog today", "origin": "voice"},
                                 headers=_auth())
                return r.json(), gw
    body, gw = asyncio.run(go())
    assert body == {"status": "dropped", "reason": "echo_match"}                 # nothing echoed back
    log = " ".join(r.getMessage() for r in caplog.records)
    assert "reason=echo_match" in log and "1 dropped so far" in log and "quick brown fox" not in log


@pytest.mark.asyncio
async def test_gateway_drops_echo_only_for_voice_transcripts_and_lets_typed_text_through():
    async with _running_gateway() as (gw, port, bus, brain):
        phrase = "I have hit the language model's rate limit, Sir, try again"
        gw._echo.add_reply(phrase)
        async with httpx.AsyncClient() as c:
            url = f"http://127.0.0.1:{port}"
            v = await c.post(f"{url}/message", json={"text": phrase, "origin": "voice"}, headers=_auth())
            t = await c.post(f"{url}/message", json={"text": phrase}, headers=_auth())          # typed in the HUD / CLI: the user means it
            await asyncio.sleep(0.2)
        assert v.json()["status"] == "dropped" and t.json()["status"] == "queued"
        assert brain.received == [phrase]


@pytest.mark.asyncio
async def test_the_gateways_own_replies_feed_the_filter():
    from schemas.events import VoiceOutputEvent

    async with _running_gateway() as (gw, port, bus, brain):
        await bus.emit(VoiceOutputEvent(text="Sir, the weather service answered with an error today", source="Brain"))
        assert gw._echo.check("the weather service answered with an error today") == "echo_match"


# ------------------------------------------------------------------ the loop breaker

def breaker(c=None, **kw):
    c = c or Clock()
    trips: List[str] = []
    return LoopBreaker(clock=c, on_trip=trips.append, **kw), c, trips


def test_three_failed_turns_within_a_minute_pause_voice_for_30_seconds():
    b, c, trips = breaker()
    b.record_failure(); c.t += 20; b.record_failure()
    assert b.paused_for() == 0
    c.t += 20; b.record_failure()
    assert trips == ["failed_turns"] and b.paused_for() == pytest.approx(30.0)
    c.t += 31
    assert b.paused_for() == 0                                              # recovery


def test_failures_spread_over_more_than_a_minute_do_not_trip():
    b, c, trips = breaker()
    for _ in range(5):
        b.record_failure(); c.t += 35                                    # never 3 inside one 60 s window
    assert trips == []


def test_three_quick_consecutive_turns_after_speech_trip_the_breaker():
    b, c, trips = breaker()
    for gap in (0.5, 1.0, 2.9):
        b.record_turn_start(gap)
    assert trips == ["quick_turns"] and b.paused_for() > 0


def test_a_normal_human_pause_resets_the_quick_turn_count():
    b, c, trips = breaker()
    b.record_turn_start(1.0); b.record_turn_start(2.0)
    b.record_turn_start(8.0)                                                # a human answered after thinking
    b.record_turn_start(1.0); b.record_turn_start(1.0)
    assert trips == []
    b.record_turn_start(None)                                               # it had not spoken at all
    assert trips == []


def test_trips_once_per_pause_and_recovers():
    b, c, trips = breaker()
    for _ in range(6):
        b.record_failure()
    assert trips == ["failed_turns"]                                        # not re-tripped while already paused
    c.t += 31
    for _ in range(3):
        b.record_failure()
    assert len(trips) == 2


@pytest.mark.asyncio
async def test_a_trip_shows_one_hud_notice_and_says_nothing_aloud_and_voice_is_refused_meanwhile():
    from schemas.events import ProactiveNotificationEvent, VoiceOutputEvent

    async with _running_gateway() as (gw, port, bus, brain):
        seen = []

        async def grab(e):
            seen.append(e)

        bus.subscribe(ProactiveNotificationEvent, grab)
        bus.subscribe(VoiceOutputEvent, grab)
        for _ in range(3):
            gw._turn_outcome(True)
        await asyncio.sleep(0.2)
        notes = [e for e in seen if isinstance(e, ProactiveNotificationEvent)]
        assert len(notes) == 1 and notes[0].kind == "voice_pause" and len(notes[0].body) < 80
        assert not [e for e in seen if isinstance(e, VoiceOutputEvent)]
        async with httpx.AsyncClient() as c:
            url = f"http://127.0.0.1:{port}"
            snap = (await c.get(f"{url}/speech", headers=_auth())).json()
            assert 0 < snap["pause_s"] <= 30
            r = await c.post(f"{url}/message", json={"text": "turn on the lights please now", "origin": "voice"}, headers=_auth())
            assert r.json() == {"status": "dropped", "reason": "paused"}


@pytest.mark.asyncio
async def test_quick_voice_turns_right_after_it_spoke_trip_the_gateway_breaker():
    async with _running_gateway() as (gw, port, bus, brain):
        clock = Clock()
        gw._speech = SpeechState(clock=clock)
        gw._breaker._clock = clock
        async with httpx.AsyncClient() as c:
            url = f"http://127.0.0.1:{port}"
            for i, text in enumerate(("play something nice", "what time is it now", "open the notes app please")):
                gw._speech.set_speaking(True); clock.t += 2; gw._speech.set_speaking(False)
                clock.t += 1.0
                await c.post(f"{url}/message", json={"text": text, "origin": "voice"}, headers=_auth())
        assert gw._breaker.trips == 1 and gw._breaker.paused_for() > 0


def test_brain_reports_aborted_turns_to_the_observer():
    from unittest.mock import AsyncMock
    from orchestrator.brain import Brain
    from schemas.events import VoiceInputEvent

    brain = Brain(config={})
    got = []
    brain.turn_observer = lambda failed, cid: got.append(failed)
    for aborted in (True, False):
        brain.handle_user_text = AsyncMock(return_value=type("R", (), {"aborted": aborted})())
        asyncio.run(brain._handle_voice_input(VoiceInputEvent(text="hello there", source="gateway")))
    assert got == [True, False]


def test_sink_treats_a_dropped_response_as_not_accepted_and_the_pipeline_stops_the_conversation():
    from voice.input.sink import GatewayRestSink

    sink = GatewayRestSink(VoiceInputConfig())
    sink._request = lambda m, p, payload, timeout: (True, {"status": "dropped", "reason": "echo_match"})
    assert sink.inject("anything at all here") is False
    sink._request = lambda m, p, payload, timeout: (True, {"status": "queued"})
    assert sink.inject("anything at all here") is True

    c = cfg()
    pipe = VoiceInputPipeline(config=c, wake=FakeWake(), transcriber=Stt(), sink=CallableSink(lambda t: False), vad_factory=lambda: FakeVad(),
                              chime=Chime(enabled=False), session_clock=SessionClock(60))
    pipe.run_stream([frame(m, c) for m in [WAKE] + [SPEECH] * 6 + [QUIET] * 6])
    assert pipe.state.value == "idle"
