"""Sentence splitting (HUD display twin: hud/tests/sentences.test.ts) and the TTS chunker built on it."""
import pytest

from utils.sentences import split_sentences
from voice.output.speaker import Speaker, split_sentences as tts_split
from voice.output.config import VoiceOutputConfig


@pytest.mark.parametrize("text", [
    "Gate No. 2.", "RFID card at Gate No. 2.", "Mr. Rao called.", "Mrs. Rao and Dr. Mehta are here.", "St. Mary's school.",
    "Cats vs. dogs is settled.", "Apples, pears, etc. are fine.", "Standup is at 9 a.m. tomorrow.", "Call at 5 p.m. sharp.",
    "Use a tool, e.g. a hammer.", "That is, i.e. the second one.", "It costs 3. 5 more are due.", "Approx. five minutes.",
    "Wait? no, go.", "J. Smith called.",
])
def test_one_chunk(text):
    assert split_sentences(text) == [text]


def test_real_sentences_still_split():
    assert split_sentences("Good evening, Sir. You have two meetings. Shall I read them?") == [
        "Good evening, Sir.", "You have two meetings.", "Shall I read them?"]
    assert split_sentences("Meet at Gate No. 2. Then wait.") == ["Meet at Gate No. 2.", "Then wait."]
    assert split_sentences("Done! Next.") == ["Done!", "Next."]
    assert split_sentences("") == []


def test_tts_chunking_keeps_gate_no_2_together():
    assert tts_split("RFID card at Gate No. 2. Bring your badge.") == ["RFID card at Gate No. 2.", "Bring your badge."]


def test_speaker_synthesises_gate_no_2_as_one_chunk():
    spoken = []

    class Tts:
        def synthesize(self, s):
            spoken.append(s)
            return [0.0], 24000

    class Player:
        def play(self, samples, rate):
            return True

        def stop(self):
            pass

    Speaker(VoiceOutputConfig(streaming=True), tts=Tts(), player=Player()).speak("RFID card at Gate No. 2.")
    assert spoken == ["RFID card at Gate No. 2."]
