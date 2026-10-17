"""`vesper logs <component> --tail N --redact-transcripts`: recent log lines with anything said (or said to you) replaced by
<text:N> placeholders. Synthetic transcripts only — nothing here reads a real log."""

from __future__ import annotations

import re

import pytest

from launcher import cli
from launcher.logsafe import Placeholders, redact_line, redact_lines

# Unmistakably fake things a person might say / a reply might contain.
CANARIES = ["glockenspiel rehearsal", "Wilhelmina Fizzlewick", "marzipan invoice", "call my dentist about the tangerine",
            "quokka", "4155550142", "hunter2"]


def emit(event, payload, eid="3f9a1c2e-0b7d-4e55-9a61-8c2d4f6a1b09"):
    return (f"[02:40:11] INFO event_bus: [EMIT] Event: {event} | ID: {eid} | Source: Brain | Priority: NORMAL | "
            f"Payload: {payload}")


LEAKY_LINES = [
    emit("VoiceInputEvent", "{'text': 'play the glockenspiel rehearsal please', 'confidence': 0.93, 'source': 'voice'}"),
    emit("VoiceOutputEvent", "{'text': 'Wilhelmina Fizzlewick called about the marzipan invoice, Sir.', 'voice': 'bm_lewis'}"),
    emit("VoiceInputEvent", '{"text": "call my dentist about the tangerine", "wake": true}'),
    emit("ObservationEvent", "{'kind': 'meeting_soon', 'detail': \"Your meeting 'quokka sync' starts in 5 minutes.\", 'observation_id': 'abc'}"),
    emit("ConfirmationRequestedEvent", "{'summary': 'draft to Wilhelmina Fizzlewick about marzipan invoice', 'tool_name': 'gmail_create_draft', "
                                       "'arguments': {'to': ['wilhelmina@fizzle.example'], 'body': 'hunter2'}, 'request_id': 'r-1'}"),
    emit("ContextUpdatedEvent", "{'context_type': 'turn', 'context_key': 'user_input', 'context_value': 'quokka at 4155550142', 'turn_number': 3}"),
    emit("ToolCallFinishedEvent", "{'tool_name': 'search', 'success': True, 'result': 'marzipan invoice from Wilhelmina Fizzlewick'}"),
    "[VoiceAgent] Received VoiceOutputEvent: 'the glockenspiel rehearsal moves to Thursday' from Brain",
    "2026-10-17 02:41:00 INFO orchestrator.brain [brain.py:1210] Observation: kind=meeting_soon detail='quokka sync soon'",
    "2026-10-17 02:41:01,100 | INFO | vesper.voice | heard: call my dentist about the tangerine",
    "2026-10-17 02:41:01,100 | INFO | vesper.voice | Transcript: Wilhelmina Fizzlewick wants the marzipan invoice",
    "2026-10-17 02:41:01,100 | INFO | vesper.voice | transcript='hunter2 is my password' confidence=0.8",
    "2026-10-17 02:41:02 INFO x: user_text=glockenspiel rehearsal at 4155550142",
    "2026-10-17 02:41:03,200 | INFO | vesper.voice | voice transcribing -> injected (call my dentist about the tangerine)",
    "2026-10-17 02:41:03,201 | INFO | vesper.voice | voice transcribing -> injected (hunter2 and (4155550142) quokka)",
    "[02:40:42] INFO brain: User spoke: 'play the glockenspiel rehearsal, it's for Wilhelmina Fizzlewick'",
    "[02:40:43] INFO voice_agent: Transcribed: 'call my dentist about the tangerine' (confidence: 0.91)",
    # a payload cut off mid-string by the logger
    emit("VoiceInputEvent", "{'text': 'play the glockenspiel rehea"),
    # an UNKNOWN key must be treated as text (fail closed)
    emit("SomethingNewEvent", "{'brand_new_field': 'quokka secret', 'turn_number': 2}"),
]

BENIGN = [
    "2026-10-17 02:21:09,111 | INFO | vesper.voice | Voice input starting | wake_model=hey_jarvis | whisper=base.en (int8) | vad=webrtcvad | mic=default",
    "2026-10-17 02:21:10,652 | INFO | vesper.voice | voice input ready",
    "[02:32:00] WARNING PluginAgent: [PluginAgent] Attempted to stop agent in state STOPPED",
    "[02:32:00] INFO event_bus: [STOP] EventBus stopped | Published: 25 | Delivered: 43 | Failed: 0",
    "[02:40:11] INFO event_bus: [HANDLED] VoiceOutputEvent by _forward | Time: 0.4ms | Event ID: 3f9a1c2e-0b7d-4e55-9a61-8c2d4f6a1b09",
    "[02:40:11] INFO event_bus: [SUBSCRIBE] Handler 'on_x' subscribed to VoiceInputEvent (total handlers: 2)",
    "launcher: voice_output: died (signal SIGKILL (-9)); restarting in 1s",
    "2026-10-17 02:41:03,000 | INFO | vesper.voice | voice idle -> wake_detected",
    "2026-10-17 02:41:03,100 | INFO | vesper.voice | voice listening -> idle (no speech)",
    "2026-10-17 02:41:03,300 | INFO | vesper.voice | voice injected -> idle",
    "Traceback (most recent call last):",
]


@pytest.mark.parametrize("line", LEAKY_LINES)
def test_no_canary_survives_in_any_recognised_shape(line):
    out, changed = redact_line(line, Placeholders())
    assert changed is True
    low = out.lower()
    assert [c for c in CANARIES if c.lower() in low] == [], out
    assert "<text:" in out


def test_the_structure_of_a_payload_line_is_kept_so_the_log_is_still_readable():
    out, _ = redact_line(LEAKY_LINES[0], Placeholders())
    assert "Event: VoiceInputEvent" in out and "ID: 3f9a1c2e-0b7d-4e55-9a61-8c2d4f6a1b09" in out
    assert "Source: Brain" in out and "'confidence': 0.93" in out and "'source': 'voice'" in out
    out2, _ = redact_line(LEAKY_LINES[6], Placeholders())
    assert "'tool_name': 'search'" in out2 and "'success': True" in out2 and "'result': <text:1>" in out2
    out3, _ = redact_line(LEAKY_LINES[4], Placeholders())
    assert "'tool_name': 'gmail_create_draft'" in out3 and "'request_id': 'r-1'" in out3


@pytest.mark.parametrize("line", BENIGN)
def test_ordinary_lines_are_left_alone(line):
    out, changed = redact_line(line, Placeholders())
    assert out == line and changed is False


def test_the_same_text_gets_the_same_placeholder_and_different_text_a_different_one():
    ph = Placeholders()
    a1, _ = redact_line(emit("VoiceInputEvent", "{'text': 'play the glockenspiel rehearsal'}"), ph)
    b, _ = redact_line(emit("VoiceInputEvent", "{'text': 'call my dentist about the tangerine'}"), ph)
    a2, _ = redact_line(emit("VoiceOutputEvent", "{'text': 'Play the  glockenspiel rehearsal'}"), ph)      # case/space-insensitive
    n = lambda s: re.search(r"<text:(\d+)>", s).group(1)
    assert n(a1) == n(a2) and n(a1) != n(b)


def test_placeholder_numbers_carry_nothing_about_the_content():
    ph = Placeholders()
    outs = [redact_line(emit("VoiceInputEvent", "{'text': %r}" % t), ph)[0] for t in ("a", "a much longer sentence than the first", "zzz")]
    assert [re.search(r"<text:(\d+)>", o).group(1) for o in outs] == ["1", "2", "3"]       # first-appearance order only


def test_safe_keys_with_a_long_free_text_value_are_still_masked():
    line = emit("OddEvent", "{'source': 'the glockenspiel rehearsal moves to Thursday because quokka', 'priority': 'NORMAL'}")
    out, _ = redact_line(line, Placeholders())
    assert "glockenspiel" not in out and "'priority': 'NORMAL'" in out


def test_redact_lines_counts_changed_lines():
    out, changed = redact_lines(LEAKY_LINES[:3] + BENIGN[:2])
    assert changed == 3 and len(out) == 5 and out[3:] == BENIGN[:2]


# ------------------------------------------------------------------ the command

@pytest.fixture
def logdir(tmp_path, monkeypatch):
    d = tmp_path / "logs" / "launcher"
    d.mkdir(parents=True)
    (d / "gateway.log").write_text("\n".join(BENIGN[:2] + LEAKY_LINES[:4] + BENIGN[2:4]) + "\n")
    (d / "voice_input.log").write_text("\n".join(BENIGN[:2] + LEAKY_LINES[9:13]) + "\n")
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    return d


def run(capsys, *argv):
    code = cli.main(["logs", *argv])
    return code, capsys.readouterr().out


def test_without_the_flag_nothing_changes(logdir, capsys):
    code, out = run(capsys, "gateway", "--tail", "50")
    assert code == 0 and "glockenspiel" in out and "<text:" not in out


def test_with_the_flag_no_canary_reaches_the_output(logdir, capsys):
    code, out = run(capsys, "gateway", "--tail", "50", "--redact-transcripts")
    assert code == 0
    assert [c for c in CANARIES if c.lower() in out.lower()] == []
    assert "transcripts/message text replaced in 4 line(s)" in out and "<text:1>" in out
    assert "voice input ready" in out and "Attempted to stop agent" in out            # ordinary lines survive


def test_tail_limits_the_lines_and_all_components_share_placeholder_numbering(logdir, capsys):
    code, out = run(capsys, "--tail", "3", "--redact-transcripts")
    assert code == 0 and "===== gateway (last 3 lines" in out and "===== voice_input (last 3 lines" in out
    assert [c for c in CANARIES if c.lower() in out.lower()] == []


def test_short_option_n_still_works_and_tail_is_an_alias(logdir, capsys):
    _, a = run(capsys, "gateway", "-n", "2", "--redact-transcripts")
    _, b = run(capsys, "gateway", "--tail", "2", "--redact-transcripts")
    assert a == b and "last 2 lines" in a


def test_unknown_component_is_reported(logdir, capsys):
    code, out = run(capsys, "nope", "--redact-transcripts")
    assert code == 1 and "No log for 'nope'" in out


def test_the_log_files_themselves_are_not_modified(logdir, capsys):
    before = {p.name: p.read_bytes() for p in logdir.iterdir()}
    run(capsys, "--tail", "50", "--redact-transcripts")
    assert {p.name: p.read_bytes() for p in logdir.iterdir()} == before


def test_the_parser_accepts_the_new_options():
    ns = cli.build_parser().parse_args(["logs", "voice_output", "--tail", "7", "--redact-transcripts"])
    assert ns.lines == 7 and ns.redact_transcripts is True and ns.component == "voice_output"


@pytest.mark.parametrize("payload", [
    "{text=play the glockenspiel rehearsal, id=3}",                    # not a dict repr at all
    "{ 'weird' ; glockenspiel rehearsal ; quokka }",                    # nothing the pair parser understands
    "[('text', 'marzipan invoice')]",
])
def test_a_payload_the_parser_cannot_understand_is_masked_not_left_in_the_clear(payload):
    out, changed = redact_line(emit("VoiceInputEvent", payload), Placeholders())
    assert changed and [c for c in CANARIES if c.lower() in out.lower()] == [] and "<text:" in out
    assert "Event: VoiceInputEvent" in out
