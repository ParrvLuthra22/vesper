"""Channel-agnostic pieces: the normalized message, limits, confirmation nonces, credentials."""

from __future__ import annotations

import io
import logging

import pytest

from channels.confirm import ConfirmationBroker, Outcome
from channels.credentials import (CredentialError, TokenScrubFilter, install_token_scrubber,
                                  looks_like_bot_token, resolve_bot_token)
from channels.limits import LogThrottle, RateLimiter, split_reply
from channels.message import Attachment, InboundMessage, InvalidMessage, TRUST_THIRD_PARTY, TRUST_USER

TOKEN = "123456789:AAFakeTokenForTestsOnly_abcdefghijklmnopq"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


# ------------------------------------------------------------------ message

def test_message_round_trip():
    m = InboundMessage(text="hi", channel="telegram", user_id="42", attachments=(Attachment("photo", "image/jpeg", 10),))
    again = InboundMessage.from_dict(m.to_dict())
    assert again == m


@pytest.mark.parametrize("data", [
    {"text": "hi", "channel": "telegram", "user_id": "1"},                      # trust missing -> rejected, not defaulted
    {"text": "hi", "channel": "telegram", "user_id": "1", "trust": "admin"},    # unknown trust
    {"text": "hi", "channel": "telegram", "user_id": "1", "trust": ""},
    {"text": "", "channel": "telegram", "user_id": "1", "trust": "user"},
    {"text": "   ", "channel": "telegram", "user_id": "1", "trust": "user"},
    {"text": "hi", "channel": "Tele gram!", "user_id": "1", "trust": "user"},
    {"text": "hi", "channel": "telegram", "user_id": "", "trust": "user"},
    {"text": "x" * 20000, "channel": "telegram", "user_id": "1", "trust": "user"},
    {"text": ["hi"], "channel": "telegram", "user_id": "1", "trust": "user"},
    "not a dict", None,
])
def test_invalid_messages_are_rejected(data):
    with pytest.raises(InvalidMessage):
        InboundMessage.from_dict(data)


def test_forward_or_attachment_overrides_a_claimed_user_trust():
    plain = InboundMessage(text="do it", channel="telegram", user_id="1", trust=TRUST_USER)
    fwd = InboundMessage(text="do it", channel="telegram", user_id="1", trust=TRUST_USER, is_forward=True)
    att = InboundMessage(text="do it", channel="telegram", user_id="1", trust=TRUST_USER,
                         attachments=(Attachment("document"),))
    assert plain.effective_trust == TRUST_USER and plain.supplied_text == "do it"
    assert fwd.effective_trust == TRUST_THIRD_PARTY and fwd.supplied_text == ""
    assert att.effective_trust == TRUST_THIRD_PARTY


def test_third_party_supplied_text_is_only_the_typed_part():
    m = InboundMessage(text="typed + forwarded", channel="telegram", user_id="1", trust=TRUST_THIRD_PARTY,
                       trusted_text="typed")
    assert m.supplied_text == "typed"


def test_attachment_has_no_filename_and_clamps_unknown_kind():
    a = Attachment.from_dict({"kind": "rm -rf", "mime": "x" * 500, "size": -5, "file_name": "evil.pdf"})
    assert a.kind == "other" and len(a.mime) == 64 and a.size == 0 and "file_name" not in a.to_dict()


# ------------------------------------------------------------------ limits

def test_rate_limiter_window_and_per_user():
    clock = Clock()
    rl = RateLimiter(3, 60, clock)
    assert [rl.allow("a") for _ in range(4)] == [True, True, True, False]
    assert rl.allow("b") is True                      # another user is unaffected
    assert 0 < rl.retry_after("a") <= 60
    clock.advance(61)
    assert rl.allow("a") is True


def test_log_throttle_counts_suppressed():
    clock = Clock()
    lt = LogThrottle(60, clock)
    assert lt.ready("k") == (True, 0)
    assert lt.ready("k") == (False, 0) and lt.ready("k") == (False, 0)
    clock.advance(61)
    assert lt.ready("k") == (True, 2)
    assert lt.ready("other") == (True, 0)


def test_split_reply_short_is_untouched():
    assert split_reply("hello", 4000) == ["hello"]
    assert split_reply("", 4000) == ["(no reply)"]


def test_split_reply_respects_limit_numbers_parts_and_loses_nothing():
    paras = [f"Paragraph {i}. " + ("word " * 60).strip() for i in range(12)]
    text = "\n\n".join(paras)
    parts = split_reply(text, chunk_chars=600, max_total=10_000)
    assert len(parts) > 1 and all(len(p) <= 600 for p in parts)
    assert parts[0].startswith(f"(1/{len(parts)}) ") and parts[-1].startswith(f"({len(parts)}/{len(parts)}) ")
    body = " ".join(p.split(") ", 1)[1] for p in parts)
    assert body.split() == text.split()             # same words, same order, nothing dropped
    assert all(not p.split(") ", 1)[1].startswith(" ") for p in parts)


def test_split_reply_prefers_paragraph_then_sentence_boundaries():
    text = ("First sentence here. " * 20).strip() + "\n\n" + ("Second block of words. " * 20).strip()
    parts = split_reply(text, chunk_chars=500, max_total=5000)
    assert parts[0].rstrip().endswith("here.") or parts[0].rstrip().endswith(".")
    assert all(len(p) <= 500 for p in parts)


def test_split_reply_unbroken_word_is_hard_cut_and_truncation_is_explicit():
    parts = split_reply("x" * 5000, chunk_chars=500, max_total=20000)
    assert all(len(p) <= 500 for p in parts) and len(parts) >= 10
    long = split_reply("word " * 5000, chunk_chars=1000, max_total=3000)
    assert "truncated" in long[-1] and all(len(p) <= 1000 for p in long)


# ------------------------------------------------------------------ confirmation nonces

def test_nonce_approve_once_then_reused():
    clock = Clock()
    b = ConfirmationBroker(120, clock)
    n = b.issue("req-1", "42")
    r = b.resolve(n, "42", True)
    assert (r.outcome, r.request_id) == (Outcome.APPROVED, "req-1")
    assert b.resolve(n, "42", True).outcome == Outcome.REUSED
    assert b.resolve(n, "42", False).outcome == Outcome.REUSED      # the other button too


def test_nonce_deny():
    b = ConfirmationBroker(120, Clock())
    assert b.resolve(b.issue("r", "42"), "42", False).outcome == Outcome.DENIED


def test_nonce_expires_after_two_minutes():
    clock = Clock()
    b = ConfirmationBroker(120, clock)
    n = b.issue("r", "42")
    clock.advance(119)
    n2 = b.issue("r2", "42")
    assert b.resolve(n, "42", True).outcome == Outcome.APPROVED     # still inside the window
    clock.advance(121)
    assert b.resolve(n2, "42", True).outcome == Outcome.STALE
    assert b.resolve(n2, "42", True).outcome == Outcome.REUSED      # and it cannot be revived


def test_foreign_user_is_rejected_without_burning_the_nonce():
    b = ConfirmationBroker(120, Clock())
    n = b.issue("r", "42")
    assert b.resolve(n, "99", True).outcome == Outcome.FOREIGN
    assert b.resolve(n, "99", False).outcome == Outcome.FOREIGN
    assert b.resolve(n, "42", True).outcome == Outcome.APPROVED     # the owner can still answer


def test_unknown_nonce_and_uniqueness():
    b = ConfirmationBroker(120, Clock())
    assert b.resolve("nope", "42", True).outcome == Outcome.UNKNOWN
    nonces = {b.issue(f"r{i}", "42") for i in range(500)}
    assert len(nonces) == 500
    assert all(len(n) <= 40 for n in nonces)                         # fits Telegram's 64-byte callback_data


def test_nonce_collision_never_aliases_two_actions():
    seq = iter(["same", "same", "other"])
    b = ConfirmationBroker(120, Clock(), nonce_factory=lambda: next(seq))
    a, c = b.issue("r1", "1"), b.issue("r2", "1")
    assert a != c and b.resolve(a, "1", True).request_id == "r1" and b.resolve(c, "1", True).request_id == "r2"


def test_each_action_gets_its_own_nonce():
    b = ConfirmationBroker(120, Clock())
    n1, n2 = b.issue("r1", "42"), b.issue("r2", "42")
    assert b.resolve(n1, "42", True).request_id == "r1"
    assert b.resolve(n2, "42", True).request_id == "r2"


# ------------------------------------------------------------------ credentials

def test_token_shape():
    assert looks_like_bot_token(TOKEN) and not looks_like_bot_token("hunter2")


def test_token_resolution_order_and_errors():
    env = {"MY_TOKEN": TOKEN}
    other = "987654321:AAOtherFakeTokenForTests_zyxwvutsrqponm"
    assert resolve_bot_token("MY_TOKEN", "svc", other, env=env, keychain=lambda s: None) == (TOKEN, "env")
    assert resolve_bot_token("MY_TOKEN", "svc", other, env={}, keychain=lambda s: other) == (other, "keychain")
    assert resolve_bot_token("MY_TOKEN", "svc", TOKEN, env={}, keychain=lambda s: None) == (TOKEN, "config")
    with pytest.raises(CredentialError) as exc:
        resolve_bot_token("MY_TOKEN", "svc", "", env={}, keychain=lambda s: None)
    assert "MY_TOKEN" in str(exc.value)
    with pytest.raises(CredentialError) as exc:
        resolve_bot_token("MY_TOKEN", "svc", "", env={"MY_TOKEN": "hunter2-not-a-token"}, keychain=lambda s: None)
    assert "hunter2" not in str(exc.value)             # a malformed value is never echoed


def test_scrub_filter_masks_token_in_message_args_and_exception_text():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(TokenScrubFilter(TOKEN))
    log = logging.getLogger("scrub-test")
    log.propagate = False
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.info("calling https://api.telegram.org/bot%s/getUpdates", TOKEN)
    log.info("token=" + TOKEN)
    try:
        raise RuntimeError(f"boom https://api.telegram.org/bot{TOKEN}/sendMessage")
    except RuntimeError:
        log.exception("failed")
    # a different token-shaped URL segment is masked by shape, even if it is not OUR token
    log.info("https://api.telegram.org/bot111111111:AAAnotherShapeOnlyxxxxxxxxxxxxxxxxxx/x")
    out = stream.getvalue()
    assert TOKEN not in out and "AAFakeToken" not in out and "AAAnotherShape" not in out
    assert out.count("<bot-token>") >= 4


def test_install_token_scrubber_quiets_http_loggers():
    saved = {n: logging.getLogger(n).level for n in ("httpx", "httpcore", "hpack", "urllib3", "asyncio")}
    flt = install_token_scrubber(TOKEN)
    try:
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
    finally:                                   # global logging state: put it back
        logging.getLogger().removeFilter(flt)
        for h in logging.getLogger().handlers:
            h.removeFilter(flt)
        for n, lvl in saved.items():
            logging.getLogger(n).setLevel(lvl)
