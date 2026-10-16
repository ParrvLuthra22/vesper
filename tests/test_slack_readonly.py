"""The Slack read-only guard and credential handling. No network: every call goes to a fake transport.

Mirrors the Gmail/Calendar guard tests in test_briefing_collectors.py: an allow-list that contains only
reads, a refusal for every mutating method (and no request is ever built for it), and a source scan."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from briefing import slack as slack_module
from briefing import slack_api
from briefing.readonly import ReadOnlyViolation
from briefing.slack_api import (
    SLACK_READ_METHODS, SlackApiError, SlackNotConfigured, SlackReadOnlyClient, resolve_slack_token, scrub)

FAKE_TOKEN = "xoxp-0000000000-fake-fixture-token"

MUTATING = [
    "chat.postMessage", "chat.postEphemeral", "chat.update", "chat.delete", "chat.meMessage", "chat.scheduleMessage",
    "reactions.add", "reactions.remove", "conversations.mark", "conversations.join", "conversations.leave",
    "conversations.open", "conversations.close", "conversations.create", "conversations.invite",
    "conversations.kick", "conversations.archive", "conversations.rename", "conversations.setTopic",
    "im.mark", "im.open", "im.close", "channels.mark", "pins.add", "pins.remove", "stars.add", "files.upload",
    "files.delete", "files.remote.add", "users.setPresence", "users.profile.set", "dnd.setSnooze", "reminders.add",
    "usergroups.create", "views.open", "views.publish", "dialog.open", "bookmarks.add", "calls.add",
    "files.getUploadURLExternal", "files.completeUploadExternal", "emoji.list", "admin.users.remove",
    "oauth.v2.access", "auth.revoke", "apps.uninstall", "team.info",
]


class Recorder:
    def __init__(self):
        self.requests = []

    async def __call__(self, method, params, token):
        self.requests.append((method, params, token))
        return {"ok": True}


def test_allow_list_contains_only_reads():
    assert SLACK_READ_METHODS == {"auth.test", "users.conversations", "conversations.info",
                                  "conversations.history", "users.info", "search.messages"}
    for bad in ("post", "update", "delete", "add", "remove", "mark", "join", "leave", "open", "close", "create",
                "invite", "kick", "archive", "set", "upload", "react", "revoke", "write"):
        assert not any(bad in m.lower() for m in SLACK_READ_METHODS), bad


@pytest.mark.asyncio
@pytest.mark.parametrize("method", MUTATING)
async def test_every_mutating_method_is_refused_before_a_request_is_built(method):
    rec = Recorder()
    client = SlackReadOnlyClient(FAKE_TOKEN, transport=rec)
    with pytest.raises(ReadOnlyViolation, match="not an allow-listed read-only Slack method"):
        await client.call(method, {"channel": "C1", "text": "hello"})
    assert rec.requests == [] and client.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["", "chat", "chat.postMessage ", "conversations.history/../chat.postMessage",
                                    "https://slack.com/api/chat.postMessage", "CONVERSATIONS.HISTORY",
                                    "conversations.history?x=1", None, 7])
async def test_malformed_or_smuggled_method_names_are_refused(method):
    rec = Recorder()
    with pytest.raises(ReadOnlyViolation):
        await SlackReadOnlyClient(FAKE_TOKEN, transport=rec).call(method, {})
    assert rec.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method", sorted(SLACK_READ_METHODS))
async def test_allow_listed_methods_pass_with_the_token_only_as_a_separate_argument(method):
    rec = Recorder()
    await SlackReadOnlyClient(FAKE_TOKEN, transport=rec).call(method, {"limit": 5})
    (m, params, token), = rec.requests
    assert m == method and params == {"limit": 5} and token == FAKE_TOKEN
    assert FAKE_TOKEN not in str(params)           # the token is never part of the request parameters


def test_the_client_never_shows_its_token():
    c = SlackReadOnlyClient(FAKE_TOKEN)
    assert FAKE_TOKEN not in repr(c) and FAKE_TOKEN not in str(vars(c).get("calls"))


def test_slack_sources_never_mention_a_mutating_method():
    for module in (slack_module, slack_api):
        src = Path(module.__file__).read_text()
        code = re.sub(r'""".*?"""', "", src, flags=re.S)
        for word in ("chat.", "postMessage", "reactions.", "conversations.mark", "conversations.join",
                     "conversations.open", "conversations.close", "im.mark", "files.upload", "pins.", "stars.",
                     "setPresence", "users.profile", "conversations.create", "conversations.invite"):
            assert word not in code, (module.__name__, word)


def test_the_collector_reaches_slack_only_through_the_guarded_client():
    """The only module that builds a request is slack_api.http_transport, and the only caller of a
    transport is SlackReadOnlyClient.call (which validates the method first)."""
    api_src = re.sub(r'""".*?"""', "", Path(slack_api.__file__).read_text(), flags=re.S)
    assert api_src.count("self._transport(") == 1
    assert api_src.count("session.post(") == 1
    coll_src = Path(slack_module.__file__).read_text()
    for forbidden in ("aiohttp", "requests", "urllib", "http_transport", "slack_sdk", "WebClient", "subprocess"):
        assert forbidden not in coll_src, forbidden


# ------------------------------------------------------------------ credentials

def test_no_token_anywhere_is_not_configured_not_an_error():
    with pytest.raises(SlackNotConfigured):
        resolve_slack_token(env={}, keychain=lambda service: None)


def test_token_from_env_then_keychain():
    assert resolve_slack_token(env={"VESPER_SLACK_USER_TOKEN": FAKE_TOKEN}, keychain=lambda s: "xoxp-other-fake-1234") \
        == FAKE_TOKEN
    seen = []

    def keychain(service):
        seen.append(service)
        return FAKE_TOKEN

    assert resolve_slack_token(env={}, keychain=keychain) == FAKE_TOKEN
    assert seen == ["vesper-slack-user"]


@pytest.mark.parametrize("bad", ["xoxb-1111111111-fake-bot-token", "not-a-token", "xoxe-1234567890-fake"])
def test_only_a_user_token_is_accepted_and_the_message_never_contains_it(bad):
    with pytest.raises(SlackApiError) as exc:
        resolve_slack_token(env={"VESPER_SLACK_USER_TOKEN": bad}, keychain=lambda s: None)
    assert exc.value.code == "bad_token_format" and bad not in str(exc.value)


def test_scrub_masks_token_shaped_strings():
    text = f"boom: Authorization xoxp-1234567890-abcdefghij-secret and {FAKE_TOKEN} and xoxb-99999999-zzzzzzzz"
    out = scrub(text)
    assert "xoxp-" not in out and "xoxb-" not in out and "secret" not in out and out.count("<slack-token>") == 3


@pytest.mark.asyncio
async def test_transport_errors_carry_only_a_code(monkeypatch):
    import aiohttp

    class Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise RuntimeError(f"connect failed for Bearer {FAKE_TOKEN}")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(aiohttp, "ClientSession", Boom)
    with pytest.raises(SlackApiError) as exc:
        await SlackReadOnlyClient(FAKE_TOKEN).call("auth.test")
    assert FAKE_TOKEN not in str(exc.value) and FAKE_TOKEN not in repr(exc.value)
    assert exc.value.code == "transport_error:RuntimeError"
