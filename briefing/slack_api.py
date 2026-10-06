"""
The SlackCollector's only way to reach Slack — and it can only read.

Same idea as briefing/readonly.py (the Gmail/Calendar guard), one level lower: there is no tool
registry for Slack, so the collector gets a `SlackReadOnlyClient`, and `call()` refuses every
Slack method that is not on SLACK_READ_METHODS *before* any request is built. Posting, reacting,
marking read, joining, pinning, uploading... are unreachable from this package; tests assert that
(allow-list contents, a refusal per mutating method, no request on refusal, a source scan).

Credentials: a Slack USER token (xoxp-...) with read-only scopes, from the environment or the macOS
Keychain. It is sent only as an Authorization header, never in a URL, and nothing here logs it;
`scrub()` masks any token-shaped string in error text as a backstop.
"""

from __future__ import annotations

import os
import re
from typing import Any, Awaitable, Callable, Dict, Optional

from briefing.readonly import ReadOnlyViolation

#: Every Slack Web API method the briefing may ever call. All are pure reads.
SLACK_READ_METHODS = frozenset({
    "auth.test",              # who am I (user id) — no scope
    "users.conversations",    # my DMs / group DMs        — im:read, mpim:read
    "conversations.info",     # last_read marker           — im:read, mpim:read, channels:read
    "conversations.history",  # the messages               — im:history, mpim:history, channels:history
    "users.info",             # display name, bot flag     — users:read
    "search.messages",        # @mentions of me            — search:read (optional)
})

API_BASE = "https://slack.com/api/"
_METHOD_SHAPE = re.compile(r"^[a-z]+(?:\.[a-zA-Z]+)+$")
_TOKEN_SHAPE = re.compile(r"xox[a-z]-[A-Za-z0-9-]{8,}")

DEFAULT_TOKEN_ENV = "VESPER_SLACK_USER_TOKEN"
DEFAULT_KEYCHAIN_SERVICE = "vesper-slack-user"
KEYCHAIN_ACCOUNT = "user-token"


class SlackNotConfigured(Exception):
    """No usable token. Not an error: the briefing simply has no Slack section."""


class SlackApiError(Exception):
    """A Slack error code (e.g. 'invalid_auth', 'missing_scope', 'ratelimited'). Never carries a token
    or message text — only the code Slack returned."""

    def __init__(self, code: str, retry_after: Optional[float] = None):
        super().__init__(code)
        self.code = code
        self.retry_after = retry_after


def scrub(text: object) -> str:
    """Mask anything token-shaped. A backstop for error text; nothing is meant to log a token."""
    return _TOKEN_SHAPE.sub("<slack-token>", str(text))


# ------------------------------------------------------------------ credentials

def resolve_slack_token(
    env_name: str = DEFAULT_TOKEN_ENV,
    keychain_service: str = DEFAULT_KEYCHAIN_SERVICE,
    env: Optional[Dict[str, str]] = None,
    keychain: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    """The user token, from $env then the Keychain. Raises SlackNotConfigured (no token anywhere) or
    SlackApiError('bad_token_format') — never with the value in the message."""
    env = os.environ if env is None else env
    if keychain is None:
        from channels.credentials import keychain_lookup

        def keychain(service: str) -> Optional[str]:  # noqa: F811 — the default lookup
            return keychain_lookup(service, KEYCHAIN_ACCOUNT)

    for value in ((env.get(env_name) or "").strip() if env_name else "",
                  (keychain(keychain_service) or "").strip() if keychain_service else ""):
        if value:
            # Only a USER token: a bot token (xoxb-) sees different things and is not what this is for.
            if not value.startswith("xoxp-"):
                raise SlackApiError("bad_token_format")
            return value
    raise SlackNotConfigured(
        f"no Slack user token: set ${env_name} or store one in the Keychain "
        f"(security add-generic-password -s {keychain_service} -a {KEYCHAIN_ACCOUNT} -w)")


# -------------------------------------------------------------------- transport

Transport = Callable[[str, Dict[str, Any], str], Awaitable[Dict[str, Any]]]


async def http_transport(method: str, params: Dict[str, Any], token: str) -> Dict[str, Any]:
    """POST https://slack.com/api/<method>. Only ever reached through SlackReadOnlyClient.call()."""
    import aiohttp

    form = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in params.items() if v is not None}
    try:
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(API_BASE + method, data=form,
                                    headers={"Authorization": f"Bearer {token}"}) as resp:
                if resp.status == 429:
                    raise SlackApiError("ratelimited", float(resp.headers.get("Retry-After", "30") or 30))
                payload = await resp.json(content_type=None)
    except SlackApiError:
        raise
    except Exception as exc:   # transport text may quote the request: keep only the type
        raise SlackApiError(f"transport_error:{type(exc).__name__}") from None
    if not isinstance(payload, dict):
        raise SlackApiError("bad_response")
    if not payload.get("ok"):
        raise SlackApiError(str(payload.get("error") or "unknown_error"))
    return payload


class SlackReadOnlyClient:
    """What a collector holds instead of a Slack SDK. `call()` is the single choke point."""

    def __init__(self, token: str, transport: Optional[Transport] = None):
        self._token = token
        self._transport: Transport = transport or http_transport
        self.calls = 0

    def __repr__(self) -> str:   # never show the token
        return "SlackReadOnlyClient(token=<hidden>)"

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not isinstance(method, str) or not _METHOD_SHAPE.match(method) or method not in SLACK_READ_METHODS:
            raise ReadOnlyViolation(f"'{scrub(method)}' is not an allow-listed read-only Slack method")
        self.calls += 1
        return await self._transport(method, dict(params or {}), self._token)
