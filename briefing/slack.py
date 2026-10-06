"""
SlackCollector — unread DMs, group DMs and @mentions as briefing Items. READ-ONLY.

    fetch_since(cursor)   unread messages newer than the cursor (and than Slack's own read marker)
    live_ids()            which cached Slack items are still unread, so read ones drop out
    available()           False (-> "not configured") when no user token exists; the briefing is
                          unaffected: an unconfigured Slack has no section and no staleness note

Contract (same as Gmail/Calendar, see briefing/collectors.py): it reaches Slack ONLY through a
SlackReadOnlyClient, whose allow-list is six read methods. It never posts, reacts, joins or marks
anything read. Everything it returns is third-party text: truncated hard at collection, neutralised
(markup, links, addresses, shell substitutions), and flagged `hostile` when it reads like instructions
to an AI — such an item stores NO snippet and is demoted by the scorer.

What it maps:
    DM / group DM from someone else, newer than my read marker      -> source "slack", kind dm|group_dm
    a channel message that @mentions me (via search.messages)       -> kind "mention"
    ordinary channel chatter, my own messages, muted channels       -> ignored
"""

from __future__ import annotations

import html
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from briefing.collectors import Collector
from briefing.config import BriefingConfig
from briefing.items import Item
from briefing.sanitize import clean_text, looks_like_instructions
from briefing.slack_api import (
    SlackApiError, SlackNotConfigured, SlackReadOnlyClient, resolve_slack_token, scrub)
from utils.logger import get_logger

logger = get_logger(__name__)

#: Message subtypes worth reading. Everything else (joins, leaves, topic changes, edits...) is noise.
_READABLE_SUBTYPES = {None, "thread_broadcast", "bot_message", "file_share", "me_message"}
#: Slack error codes that mean "this whole run is unusable" (vs. one conversation being gone).
_FATAL = {"invalid_auth", "not_authed", "token_revoked", "token_expired", "account_inactive", "missing_scope",
          "ratelimited", "bad_response", "not_allowed_token_type", "access_denied"}
_SYSTEM_BOTS = {"USLACKBOT"}

_LINK = re.compile(r"<(?:https?://|mailto:)[^|>]*\|([^>]*)>", re.IGNORECASE)
_BARE_LINK = re.compile(r"<(?:https?://|mailto:)[^>]*>", re.IGNORECASE)
_USER_REF = re.compile(r"<@[UW][A-Z0-9]+(?:\|[^>]*)?>")
_CHAN_REF = re.compile(r"<#[CG][A-Z0-9]+\|([^>]*)>")
_SPECIAL = re.compile(r"<!(here|channel|everyone)(?:\|[^>]*)?>")
_SUBTEAM = re.compile(r"<!subteam\^[A-Z0-9]+(?:\|([^>]*))?>")
_SHELL_SUBST = re.compile(r"\$\s*([({])")
#: Code-execution shapes that have no business in a chat message addressed to an assistant.
_SHELL_PAYLOAD = re.compile(
    r"\$\s*[({]|`[^`]*`|\b(?:curl|wget)\b[^\n]{0,80}\|\s*(?:sh|bash|zsh)\b|\|\s*(?:sh|bash|zsh)\b"
    r"|;\s*(?:rm|curl|wget|sudo|chmod|osascript)\b|&&\s*(?:rm|curl|wget|sudo)\b|\brm\s+-rf?\b|\bosascript\b",
    re.IGNORECASE)


def slack_plain(text: object) -> str:
    """Slack mrkdwn -> plain text, before the shared `clean_text`: mentions become "@user" (never resolved),
    links keep their label, and shell substitutions are defused ("$(" -> "$ (")."""
    s = html.unescape(str(text or ""))
    s = _LINK.sub(r"\1", s)
    s = _BARE_LINK.sub(" [link] ", s)
    s = _USER_REF.sub("@user", s)
    s = _CHAN_REF.sub(r"\1", s)
    s = _SPECIAL.sub(r"@\1", s)
    s = _SUBTEAM.sub(lambda m: "@" + (m.group(1) or "group").lstrip("@"), s)
    s = _SHELL_SUBST.sub(r"$ \1", s)
    return s


def hostile_text(*texts: object) -> bool:
    """Reads like instructions to an AI, or carries a shell payload."""
    for t in texts:
        raw = str(t or "")
        if looks_like_instructions(raw, slack_plain(raw)) or _SHELL_PAYLOAD.search(raw):
            return True
    return False


def _ts(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class SlackCollector(Collector):
    name = "slack"
    optional = True        # not configured => skipped silently, never reported as a failure
    #: seconds of overlap when advancing the cursor, so a message that lands during a fetch is not missed
    OVERLAP_SECONDS = 120
    #: how long a "no token" answer is trusted before the Keychain/env is consulted again
    RECHECK_SECONDS = 60

    def __init__(
        self,
        cfg: BriefingConfig,
        client: Optional[SlackReadOnlyClient] = None,
        *,
        cached: Optional[Callable[[], List[Item]]] = None,
        token_resolver: Optional[Callable[[], str]] = None,
        clock: Callable[[], float] = time.time,
    ):
        self._cfg = cfg
        self._client = client
        self._cached = cached
        self._resolve = token_resolver or (lambda: resolve_slack_token(
            cfg.slack_token_env, cfg.slack_token_keychain_service))
        self._clock = clock
        self._missing_until = 0.0
        self._me: Optional[str] = None
        self._users: Dict[str, tuple] = {}
        self._last_read: Dict[str, float] = {}
        self.degraded: List[str] = []       # e.g. ["mentions: missing_scope"] — codes only, never content
        self.not_configured_reason: Optional[str] = None

    # ------------------------------------------------------------ configuration
    def _get_client(self) -> Optional[SlackReadOnlyClient]:
        if self._client is not None:
            return self._client
        if self._clock() < self._missing_until:
            return None
        try:
            self._client = SlackReadOnlyClient(self._resolve())
            self.not_configured_reason = None
        except SlackNotConfigured as exc:
            self._missing_until = self._clock() + self.RECHECK_SECONDS
            self.not_configured_reason = "not configured"
            logger.debug(f"[Briefing] slack: {exc}")
        except SlackApiError as exc:
            self._missing_until = self._clock() + self.RECHECK_SECONDS
            self.not_configured_reason = f"not configured ({exc.code})"
        return self._client

    def available(self) -> bool:
        return self._get_client() is not None

    # ------------------------------------------------------------------- fetch
    async def fetch_since(self, cursor: Optional[str]) -> List[Item]:
        client = self._get_client()
        if client is None:
            raise SlackNotConfigured(self.not_configured_reason or "not configured")
        now = self._clock()
        floor = now - self._cfg.slack_lookback_days * 86400
        try:
            since = max(floor, float(cursor) - self.OVERLAP_SECONDS) if cursor else floor
        except ValueError:
            since = floor
        self._last_read = {}
        self.degraded = []
        me = await self._whoami(client)

        items: Dict[str, Item] = {}
        convs = await self._conversations(client)
        for conv in convs:
            cid = str(conv.get("id") or "")
            if not cid or self._is_muted(cid, conv.get("name")) or conv.get("is_user_deleted"):
                continue
            kind = "group_dm" if conv.get("is_mpim") else "dm"
            last_read = await self._read_marker(client, cid)
            if last_read is None:
                continue
            oldest = max(since, last_read)
            try:
                hist = await client.call("conversations.history", {
                    "channel": cid, "oldest": f"{oldest:.6f}", "limit": self._cfg.slack_max_messages})
            except SlackApiError as exc:
                if exc.code in _FATAL or exc.code.startswith("transport_error"):
                    raise
                continue
            for m in hist.get("messages") or []:
                item = await self._map_message(client, m, cid, kind, me, last_read, oldest, None)
                if item is not None:
                    items[item.id] = item

        if self._cfg.slack_mentions_via_search:
            for item in await self._mentions(client, me, since):
                items.setdefault(item.id, item)
        return sorted(items.values(), key=lambda i: -i.timestamp)

    async def _whoami(self, client: SlackReadOnlyClient) -> str:
        if self._me is None:
            res = await client.call("auth.test")
            self._me = str(res.get("user_id") or "")
            if not self._me:
                raise SlackApiError("bad_response")
        return self._me

    async def _conversations(self, client: SlackReadOnlyClient) -> List[Dict[str, Any]]:
        res = await client.call("users.conversations", {
            "types": "im,mpim", "exclude_archived": True, "limit": min(200, self._cfg.slack_max_conversations)})
        return list(res.get("channels") or [])[: self._cfg.slack_max_conversations]

    async def _read_marker(self, client: SlackReadOnlyClient, channel: str) -> Optional[float]:
        """Slack's own 'read up to here' timestamp for me. None if the conversation can't be read."""
        try:
            res = await client.call("conversations.info", {"channel": channel})
        except SlackApiError as exc:
            if exc.code in _FATAL or exc.code.startswith("transport_error"):
                raise
            return None
        marker = _ts((res.get("channel") or {}).get("last_read"))
        self._last_read[channel] = marker
        return marker

    def _is_muted(self, channel_id: str, name: Any = None) -> bool:
        muted = set(self._cfg.slack_muted_channels)
        return channel_id.lower() in muted or (bool(name) and str(name).lower().lstrip("#") in muted)

    # ------------------------------------------------------------------ mapping
    async def _person(self, client: SlackReadOnlyClient, m: Dict[str, Any]) -> tuple:
        """(display name, is_bot, allowlisted) for a message's author."""
        bot_id, user_id = m.get("bot_id"), m.get("user")
        if m.get("subtype") == "bot_message" or (bot_id and not user_id):
            name = str(m.get("username") or (m.get("bot_profile") or {}).get("name") or "a bot")
            return name, True, self._allowlisted(bot_id, user_id, name)
        user_id = str(user_id or "")
        if user_id not in self._users:
            name, is_bot = "someone", user_id in _SYSTEM_BOTS
            try:
                u = (await client.call("users.info", {"user": user_id})).get("user") or {}
                prof = u.get("profile") or {}
                name = prof.get("display_name") or prof.get("real_name") or u.get("real_name") or u.get("name") or name
                is_bot = bool(u.get("is_bot") or u.get("is_app_user")) or user_id in _SYSTEM_BOTS
            except SlackApiError as exc:
                if exc.code in _FATAL or exc.code.startswith("transport_error"):
                    raise
            self._users[user_id] = (str(name), is_bot)
        name, is_bot = self._users[user_id]
        return name, is_bot, self._allowlisted(bot_id, user_id, name)

    def _allowlisted(self, *ids_or_names: Any) -> bool:
        allow = set(self._cfg.slack_allowlisted_senders)
        return any(str(v).lower() in allow for v in ids_or_names if v)

    async def _map_message(self, client, m: Dict[str, Any], channel: str, kind: str, me: str, last_read: float,
                           oldest: float, channel_name: Optional[str]) -> Optional[Item]:
        if m.get("type", "message") != "message" or m.get("subtype") not in _READABLE_SUBTYPES:
            return None
        if m.get("user") == me or m.get("hidden"):
            return None
        ts = _ts(m.get("ts"))
        if ts <= 0 or ts <= last_read or ts <= oldest:
            return None
        raw_text = str(m.get("text") or "")
        if not raw_text.strip() and not m.get("files"):
            return None
        name, is_bot, allowlisted = await self._person(client, m)
        mention = f"<@{me}>" in raw_text or f"<@{me}|" in raw_text
        if kind == "mention":
            mention = True
        hostile = hostile_text(raw_text, name)
        thread_ts = str(m.get("thread_ts") or "")
        is_reply = bool(thread_ts) and thread_ts != str(m.get("ts"))
        if kind == "mention":
            title = f"mention in {clean_text(channel_name or 'a channel', 40)}"
        else:
            title = "group message" if kind == "group_dm" else "direct message"
        snippet = "" if hostile else clean_text(slack_plain(raw_text) or "(attachment)", self._cfg.slack_snippet_chars)
        return Item(
            id=f"slack:{channel}:{m.get('ts')}",
            source="slack",
            sender=clean_text(slack_plain(name), 32) or "someone",
            title=title,
            snippet=snippet,
            timestamp=ts,
            thread_id=f"{channel}:{thread_ts}" if thread_ts else "",
            is_reply=is_reply,
            unread=True,
            signals={
                "kind": kind, "channel": channel, "mention": mention, "bot": is_bot, "allowlisted": allowlisted,
                "reply_to_me": bool(is_reply and m.get("parent_user_id") == me), "hostile": hostile,
            },
        )

    async def _mentions(self, client: SlackReadOnlyClient, me: str, since: float) -> List[Item]:
        day = datetime.fromtimestamp(since - 86400, tz=timezone.utc).strftime("%Y-%m-%d")
        try:
            res = await client.call("search.messages", {
                "query": f"<@{me}> after:{day}", "sort": "timestamp", "sort_dir": "desc",
                "count": self._cfg.slack_max_mentions})
        except SlackApiError as exc:
            if exc.code in ("missing_scope", "not_allowed_token_type"):
                # DMs still work without search:read; say so loudly instead of silently dropping mentions.
                self.degraded.append(f"mentions: {exc.code} (search:read)")
                logger.warning(f"[Briefing] slack mentions unavailable: {exc.code} — add the search:read scope")
                return []
            if exc.code in _FATAL or exc.code.startswith("transport_error"):
                raise
            return []
        out: List[Item] = []
        for m in ((res.get("messages") or {}).get("matches") or []):
            ch = m.get("channel") or {}
            cid = str(ch.get("id") or "")
            if not cid or ch.get("is_im") or ch.get("is_mpim") or ch.get("is_private"):
                continue          # DMs are covered above; private channels are out of scope (channels:history only)
            if self._is_muted(cid, ch.get("name")):
                continue
            last_read = self._last_read.get(cid)
            if last_read is None:
                last_read = await self._read_marker(client, cid)
            if last_read is None:
                continue          # can't tell whether it was read: skip rather than nag about an old mention
            item = await self._map_message(client, m, cid, "mention", me, last_read, since, str(ch.get("name") or ""))
            if item is not None:
                out.append(item)
        return out

    # --------------------------------------------------------------- read state
    async def live_ids(self) -> Optional[Set[str]]:
        """Cached Slack items that are still unread: newer than my read marker in their conversation,
        inside the lookback window, and not in a channel I have since muted. Without a cache accessor
        nothing can be reconciled, so nothing is retired (None)."""
        if self._cached is None:
            return None
        floor = self._clock() - self._cfg.slack_lookback_days * 86400
        live: Set[str] = set()
        for it in self._cached():
            channel = str((it.signals or {}).get("channel") or "")
            if it.timestamp <= floor or self._is_muted(channel):
                continue
            marker = self._last_read.get(channel)
            if marker is not None and it.timestamp <= marker:
                continue          # read in Slack (on any device)
            live.add(it.id)
        return live
