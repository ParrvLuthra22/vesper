"""Where the bot token comes from — and making sure it never reaches a log.

Order: the environment variable named in config, then the macOS Keychain, then (discouraged) a
literal in config. The value is returned to the caller and nowhere else: nothing here logs it, and
`TokenScrubFilter` is installed on every log handler of the channel process as a backstop (an HTTP
client's own log line or exception text can contain the request URL, and Telegram's URL contains the token).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Callable, Dict, Optional

_TOKEN_SHAPE = re.compile(r"^\d{5,16}:[A-Za-z0-9_-]{20,}$")


class CredentialError(Exception):
    """Raised without the credential in the message."""


def keychain_lookup(service: str, account: str = "bot-token") -> Optional[str]:
    """`security find-generic-password` — argv list, no shell. None if absent or not macOS."""
    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def looks_like_bot_token(value: str) -> bool:
    return bool(_TOKEN_SHAPE.match(value or ""))


def resolve_bot_token(
    env_name: str = "VESPER_TELEGRAM_BOT_TOKEN",
    keychain_service: str = "vesper-telegram-bot",
    config_value: str = "",
    env: Optional[Dict[str, str]] = None,
    keychain: Callable[[str], Optional[str]] = keychain_lookup,
) -> tuple:
    """(token, where) or raises CredentialError. `where` is "env" | "keychain" | "config" — safe to log."""
    env = os.environ if env is None else env
    candidates = (
        ("env", (env.get(env_name) or "").strip() if env_name else ""),
        ("keychain", (keychain(keychain_service) or "").strip() if keychain_service else ""),
        ("config", (config_value or "").strip()),
    )
    for where, value in candidates:
        if value:
            if not looks_like_bot_token(value):
                raise CredentialError(f"the bot token from {where} is not in Telegram's format (<digits>:<secret>)")
            return value, where
    raise CredentialError(
        f"no bot token: set ${env_name} or store one in the Keychain "
        f"(security add-generic-password -s {keychain_service} -a bot-token -w)")


class TokenScrubFilter(logging.Filter):
    """Replaces the token (and any `bot<token>` URL segment) in every record, including formatted
    exception text, before a handler sees it. A backstop: nothing is meant to log the token."""

    def __init__(self, token: str):
        super().__init__()
        self._token = token

    def scrub(self, text: str) -> str:
        if self._token and self._token in text:
            text = text.replace(self._token, "<bot-token>")
        return re.sub(r"/bot\d{5,16}:[A-Za-z0-9_-]{10,}", "/bot<bot-token>", text)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg, record.args = self.scrub(message), None
        if record.exc_info:
            import traceback

            record.exc_text = self.scrub("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = self.scrub(record.exc_text)
        return True


def install_token_scrubber(token: str) -> TokenScrubFilter:
    """Attach the scrubber to every handler on the root logger, quiet the HTTP client loggers (they
    log URLs at INFO), and return it."""
    flt = TokenScrubFilter(token)
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(flt)
    root.addFilter(flt)
    for name in ("httpx", "httpcore", "hpack", "urllib3", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return flt
