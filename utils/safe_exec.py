"""
Safe process execution helpers.

Rule: model- or content-derived text is never interpolated into a shell string
or an AppleScript source string. It travels as a discrete argv element —

  - external commands:  ``run_argv(["open", url])``        (subprocess, shell=False)
  - AppleScript values: ``run_osascript(SCRIPT, value)``   (``on run argv`` — the
    value is handed to osascript as an argument, never spliced into the source)

so no quote, ``$(...)``, backtick or ``;`` in a value can change what runs.
"""

from __future__ import annotations

import re
import subprocess
from typing import Optional, Sequence, Tuple
from urllib.parse import urlparse

OSASCRIPT = "/usr/bin/osascript"
DEFAULT_TIMEOUT = 10.0

MAX_APP_NAME_LEN = 100
MAX_URL_LEN = 2048

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
# "scheme:" prefix that is not a "host:port" (digits after the colon).
_SCHEME_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:(?!\d)")


class UnsafeInputError(ValueError):
    """Raised when a value fails validation and must not reach a process."""


def validate_app_name(name: str) -> str:
    """Return a cleaned application name, or raise UnsafeInputError.

    Names are always passed as argv, so quoting is not the concern here; the
    checks stop control characters and option-looking names (``-9``) reaching
    ``open -a`` / ``killall``.
    """
    if not isinstance(name, str):
        raise UnsafeInputError("application name must be text")
    name = name.strip().rstrip(".").strip()
    if not name:
        raise UnsafeInputError("application name is empty")
    if len(name) > MAX_APP_NAME_LEN:
        raise UnsafeInputError("application name is too long")
    if _CONTROL_CHARS.search(name):
        raise UnsafeInputError("application name contains control characters")
    if name.startswith("-"):
        raise UnsafeInputError("application name must not start with '-'")
    return name


def validate_http_url(raw: str) -> str:
    """Return a normalized http(s) URL, or raise UnsafeInputError.

    A bare host ("example.com/x") gets ``https://``. Any other scheme
    (``file:``, ``mailto:``, ``shortcuts:``, ``x-apple.systempreferences:`` ...)
    is refused — ``open`` would hand those to arbitrary apps.
    """
    if not isinstance(raw, str):
        raise UnsafeInputError("URL must be text")
    url = raw.strip()
    if not url:
        raise UnsafeInputError("URL is empty")
    if len(url) > MAX_URL_LEN:
        raise UnsafeInputError("URL is too long")
    if _CONTROL_CHARS.search(url) or re.search(r"\s", url):
        raise UnsafeInputError("URL contains whitespace or control characters")
    if "://" not in url:
        if _SCHEME_PREFIX.match(url):
            raise UnsafeInputError("only http and https URLs are allowed")
        url = f"https://{url}"
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise UnsafeInputError("only http and https URLs are allowed")
    if not parsed.hostname:
        raise UnsafeInputError("URL has no host")
    return url


def run_argv(
    argv: Sequence[str], timeout: float = DEFAULT_TIMEOUT
) -> Tuple[bool, str, str]:
    """Run ``argv`` with shell=False. Returns (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout, shell=False
        )
    except subprocess.TimeoutExpired:
        return False, "", f"timed out after {timeout:g}s"
    except (OSError, ValueError) as exc:
        return False, "", str(exc)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def run_osascript(
    script: str, *args: str, timeout: float = DEFAULT_TIMEOUT
) -> Tuple[bool, str, str]:
    """Run a *constant* AppleScript, passing values through ``on run argv``.

    ``script`` must not contain any interpolated external text; every variable
    value goes in ``args``.
    """
    return run_argv([OSASCRIPT, "-e", script, *[str(a) for a in args]], timeout=timeout)


def first_line(text: Optional[str]) -> str:
    return (text or "").strip().splitlines()[-1] if (text or "").strip() else ""
