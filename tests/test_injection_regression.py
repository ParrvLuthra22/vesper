"""
Regression tests for the shell / AppleScript injection class found in the
2026-10-02 audit (VESPER_STATUS.md §5, B1/B2).

The bug: `safe`-tier tools (open_url, open_app, focus_app, close_app, ...) built
`open "{url}"` / `tell application "{name}"` strings from model-chosen values and
ran them through `do shell script`, so `$(...)`, backticks, quotes, `;` or a
newline in a value executed with no confirmation.

The fix: values travel only as discrete argv elements (subprocess, shell=False)
or as AppleScript `on run argv` arguments, never inside a script/command string.

These tests check that at three levels:
  1. what reaches subprocess.run (argv shape, shell=False, payload only as data);
  2. real processes — `echo`/osascript fed the exact same argv — never run the payload;
  3. a source scan that fails if a new shell-string pattern is introduced.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from agents.system_agent import AppleScriptExecutor, SystemAgent
from bus.event_bus import EventBus
from tools import devtools
from utils import safe_exec
from utils.safe_exec import UnsafeInputError, validate_app_name, validate_http_url

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Each payload would create the marker file if any layer re-parsed it as code.
PAYLOAD_TEMPLATES = [
    "$(touch {m})",
    "`touch {m}`",
    '"; touch {m}; "',
    "'; touch {m}; '",
    "x; touch {m} #",
    "a && touch {m}",
    "a | touch {m}",
    "x\ntouch {m}",
    # AppleScript source injection (the exact B2 payload shape):
    'Finder" to activate\ndo shell script "touch {m}"\ntell application "Finder',
    'x\\" & (do shell script "touch {m}") & \\"',
]


@pytest.fixture
def marker() -> Path:
    # Short path on purpose: payloads embed it and app names are length-capped.
    directory = Path(tempfile.mkdtemp(prefix="p", dir="/tmp"))
    yield directory / "pwned"
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture(params=range(len(PAYLOAD_TEMPLATES)), ids=lambda i: f"payload{i}")
def payload(request, marker: Path) -> str:
    return PAYLOAD_TEMPLATES[request.param].format(m=marker)


class _Recorder:
    """Stands in for subprocess.run: records every call, always 'fails' so no
    success-only side effects (event emission) are triggered."""

    def __init__(self) -> None:
        self.calls: List[Tuple[List[str], Dict[str, Any]]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="recorded")


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(safe_exec.subprocess, "run", rec)
    return rec


@pytest.fixture
def executor() -> AppleScriptExecutor:
    # Bypass __init__'s "is /usr/bin/osascript present" check so this runs on any OS.
    ex = AppleScriptExecutor.__new__(AppleScriptExecutor)
    ex._logger = None
    return ex


@pytest.fixture
def agent(executor: AppleScriptExecutor) -> SystemAgent:
    EventBus.reset_instance()
    ag = SystemAgent(event_bus=EventBus())
    ag.executor = executor
    yield ag
    EventBus.reset_instance()


def _assert_no_shell(recorder: _Recorder) -> None:
    assert recorder.calls, "expected the tool to reach subprocess"
    for argv, kwargs in recorder.calls:
        assert kwargs.get("shell") in (None, False), f"shell=True used: {argv}"
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)
        assert os.path.basename(argv[0]) not in ("sh", "bash", "zsh", "dash"), argv


def _osascript_source(argv: List[str]) -> str:
    return argv[argv.index("-e") + 1]


def _osascript_args(argv: List[str]) -> List[str]:
    return argv[argv.index("-e") + 2:]


# ----------------------------------------------------------------------------
# 1. Argv shape: payload is data, never code
# ----------------------------------------------------------------------------

def test_open_app_payload_only_as_argv(executor, recorder, payload, marker):
    ok, _ = executor.open_application(payload)
    if "\n" in payload:  # control chars are refused before any process starts
        assert not recorder.calls
        return
    _assert_no_shell(recorder)
    for argv, _ in recorder.calls:
        if argv[0].endswith("osascript"):
            assert str(marker) not in _osascript_source(argv)
            assert payload in _osascript_args(argv)
        else:
            assert argv[:2] == ["open", "-a"] and argv[2] == payload.strip().rstrip(".")


def test_open_app_newline_payload_is_refused(executor, recorder, marker):
    ok, msg = executor.open_application(PAYLOAD_TEMPLATES[8].format(m=marker))
    assert ok is False and "Refused" in msg
    assert recorder.calls == []


def test_focus_app_payload_only_as_argv(executor, recorder, payload, marker):
    executor.focus_application(payload)
    if "\n" in payload:
        assert recorder.calls == []
        return
    _assert_no_shell(recorder)
    (argv, _), = recorder.calls
    assert str(marker) not in _osascript_source(argv)
    assert _osascript_args(argv) == [payload.strip().rstrip(".")]


@pytest.mark.parametrize("force", [False, True])
def test_close_app_payload_only_as_argv(executor, recorder, payload, marker, force):
    executor.close_application(payload, force=force)
    if "\n" in payload:
        assert recorder.calls == []
        return
    _assert_no_shell(recorder)
    (argv, _), = recorder.calls
    if force:
        assert argv == ["killall", payload.strip().rstrip(".")]
    else:
        assert str(marker) not in _osascript_source(argv)
        assert _osascript_args(argv) == [payload.strip().rstrip(".")]


def test_is_application_running_payload_only_as_argv(executor, recorder, payload, marker):
    executor.is_application_running(payload)
    if "\n" in payload:
        assert recorder.calls == []
        return
    (argv, _), = recorder.calls
    assert str(marker) not in _osascript_source(argv)
    assert payload.strip().rstrip(".") in _osascript_args(argv)


def test_show_notification_payload_only_as_argv(executor, recorder, payload, marker):
    executor.show_notification(title=payload, message=payload, subtitle=payload, sound=payload)
    _assert_no_shell(recorder)
    (argv, _), = recorder.calls
    assert str(marker) not in _osascript_source(argv)
    assert _osascript_args(argv) == [payload, payload, payload, payload]


def test_set_clipboard_payload_only_as_argv(executor, recorder, payload, marker):
    executor.set_clipboard(payload)
    (argv, _), = recorder.calls
    assert str(marker) not in _osascript_source(argv)
    assert _osascript_args(argv) == [payload]


def test_show_alert_payload_only_as_argv(executor, recorder, payload, marker):
    executor.show_alert(title=payload, message=payload, buttons=[payload, "OK"])
    (argv, _), = recorder.calls
    assert str(marker) not in _osascript_source(argv)
    assert payload in _osascript_args(argv)


def test_set_volume_coerces_to_int_argument(executor, recorder):
    executor.set_volume(250)
    (argv, _), = recorder.calls
    assert _osascript_args(argv) == ["100"]
    with pytest.raises((ValueError, TypeError)):
        executor.set_volume("$(touch x)")  # type: ignore[arg-type]


def test_get_disk_info_path_is_argv(executor, recorder, payload, marker):
    executor.get_disk_info(payload)
    if recorder.calls:
        (argv, _), = recorder.calls
        assert argv == ["df", "-H", payload]


def test_get_disk_info_refuses_option_looking_path(executor, recorder):
    executor.get_disk_info("--help")
    assert recorder.calls == []


# ----------------------------------------------------------------------------
# URL tools: http(s) only, argv only
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "file:///etc/passwd",
    "mailto:alice@example.com",
    "shortcuts://run-shortcut?name=x",
    "javascript:alert(1)",
    "x-apple.systempreferences:com.apple.preference.security",
    "ftp://example.com/x",
    "tel:+15555550100",
    "https://",
    "",
    "   ",
    "https://exa mple.com",
    "https://example.com/\nfoo",
])
def test_validate_http_url_refuses(bad):
    with pytest.raises(UnsafeInputError):
        validate_http_url(bad)


@pytest.mark.parametrize("raw,expected", [
    ("example.com", "https://example.com"),
    ("example.com/path?q=1", "https://example.com/path?q=1"),
    ("localhost:8080", "https://localhost:8080"),
    ("http://example.com", "http://example.com"),
    ("  https://example.com/a  ", "https://example.com/a"),
])
def test_validate_http_url_accepts(raw, expected):
    assert validate_http_url(raw) == expected


def test_open_url_non_http_never_reaches_open(agent, recorder):
    for bad in ("file:///etc/passwd", "mailto:a@b.c", "shortcuts://run-shortcut?name=x"):
        out = agent._handle_open_url({"url": bad}, "")
        assert "couldn't open" in out.lower()
    assert recorder.calls == []


def test_open_url_payload_is_one_argv_element(agent, recorder, payload, marker):
    url = f"https://example.com/{payload}"
    agent._handle_open_url({"url": url}, "")
    if "\n" in payload or " " in payload:
        assert recorder.calls == []  # whitespace/control chars refused up front
        return
    (argv, kwargs), = recorder.calls
    assert argv == ["open", url]
    assert kwargs.get("shell") in (None, False)


def test_open_url_real_process_does_not_execute_payload(agent, monkeypatch, marker):
    """Run the REAL handler and a REAL subprocess (argv[0] swapped from `open` to
    `echo` so no browser launches). With the old `open "{url}"` + shell this
    created the marker."""
    real_run = subprocess.run

    def echo_instead_of_open(argv, **kw):
        argv = ["/bin/echo", *argv[1:]] if argv and argv[0] == "open" else argv
        return real_run(argv, **kw)

    monkeypatch.setattr(safe_exec.subprocess, "run", echo_instead_of_open)
    # No whitespace (validation would refuse it first); ${IFS} is a shell-level space.
    for tpl in ("$(touch${{IFS}}{m})", "`touch${{IFS}}{m}`", '";touch${{IFS}}{m};"', "x;touch${{IFS}}{m}"):
        out = agent._handle_open_url({"url": "https://example.com/" + tpl.format(m=marker)}, "")
        assert out.startswith("Opening"), out
    assert not marker.exists()


def test_search_web_query_cannot_inject(agent, recorder, payload, marker):
    agent._handle_search_web({"query": payload}, "")
    (argv, kwargs), = recorder.calls
    assert argv[0] == "open" and argv[1].startswith("https://www.google.com/search?q=")
    assert "touch" not in argv[1].replace("%20", " ").replace("+", " ") or "%" in argv[1]
    assert "$(" not in argv[1] and "`" not in argv[1] and '"' not in argv[1]
    assert kwargs.get("shell") in (None, False)


def test_screenshot_is_argv(agent, recorder):
    agent._handle_screenshot({}, "")
    (argv, _), = recorder.calls
    assert argv[:2] == ["screencapture", "-x"]


# ----------------------------------------------------------------------------
# App handlers end to end (entities -> executor -> subprocess)
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("handler,key", [
    ("_handle_open_app", "app_name"),
    ("_handle_focus_app", "app_name"),
    ("_handle_close_app", "app_name"),
])
def test_app_handlers_never_build_shell(agent, recorder, marker, handler, key):
    getattr(agent, handler)({key: f"Safari$(touch {marker})"}, "")
    _assert_no_shell(recorder)
    for argv, _ in recorder.calls:
        if argv[0].endswith("osascript"):
            assert str(marker) not in _osascript_source(argv)


# ----------------------------------------------------------------------------
# validate_app_name
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "-9", "--help", "a\nb", "a\x00b", "x" * 101])
def test_validate_app_name_refuses(bad):
    with pytest.raises(UnsafeInputError):
        validate_app_name(bad)


def test_validate_app_name_keeps_quotes_as_data():
    # Quotes are legal in a name; they are safe because the name is only ever argv.
    assert validate_app_name('He said "hi".') == 'He said "hi"'


# ----------------------------------------------------------------------------
# Devtools option injection (leading '-')
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["--install-extension evil", "-p evilplugin", "--folder-uri=x"])
async def test_devtools_refuse_option_looking_paths(monkeypatch, bad):
    called: List[Any] = []

    async def fake_run(cmd, cwd=None, timeout=30.0):
        called.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(devtools, "_run", fake_run)
    assert "Refused" in await devtools.open_in_editor({"path": bad}, {})
    assert "Refused" in await devtools.run_tests({"path": bad}, {})
    assert called == []


# ----------------------------------------------------------------------------
# 2. Real osascript: argv values are returned as data, never executed (macOS)
# ----------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform != "darwin" or not os.path.exists("/usr/bin/osascript"),
    reason="needs macOS osascript",
)
def test_real_osascript_treats_argv_as_data(marker):
    script = "on run argv\nreturn item 1 of argv\nend run"
    for tpl in PAYLOAD_TEMPLATES:
        value = tpl.format(m=marker)
        ok, out, err = safe_exec.run_osascript(script, value)
        assert ok, err
        assert out == value.strip()  # echoed back verbatim = it was only data
    assert not marker.exists()


@pytest.mark.skipif(
    sys.platform != "darwin" or not os.path.exists("/usr/bin/osascript"),
    reason="needs macOS osascript",
)
def test_real_osascript_notification_script_compiles_with_hostile_values(marker):
    """The constant notification script must at least compile and run with
    hostile values as arguments (dry: it only builds the strings)."""
    script = (
        "on run argv\n"
        "set theMessage to item 1 of argv\n"
        "set theTitle to item 2 of argv\n"
        "return theTitle & \"|\" & theMessage\n"
        "end run"
    )
    ok, out, err = safe_exec.run_osascript(script, f"$(touch {marker})", '"; touch x; "')
    assert ok, err
    assert not marker.exists()


# ----------------------------------------------------------------------------
# 3. Source scan: no new shell-string patterns
# ----------------------------------------------------------------------------

#: Intentional arbitrary-execution tools, DANGEROUS tier + verbatim confirmation.
_SCAN_ALLOWLIST = {"tools/creator.py"}
#: Dead code (nothing emits MacOSCommandEvent; see VESPER_STATUS.md B4). Listed,
#: not fixed in this change; remove from this set when the module is deleted/fixed.
_KNOWN_LEGACY = {"agents/macos_control_agent.py"}


def _python_sources():
    skip = {".venv", "build", "tests", "node_modules", "target", ".git", "vesper.egg-info"}
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if any(part in skip for part in rel.parts):
            continue
        yield str(rel), path.read_text(encoding="utf-8", errors="replace")


def test_no_shell_true_or_os_system_anywhere():
    pattern = re.compile(r"shell\s*=\s*True|os\.system\(|create_subprocess_shell|subprocess\.getoutput|os\.popen\(")
    offenders = [
        rel for rel, text in _python_sources()
        if rel not in _SCAN_ALLOWLIST and pattern.search(text)
    ]
    assert offenders == []


def test_system_agent_has_no_interpolated_shell_or_applescript():
    text = (REPO_ROOT / "agents" / "system_agent.py").read_text(encoding="utf-8")
    assert "run_shell(" not in text, "string-shell helper must stay removed"
    # f-strings (or concatenation) that build do-shell-script / tell-application source
    bad = re.findall(r"""f(?:'''|\"\"\"|'|")[^\n]*?(?:do shell script|tell application)[^\n]*\{""", text)
    assert bad == [], bad


def test_no_new_fstring_osascript_outside_known_files():
    """An f-string passed to osascript/`do shell script` anywhere else is suspect."""
    pattern = re.compile(r"""f(?:'''|\"\"\"|'|")[^\n]*?(?:do shell script|tell application)[^\n]*\{""")
    offenders = [
        rel for rel, text in _python_sources()
        if rel not in _KNOWN_LEGACY and rel not in _SCAN_ALLOWLIST and pattern.search(text)
    ]
    assert offenders == []
