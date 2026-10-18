"""Notifications from the HUD's own app identity ("Vesper HUD"), osascript only as fallback. Fakes: no banner, no sound, no prompt."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock

import httpx
import pytest
import websockets

from launcher import doctor
from tests.test_gateway import TOKEN, _auth, _running_gateway
from utils import notifications as nt
from utils import notify_cli
from utils import notify_path as np_


@pytest.fixture(autouse=True)
def fresh_path(tmp_path, monkeypatch):
    np_.PATH.__init__()                                              # a fresh process-global per test
    monkeypatch.setattr(nt, "STATE_PATH", tmp_path / "n.json")
    monkeypatch.setattr(nt, "terminal_app", lambda env=None: {"iTerm.app": "iTerm"}.get((env or {}).get("TERM_PROGRAM"), "Terminal"))
    yield
    np_.PATH.__init__()


# ------------------------------------------------------------------ path selection

def test_no_hud_means_the_osascript_fallback():
    assert np_.PATH.active() == np_.OSASCRIPT


def test_hud_connected_means_the_hud_path_until_it_says_it_cannot():
    np_.PATH.hud_hello()
    assert np_.PATH.active() == np_.HUD
    np_.PATH.hud_report("granted")
    assert np_.PATH.active() == np_.HUD
    np_.PATH.hud_report("denied")
    assert np_.PATH.active() == np_.OSASCRIPT                       # fall back when the HUD cannot post banners
    np_.PATH.hud_report("granted")
    assert np_.PATH.active() == np_.HUD


def test_hud_unavailable_falls_back_and_disconnect_falls_back():
    np_.PATH.hud_hello(); np_.PATH.hud_report("unavailable")
    assert np_.PATH.active() == np_.OSASCRIPT
    np_.PATH.__init__(); np_.PATH.hud_hello(); np_.PATH.hud_gone()
    assert np_.PATH.active() == np_.OSASCRIPT


def test_two_hud_windows_one_leaves_the_hud_path_stays():
    np_.PATH.hud_hello(); np_.PATH.hud_hello(); np_.PATH.hud_gone()
    assert np_.PATH.active() == np_.HUD


def test_the_app_whose_permission_matters_follows_the_path():
    assert np_.app_for(np_.HUD, "Terminal") == "Vesper HUD" and np_.app_for(np_.OSASCRIPT, "iTerm") == "iTerm"


# ------------------------------------------------------------------ the notify tool uses the active path

def agent(show_ok=True):
    from agents.system_agent import SystemAgent

    a = SystemAgent.__new__(SystemAgent)
    a._name, a._config = "SystemAgent", {}
    a._event_bus = type("B", (), {"emit": AsyncMock()})()
    calls = []
    a.executor = type("E", (), {"show_notification": lambda self, **k: (calls.append(k) or (show_ok, "ok"))})()
    return a, calls


def notify(a):
    async def go():
        return a._handle_notify({"title": "Vesper", "message": "Tea is ready"}, "")
    return asyncio.run(go())


def test_with_the_hud_connected_osascript_is_not_used_and_the_result_names_the_hud():
    np_.PATH.hud_hello()
    a, osascript_calls = agent()
    out = notify(a)
    assert osascript_calls == []                                            # osascript only as fallback
    assert "via the HUD" in out and "Vesper HUD" in out
    assert [type(c.args[0]).__name__ for c in a._event_bus.emit.call_args_list] == ["ProactiveNotificationEvent"]
    assert "notification_displayed" in out.split("cannot_claim:")[1]        # still unconfirmed: no display claim


def test_hud_missing_falls_back_to_osascript_and_says_so():
    a, osascript_calls = agent()
    out = notify(a)
    assert len(osascript_calls) == 1 and "via osascript (fallback" in out and "Terminal" in out


def test_hud_denied_falls_back_to_osascript():
    np_.PATH.hud_hello(); np_.PATH.hud_report("denied")
    a, osascript_calls = agent()
    out = notify(a)
    assert len(osascript_calls) == 1 and "osascript" in out


def test_confirmation_is_per_app_so_a_terminal_yes_does_not_vouch_for_the_hud():
    nt.NotifyState().save(nt.CONFIRMED, "Terminal")
    np_.PATH.hud_hello()
    a, _ = agent()
    assert "notification_displayed" not in notify(a).split("can_claim:")[1].split("cannot_claim:")[0]
    nt.NotifyState().save(nt.CONFIRMED, "Vesper HUD")
    a, _ = agent()
    assert "notification_displayed" in notify(a).split("can_claim:")[1].split("cannot_claim:")[0]


def test_a_confirmation_does_not_erase_the_delivery_record():
    st = nt.NotifyState()
    st.save_path(np_.HUD, True, "granted")
    st.save(nt.CONFIRMED, "Vesper HUD")
    assert st.delivery()["path"] == np_.HUD and st.status("Vesper HUD") == nt.CONFIRMED


# ------------------------------------------------------------------ the proactive watcher

@pytest.mark.asyncio
async def test_the_watcher_skips_osascript_when_the_hud_will_post_the_banner(tmp_path):
    from tests.test_proactive_watcher import Clock, make, meeting

    np_.PATH.hud_hello()
    clock = Clock()
    w, cache, bus, notifier, _ = make(tmp_path, [meeting(1, 29, clock)], clock, voice=False)
    d = (await w.tick())[0]
    assert notifier.sent == [] and d.channels == ["hud", "macos"]


@pytest.mark.asyncio
async def test_the_watcher_uses_osascript_when_there_is_no_hud(tmp_path):
    from tests.test_proactive_watcher import Clock, make, meeting

    clock = Clock()
    w, cache, bus, notifier, _ = make(tmp_path, [meeting(1, 29, clock)], clock, voice=False)
    d = (await w.tick())[0]
    assert len(notifier.sent) == 1 and d.channels == ["hud", "macos"]


# ------------------------------------------------------------------ the gateway: hello, hud_notify, /notify

async def hud_connect(port):
    ws = await websockets.connect(f"ws://127.0.0.1:{port}/ws", additional_headers=_auth())
    await ws.recv()
    return ws


@pytest.mark.asyncio
async def test_hud_hello_switches_the_path_and_disconnect_switches_it_back():
    async with _running_gateway() as (gw, port, bus, brain):
        ws = await hud_connect(port)
        await ws.send(json.dumps({"type": "hello", "client": "hud"}))
        await asyncio.sleep(0.15)
        assert np_.PATH.active() == np_.HUD and nt.NotifyState().delivery()["path"] == "hud"
        await ws.send(json.dumps({"type": "hud_notify", "state": "denied", "delivered": False}))
        await asyncio.sleep(0.15)
        assert np_.PATH.active() == np_.OSASCRIPT and nt.NotifyState().delivery()["hud_state"] == "denied"
        await ws.send(json.dumps({"type": "hud_notify", "state": "granted", "delivered": True}))
        await asyncio.sleep(0.15)
        assert np_.PATH.active() == np_.HUD
        await ws.close()
        await asyncio.sleep(0.3)
        assert np_.PATH.active() == np_.OSASCRIPT and nt.NotifyState().delivery()["path"] == "osascript"


@pytest.mark.asyncio
async def test_a_client_that_never_said_hello_is_not_the_hud():
    async with _running_gateway() as (gw, port, bus, brain):
        ws = await hud_connect(port)                                         # e.g. voice output
        await ws.send(json.dumps({"type": "hud_notify", "state": "granted", "delivered": True}))
        await asyncio.sleep(0.15)
        assert np_.PATH.active() == np_.OSASCRIPT
        await ws.close()


@pytest.mark.asyncio
async def test_the_native_flag_reaches_the_hud_on_the_wire():
    from schemas.events import ProactiveNotificationEvent

    async with _running_gateway() as (gw, port, bus, brain):
        ws = await hud_connect(port)
        await bus.emit(ProactiveNotificationEvent(title="Vesper", body="Tea is ready", kind="notify", origin="SystemAgent"))
        msg = json.loads(await asyncio.wait_for(ws.recv(), 3))
        assert msg["type"] == "observation" and msg["native"] is True and msg["title"] == "Vesper" and msg["body"] == "Tea is ready"
        await ws.close()


@pytest.mark.asyncio
async def test_notify_endpoint_reports_the_path_and_only_emits_for_the_hud():
    from schemas.events import ProactiveNotificationEvent

    async with _running_gateway() as (gw, port, bus, brain):
        seen = []

        async def grab(e):
            seen.append(e)

        bus.subscribe(ProactiveNotificationEvent, grab)
        async with httpx.AsyncClient() as c:
            url = f"http://127.0.0.1:{port}"
            r = await c.post(f"{url}/notify", json={"title": "t", "body": "b"}, headers=_auth())
            assert r.json()["path"] == "osascript" and seen == []            # the caller posts via osascript itself
            np_.PATH.hud_hello()
            r = await c.post(f"{url}/notify", json={"title": "t", "body": "b"}, headers=_auth())
            assert r.json()["path"] == "hud" and r.json()["app"] == "Vesper HUD" and len(seen) == 1
            assert (await c.post(f"{url}/notify", json={})).status_code == 401


def test_notify_test_uses_the_hud_when_the_gateway_says_so_and_stores_the_answer_for_the_hud(monkeypatch):
    monkeypatch.setattr(notify_cli, "ask_gateway", lambda *a, **k: {"path": "hud"})
    # mimic main()'s decision: HUD path -> app 'Vesper HUD'
    seen = {"path": np_.OSASCRIPT}
    got = notify_cli.ask_gateway("t", "b", "http://x", "tok")
    if got and got.get("path") == np_.HUD:
        seen["path"] = np_.HUD
    st = nt.NotifyState()
    lines = []
    rc = nt.run_notify_test(lambda t, m: (True, ""), lambda p: "y", st, np_.app_for(seen["path"], "Terminal"), lines.append, posted=True)
    assert rc == 0 and st.status("Vesper HUD") == nt.CONFIRMED and st.status("Terminal") == nt.UNCONFIRMED


def test_ask_gateway_returns_none_when_nothing_is_listening():
    assert notify_cli.ask_gateway("t", "b", "http://127.0.0.1:9", "tok", timeout=0.3) is None


# ------------------------------------------------------------------ doctor

def test_doctor_reports_the_hud_path_and_names_the_hud_as_the_app_to_enable():
    st = nt.NotifyState()
    st.save_path(np_.HUD, True, "granted")
    [c] = doctor.check_notifications({"TERM_PROGRAM": "Apple_Terminal"}, st)
    assert c.status == doctor.WARN and "path: HUD (native banner, app identity 'Vesper HUD')" in c.detail and "unconfirmed for Vesper HUD" in c.detail
    assert "System Settings > Notifications > Vesper HUD > Allow Notifications" in c.fix
    st.save(nt.CONFIRMED, "Vesper HUD")
    [c] = doctor.check_notifications({"TERM_PROGRAM": "Apple_Terminal"}, st)
    assert c.status == doctor.PASS and "confirmed" in c.detail and "Vesper HUD" in c.detail


def test_doctor_reports_the_osascript_fallback_with_the_reason():
    st = nt.NotifyState()
    st.save_path(np_.OSASCRIPT, False, "denied")
    [c] = doctor.check_notifications({"TERM_PROGRAM": "iTerm.app"}, st)
    assert "osascript fallback (iTerm" in c.detail and "not connected or cannot post" in c.detail and "iTerm" in c.fix
    [c] = doctor.check_notifications({"TERM_PROGRAM": "iTerm.app"}, nt.NotifyState(nt.STATE_PATH.parent / "none.json"))
    assert "no HUD has been seen" in c.detail


def test_hud_rebuild_is_picked_up_by_the_freshness_check():
    """Cargo.toml, capabilities and src are all in the freshness globs, so the next `vesper up` rebuilds the HUD."""
    from launcher import hudbuild
    for g in ("src/**/*", "src-tauri/Cargo.toml", "src-tauri/capabilities/**/*", "src-tauri/src/**/*"):
        assert g in hudbuild.SOURCE_GLOBS
