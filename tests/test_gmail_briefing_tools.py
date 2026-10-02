"""The read-only Gmail extras (mcp_servers/gmail/gmail_client.py): enriched metadata,
batching, sent summary. A fake Gmail service — no network."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

import types

# gmail_client targets the MCP server's own venv; the OAuth-flow package is not in the
# main venv, and these tests never authenticate — stub just that import if it is absent.
try:
    import google_auth_oauthlib.flow  # noqa: F401
except ImportError:  # pragma: no cover - depends on the environment
    _pkg = types.ModuleType("google_auth_oauthlib")
    _flow = types.ModuleType("google_auth_oauthlib.flow")
    _flow.InstalledAppFlow = object
    _pkg.flow = _flow
    sys.modules.setdefault("google_auth_oauthlib", _pkg)
    sys.modules.setdefault("google_auth_oauthlib.flow", _flow)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp_servers" / "gmail"))
import gmail_client  # noqa: E402


class _Req:
    def __init__(self, fn):
        self._fn = fn

    def execute(self):
        return self._fn()


class _Batch:
    def __init__(self, service, callback):
        self._service, self._callback, self._queue = service, callback, []
        service.batches.append(self)

    def add(self, request, request_id=None):
        self._queue.append((request, request_id))

    def execute(self):
        self._service.batch_sizes.append(len(self._queue))
        for request, rid in self._queue:
            try:
                self._callback(rid, request.execute(), None)
            except KeyError as exc:
                self._callback(rid, None, exc)


class FakeService:
    def __init__(self, messages: Dict[str, Dict[str, Any]], listing: List[Dict[str, str]]):
        self.messages, self.listing = messages, listing
        self.batches: List[_Batch] = []
        self.batch_sizes: List[int] = []
        self.list_queries: List[str] = []
        self.single_gets = 0
        self.writes: List[str] = []

    def users(self):
        return self

    def messages(self):  # noqa: F811 - the API chains .messages().list/.get
        return self

    # attribute shadowing: expose both the dict and the method the API expects
    def __getattribute__(self, name):
        if name == "messages":
            return lambda: object.__getattribute__(self, "_api")()
        return object.__getattribute__(self, name)

    def _api(self):
        return self

    def list(self, userId, q, maxResults):
        self.list_queries.append(q)
        return _Req(lambda: {"messages": self.listing[:maxResults]})

    def get(self, userId, id, format, metadataHeaders):
        return _Req(lambda: self.msgs[id])

    def new_batch_http_request(self, callback):
        return _Batch(self, callback)

    # any write would be a bug
    def modify(self, *a, **k):
        self.writes.append("modify"); raise AssertionError("write attempted")

    send = trash = modify


def _msg(i, **kw):
    headers = [{"name": "From", "value": f"Alice <a{i}@example.com>"}, {"name": "To", "value": "me@me.com"},
               {"name": "Subject", "value": f"S{i}"}, {"name": "Date", "value": "Fri, 2 Oct 2026 07:00:00 +0000"},
               {"name": "List-Unsubscribe", "value": "<mailto:u@x>"}, {"name": "Precedence", "value": "Bulk"},
               {"name": "In-Reply-To", "value": "<abc@x>"}]
    m = {"id": f"m{i}", "threadId": f"t{i}", "snippet": "hi", "labelIds": ["INBOX", "UNREAD", "CATEGORY_PROMOTIONS"],
         "internalDate": "1790000000000", "payload": {"headers": headers}}
    m.update(kw)
    return m


@pytest.fixture
def fake(monkeypatch):
    svc = FakeService({}, [])
    svc.msgs = {f"m{i}": _msg(i) for i in range(120)}
    svc.listing = [{"id": f"m{i}", "threadId": f"t{i}"} for i in range(120)]
    monkeypatch.setattr(gmail_client, "_get_service_sync", lambda: svc)
    return svc


def test_enriched_metadata_fields():
    out = gmail_client._enriched_metadata(_msg(1))
    assert out["id"] == "m1" and out["thread_id"] == "t1" and out["unread"] is True
    assert out["labels"] == ["INBOX", "UNREAD", "CATEGORY_PROMOTIONS"]
    assert out["has_list_unsubscribe"] is True and out["precedence"] == "bulk" and out["in_reply_to"] is True
    assert out["internal_date_ms"] == 1790000000000 and out["list_id"] == ""


def test_list_inbox_uses_batches_not_one_call_per_message(fake):
    rows = gmail_client._list_inbox_sync(60, "in:inbox is:unread after:1")
    assert len(rows) == 60 and rows[0]["id"] == "m0" and rows[-1]["id"] == "m59"      # order preserved
    assert fake.batch_sizes == [50, 10]                                                 # 2 round trips, not 60
    assert fake.list_queries == ["in:inbox is:unread after:1"]


def test_list_inbox_caps_max_n_and_skips_messages_that_vanished(fake):
    assert len(gmail_client._list_inbox_sync(10_000, "q")) == 100                       # clamped to 100
    del fake.msgs["m3"]                                                                  # deleted between list and get
    rows = gmail_client._list_inbox_sync(10, "in:inbox")
    assert [r["id"] for r in rows] == [f"m{i}" for i in range(10) if i != 3]


def test_unread_ids_is_a_single_list_call(fake):
    ids = gmail_client._unread_ids_sync(500)
    assert len(ids) == 120 and fake.batch_sizes == [] and fake.list_queries == ["is:unread in:inbox"]


def test_sent_summary_returns_threads_and_lowercased_recipients(fake):
    for i in range(3):
        fake.msgs[f"m{i}"]["payload"]["headers"] = [
            {"name": "To", "value": f"Bob <BOB{i}@Example.com>, carol@x.org"}, {"name": "Cc", "value": "dave@y.io"}]
    out = gmail_client._sent_summary_sync(30, 3)
    assert out["thread_ids"] == ["t0", "t1", "t2"]
    assert out["recipients"] == sorted({"bob0@example.com", "bob1@example.com", "bob2@example.com",
                                        "carol@x.org", "dave@y.io"})
    assert fake.list_queries == ["in:sent newer_than:30d"]


def test_sent_summary_reports_per_recipient_message_counts_and_no_message_text(fake):
    fake.msgs["m0"]["payload"]["headers"] = [{"name": "To", "value": "Bob <bob@x.org>, carol@x.org"}]
    fake.msgs["m1"]["payload"]["headers"] = [{"name": "To", "value": "BOB@x.org"}, {"name": "Cc", "value": "bob@x.org"}]
    fake.msgs["m2"]["payload"]["headers"] = [{"name": "To", "value": "dave@y.io"}]
    out = gmail_client._sent_summary_sync(365, 3)
    assert out["recipient_counts"] == {"bob@x.org": 2, "carol@x.org": 1, "dave@y.io": 1}   # per message, not per header
    assert out["message_count"] == 3 and out["recipients"] == sorted(out["recipient_counts"])
    assert set(out) == {"thread_ids", "recipients", "recipient_counts", "message_count"}     # addresses only: no subject/body/snippet


def test_nothing_in_the_new_tools_can_write(fake):
    gmail_client._list_inbox_sync(5, "in:inbox")
    gmail_client._unread_ids_sync(5)
    gmail_client._sent_summary_sync(30, 5)
    assert fake.writes == []
    src = Path(gmail_client.__file__).read_text()
    start = src.index("# Read-only extras for the briefing engine")
    end = src.index("async def list_unread")
    section = src[start:end]
    for forbidden in (".modify(", ".send(", ".trash(", ".delete(", ".insert(", ".drafts()", "batchModify"):
        assert forbidden not in section, forbidden
