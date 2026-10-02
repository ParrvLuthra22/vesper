"""
gmail_client — thin async wrapper over the Gmail REST API for the Gmail
MCP server (server.py).

Auth: OAuth2 via google-api-python-client. On first run, opens the
user's browser for consent (InstalledAppFlow.run_local_server); the
resulting token is cached at data/google_token.json (gitignored) and
refreshed automatically afterward — no repeat consent needed.

Scope: gmail.modify only. There is no send scope: draft_reply creates a
DRAFT and nothing more. Sending mail is a deliberate v2 decision.

The googleapiclient/httplib2 stack is synchronous; every public function
here wraps its blocking call via asyncio.to_thread so the MCP server's
event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from email.message import EmailMessage
from email.utils import getaddresses
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN_PATH = PROJECT_ROOT / "data" / "google_token.json"
CLIENT_SECRET_PATH = PROJECT_ROOT / "data" / "google_client_secret.json"

# gmail.modify covers read, label changes (archive/mark-read), and drafts —
# it does NOT include gmail.send. Sending is out of scope for v1 by design.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

MAX_THREAD_MESSAGE_CHARS = 1500

_service = None  # lazily built googleapiclient Resource, cached for the process lifetime


def _load_client_config() -> Dict[str, Any]:
    """OAuth client config: GOOGLE_CLIENT_ID/SECRET env vars, or a
    data/google_client_secret.json file downloaded from Google Cloud Console."""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if client_id and client_secret:
        return {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }

    if CLIENT_SECRET_PATH.exists():
        return json.loads(CLIENT_SECRET_PATH.read_text(encoding="utf-8"))

    raise RuntimeError(
        "No Google OAuth client credentials found. Set GOOGLE_CLIENT_ID + "
        f"GOOGLE_CLIENT_SECRET, or place a client_secret.json at {CLIENT_SECRET_PATH}."
    )


def _get_credentials_sync() -> Credentials:
    creds: Optional[Credentials] = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as exc:
                # Google revoked/expired the refresh token itself (not just
                # the short-lived access token) — e.g. the user revoked
                # Vesper's access, or it sat unused past Google's expiry
                # window. A plain retry can never fix this; only a fresh
                # consent flow can, and popping a browser open unannounced
                # mid-session would be worse than a clear instruction.
                raise RuntimeError(
                    "Gmail authentication has expired and Google would not refresh it "
                    f"(reason: {exc}). Delete {TOKEN_PATH} and restart Vesper to "
                    "re-authenticate via the browser consent flow."
                ) from exc
        else:
            flow = InstalledAppFlow.from_client_config(_load_client_config(), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

    return creds


def _get_service_sync():
    global _service
    if _service is None:
        creds = _get_credentials_sync()
        _service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return _service


def _header(headers: List[Dict[str, str]], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _extract_body_text(payload: Dict[str, Any]) -> str:
    """Best-effort extraction of the plain-text body from a Gmail message payload."""
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if mime_type == "text/plain" and body_data:
        return base64.urlsafe_b64decode(body_data.encode("utf-8")).decode("utf-8", errors="replace")

    parts = payload.get("parts") or []
    # Prefer a text/plain part; fall back to the first part with any body data.
    for part in parts:
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            return _extract_body_text(part)
    for part in parts:
        text = _extract_body_text(part)
        if text:
            return text
    return ""


def _summarize_message_metadata(message: Dict[str, Any]) -> Dict[str, Any]:
    headers = message.get("payload", {}).get("headers", [])
    return {
        "id": message.get("id"),
        "thread_id": message.get("threadId"),
        "sender": _header(headers, "From"),
        "subject": _header(headers, "Subject"),
        "date": _header(headers, "Date"),
        "snippet": message.get("snippet", ""),
    }


def _list_unread_sync(max_n: int) -> List[Dict[str, Any]]:
    service = _get_service_sync()
    response = service.users().messages().list(userId="me", q="is:unread", maxResults=max_n).execute()
    results = []
    for ref in response.get("messages", []):
        message = (
            service.users()
            .messages()
            .get(userId="me", id=ref["id"], format="metadata", metadataHeaders=["From", "Subject", "Date"])
            .execute()
        )
        results.append(_summarize_message_metadata(message))
    return results


def _search_sync(query: str, max_n: int = 20) -> List[Dict[str, Any]]:
    service = _get_service_sync()
    response = service.users().messages().list(userId="me", q=query, maxResults=max_n).execute()
    results = []
    for ref in response.get("messages", []):
        message = (
            service.users()
            .messages()
            .get(userId="me", id=ref["id"], format="metadata", metadataHeaders=["From", "Subject", "Date"])
            .execute()
        )
        results.append(_summarize_message_metadata(message))
    return results


def _get_message_sync(message_id: str) -> Dict[str, Any]:
    service = _get_service_sync()
    message = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    summary = _summarize_message_metadata(message)
    summary["body"] = _extract_body_text(message.get("payload", {}))
    return summary


def _thread_text_sync(thread_id: str) -> str:
    service = _get_service_sync()
    thread = service.users().threads().get(userId="me", id=thread_id, format="full").execute()
    blocks = []
    for message in thread.get("messages", []):
        headers = message.get("payload", {}).get("headers", [])
        body = _extract_body_text(message.get("payload", {}))[:MAX_THREAD_MESSAGE_CHARS]
        blocks.append(
            f"From: {_header(headers, 'From')}\n"
            f"Date: {_header(headers, 'Date')}\n"
            f"Subject: {_header(headers, 'Subject')}\n\n"
            f"{body}"
        )
    return "\n\n---\n\n".join(blocks)


def _draft_reply_sync(message_id: str, instruction: str) -> str:
    service = _get_service_sync()
    original = service.users().messages().get(userId="me", id=message_id, format="metadata",
                                               metadataHeaders=["From", "Subject", "Message-ID", "References"]).execute()
    headers = original.get("payload", {}).get("headers", [])
    thread_id = original.get("threadId")

    to_addr = _header(headers, "From")
    subject = _header(headers, "Subject")
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    original_message_id = _header(headers, "Message-ID")
    references = _header(headers, "References")

    reply = EmailMessage()
    reply["To"] = to_addr
    reply["Subject"] = subject
    if original_message_id:
        reply["In-Reply-To"] = original_message_id
        reply["References"] = f"{references} {original_message_id}".strip()
    reply.set_content(instruction)

    raw = base64.urlsafe_b64encode(reply.as_bytes()).decode("utf-8")
    draft = (
        service.users()
        .drafts()
        .create(userId="me", body={"message": {"raw": raw, "threadId": thread_id}})
        .execute()
    )
    return draft.get("id", "")


def _archive_sync(message_id: str) -> None:
    service = _get_service_sync()
    service.users().messages().modify(userId="me", id=message_id, body={"removeLabelIds": ["INBOX"]}).execute()


def _mark_read_sync(message_id: str) -> None:
    service = _get_service_sync()
    service.users().messages().modify(userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}).execute()


def _unread_count_sync() -> int:
    service = _get_service_sync()
    response = service.users().messages().list(userId="me", q="is:unread", maxResults=1).execute()
    return int(response.get("resultSizeEstimate", 0))


# =============================================================================
# Public async API
# =============================================================================

# googleapiclient/httplib2 share one underlying HTTP/credentials object
# (see _get_service_sync's module-level cache) that is not safe for truly
# concurrent requests from multiple threads — e.g. InboxSensor's poll
# landing at the same moment as a user-initiated call reliably produced
# "SSL record layer failure". Serializing every call through one lock
# costs nothing observable (individual Gmail calls are already sequential
# from the caller's point of view) and removes the race entirely.
_api_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Read-only extras for the briefing engine (briefing/). They exist because
# list_unread's metadata has no labels, mailing-list headers, recipients or reply
# signals — everything the deterministic scorer needs. Strictly READ-ONLY: only
# messages.list / messages.get with format=metadata (no body), never modify/send.
# ---------------------------------------------------------------------------

_INBOX_HEADERS = [
    "From", "To", "Subject", "Date", "List-Id", "List-Unsubscribe",
    "Precedence", "Auto-Submitted", "In-Reply-To",
]


def _enriched_metadata(message: Dict[str, Any]) -> Dict[str, Any]:
    headers = message.get("payload", {}).get("headers", [])
    labels = message.get("labelIds") or []
    try:
        internal_ms = int(message.get("internalDate") or 0)
    except (TypeError, ValueError):
        internal_ms = 0
    return {
        "id": message.get("id"),
        "thread_id": message.get("threadId"),
        "sender": _header(headers, "From"),
        "to": _header(headers, "To"),
        "subject": _header(headers, "Subject"),
        "date": _header(headers, "Date"),
        "internal_date_ms": internal_ms,
        "snippet": message.get("snippet", ""),
        "labels": labels,
        "unread": "UNREAD" in labels,
        "list_id": _header(headers, "List-Id"),
        "has_list_unsubscribe": bool(_header(headers, "List-Unsubscribe")),
        "precedence": _header(headers, "Precedence").lower(),
        "auto_submitted": _header(headers, "Auto-Submitted").lower(),
        "in_reply_to": bool(_header(headers, "In-Reply-To")),
    }


BATCH_CHUNK = 50  # Gmail allows up to 100 calls per batch request


def _batch_get_metadata(service: Any, ids: List[str], headers: List[str]) -> List[Dict[str, Any]]:
    """messages.get(format=metadata) for many ids in ONE HTTP round trip per chunk.

    The briefing's first run reads up to 50-200 messages; one-by-one that was ~1 s each
    and blew the collector timeout. Order of `ids` is preserved; messages that fail
    individually (deleted meanwhile) are skipped."""
    found: Dict[str, Dict[str, Any]] = {}

    def on_response(request_id, response, exception):  # googleapiclient batch callback
        if exception is None and response:
            found[request_id] = response

    for start in range(0, len(ids), BATCH_CHUNK):
        batch = service.new_batch_http_request(callback=on_response)
        for message_id in ids[start:start + BATCH_CHUNK]:
            batch.add(
                service.users().messages().get(userId="me", id=message_id, format="metadata", metadataHeaders=headers),
                request_id=message_id,
            )
        batch.execute()
    return [found[i] for i in ids if i in found]


def _list_inbox_sync(max_n: int, query: str) -> List[Dict[str, Any]]:
    service = _get_service_sync()
    response = service.users().messages().list(userId="me", q=query, maxResults=max(1, min(max_n, 100))).execute()
    ids = [ref["id"] for ref in response.get("messages", [])]
    return [_enriched_metadata(m) for m in _batch_get_metadata(service, ids, _INBOX_HEADERS)]


def _unread_ids_sync(max_n: int) -> List[str]:
    service = _get_service_sync()
    response = service.users().messages().list(
        userId="me", q="is:unread in:inbox", maxResults=max(1, min(max_n, 500))
    ).execute()
    return [ref["id"] for ref in response.get("messages", [])]


def _sent_summary_sync(days: int, max_n: int) -> Dict[str, Any]:
    """Thread ids and recipient addresses from my recent SENT mail — the signals
    for "I have replied to this sender before" and "this is a reply to a thread I
    wrote in"."""
    service = _get_service_sync()
    response = service.users().messages().list(
        userId="me", q=f"in:sent newer_than:{max(1, int(days))}d", maxResults=max(1, min(max_n, 500))
    ).execute()
    refs = response.get("messages", [])
    thread_ids = sorted({ref.get("threadId") for ref in refs if ref.get("threadId")})
    counts: Dict[str, int] = {}
    for message in _batch_get_metadata(service, [ref["id"] for ref in refs], ["To", "Cc"]):
        headers = message.get("payload", {}).get("headers", [])
        for addr in {a.lower() for _n, a in getaddresses([_header(headers, "To"), _header(headers, "Cc")]) if a}:
            counts[addr] = counts.get(addr, 0) + 1   # messages (not bodies) that went to this address
    return {
        "thread_ids": thread_ids,
        "recipients": sorted(counts),
        "recipient_counts": counts,
        "message_count": len(refs),
    }


async def list_inbox(max_n: int = 50, query: str = "in:inbox") -> List[Dict[str, Any]]:
    async with _api_lock:
        return await asyncio.to_thread(_list_inbox_sync, max_n, query)


async def unread_ids(max_n: int = 200) -> List[str]:
    async with _api_lock:
        return await asyncio.to_thread(_unread_ids_sync, max_n)


async def sent_summary(days: int = 60, max_n: int = 200) -> Dict[str, Any]:
    async with _api_lock:
        return await asyncio.to_thread(_sent_summary_sync, days, max_n)


async def list_unread(max_n: int = 10) -> List[Dict[str, Any]]:
    async with _api_lock:
        return await asyncio.to_thread(_list_unread_sync, max_n)


async def get_message(message_id: str) -> Dict[str, Any]:
    async with _api_lock:
        return await asyncio.to_thread(_get_message_sync, message_id)


async def thread_text(thread_id: str) -> str:
    async with _api_lock:
        return await asyncio.to_thread(_thread_text_sync, thread_id)


async def search(query: str, max_n: int = 20) -> List[Dict[str, Any]]:
    async with _api_lock:
        return await asyncio.to_thread(_search_sync, query, max_n)


async def draft_reply(message_id: str, instruction: str) -> str:
    async with _api_lock:
        return await asyncio.to_thread(_draft_reply_sync, message_id, instruction)


async def archive(message_id: str) -> None:
    async with _api_lock:
        await asyncio.to_thread(_archive_sync, message_id)


async def mark_read(message_id: str) -> None:
    async with _api_lock:
        await asyncio.to_thread(_mark_read_sync, message_id)


async def unread_count() -> int:
    async with _api_lock:
        return await asyncio.to_thread(_unread_count_sync)
