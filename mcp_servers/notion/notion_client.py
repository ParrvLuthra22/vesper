"""
notion_client — read-only Notion API access for the notion MCP server.

Runs inside the notion MCP server's own Python 3.10+ virtualenv
(mcp_servers/.venv) — separate from the main app's Python 3.9.

Auth: NOTION_API_KEY env var (an internal integration token — share the
specific pages/databases with it from Notion's UI first; integrations
see nothing by default). NOTION_DATABASES optionally maps friendly
names (e.g. "projects", "tasks") to database ids as a JSON object, so
the Planner can name a database instead of needing its raw id.

v1 is read-only by design: search, get_page, get_database_rows. No
create/update/delete — that's a deliberate later decision, same as
Gmail's no-send-in-v1 scoping.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"

_client: Optional[httpx.AsyncClient] = None


def _get_api_key() -> str:
    api_key = os.getenv("NOTION_API_KEY")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY is not configured")
    return api_key


def _get_database_map() -> Dict[str, str]:
    raw = os.getenv("NOTION_DATABASES", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _resolve_database_id(db_name_or_id: str) -> str:
    """A configured friendly name (e.g. "tasks") if there is one, else the raw id as given."""
    return _get_database_map().get(db_name_or_id, db_name_or_id)


def configured_database_names() -> List[str]:
    """Friendly database names mapped via NOTION_DATABASES, for tool descriptions."""
    return sorted(_get_database_map())


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=NOTION_API_BASE,
            headers={
                "Authorization": f"Bearer {_get_api_key()}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
    return _client


def _extract_title(page: Dict[str, Any]) -> str:
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            return "".join(t.get("plain_text", "") for t in prop.get("title", [])) or "Untitled"
    return "Untitled"


def _summarize_page(page: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": page.get("id"),
        "title": _extract_title(page),
        "url": page.get("url"),
        "last_edited_time": page.get("last_edited_time"),
        "archived": page.get("archived", False),
    }


def _flatten_property(prop: Dict[str, Any]) -> Any:
    """Reduce one Notion property value to a plain Python value for the LLM."""
    prop_type = prop.get("type")
    if prop_type == "title":
        return "".join(t.get("plain_text", "") for t in prop.get("title", []))
    if prop_type == "rich_text":
        return "".join(t.get("plain_text", "") for t in prop.get("rich_text", []))
    if prop_type == "select":
        value = prop.get("select")
        return value.get("name") if value else None
    if prop_type == "status":
        value = prop.get("status")
        return value.get("name") if value else None
    if prop_type == "multi_select":
        return [item.get("name") for item in prop.get("multi_select", [])]
    if prop_type == "date":
        value = prop.get("date")
        return value.get("start") if value else None
    if prop_type == "checkbox":
        return prop.get("checkbox")
    if prop_type == "number":
        return prop.get("number")
    if prop_type == "people":
        return [person.get("name") for person in prop.get("people", [])]
    if prop_type == "url":
        return prop.get("url")
    return None


def _summarize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "url": row.get("url"),
        "properties": {
            name: _flatten_property(value) for name, value in row.get("properties", {}).items()
        },
    }


async def search(query: str, max_n: int = 10) -> List[Dict[str, Any]]:
    client = _get_client()
    response = await client.post("/search", json={"query": query, "page_size": max_n})
    response.raise_for_status()
    return [_summarize_page(r) for r in response.json().get("results", [])]


async def get_page(page_id: str) -> Dict[str, Any]:
    client = _get_client()
    response = await client.get(f"/pages/{page_id}")
    response.raise_for_status()
    page = response.json()
    summary = _summarize_page(page)
    summary["properties"] = {
        name: _flatten_property(value) for name, value in page.get("properties", {}).items()
    }
    return summary


async def get_database_rows(
    db_id: str, filter: Optional[Dict[str, Any]] = None, max_n: int = 20
) -> List[Dict[str, Any]]:
    database_id = _resolve_database_id(db_id)
    client = _get_client()
    body: Dict[str, Any] = {"page_size": max_n}
    if filter:
        body["filter"] = filter
    response = await client.post(f"/databases/{database_id}/query", json=body)
    response.raise_for_status()
    return [_summarize_row(r) for r in response.json().get("results", [])]
