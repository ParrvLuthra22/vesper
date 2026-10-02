"""SQLite cache for the briefing engine (data/briefing.db).

    items    one row per collected item: the Item JSON, its latest score + reasons,
             whether it is still `active` at the source (unread / still on the calendar)
    sources  per-collector cursor and health: last attempt/success/error
    meta     small JSON blobs (e.g. the sent-mail summary) with an updated-at

Scores are recomputed at briefing time (a meeting's urgency changes by the minute)
and written back, so the file always shows the latest ranking.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from briefing.items import Item

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, item TEXT NOT NULL,
    score INTEGER NOT NULL DEFAULT 0, reasons TEXT NOT NULL DEFAULT '[]',
    excluded INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
    first_seen REAL NOT NULL, updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_source_active ON items(source, active);
CREATE TABLE IF NOT EXISTS sources (
    source TEXT PRIMARY KEY, cursor TEXT, last_attempt REAL, last_success REAL,
    last_error TEXT, consecutive_failures INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated REAL NOT NULL);
"""


@dataclass
class SourceHealth:
    source: str
    cursor: Optional[str]
    last_attempt: Optional[float]
    last_success: Optional[float]
    last_error: Optional[str]
    consecutive_failures: int

    def age_seconds(self, now: float) -> Optional[float]:
        return None if self.last_success is None else max(0.0, now - self.last_success)


class BriefingCache:
    def __init__(self, path: str | Path = "data/briefing.db"):
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---------------------------------------------------------------- items
    def upsert_items(self, items: Iterable[Item], now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        n = 0
        with self._lock:
            for it in items:
                self._db.execute(
                    """INSERT INTO items (id, source, item, first_seen, updated, active)
                       VALUES (?, ?, ?, ?, ?, 1)
                       ON CONFLICT(id) DO UPDATE SET item=excluded.item, updated=excluded.updated, active=1""",
                    (it.id, it.source, json.dumps(it.to_dict()), now, now),
                )
                n += 1
            self._db.commit()
        return n

    def deactivate_missing(self, source: str, live_ids: Set[str]) -> int:
        """Items of `source` no longer present at the source (read, cancelled, past)."""
        with self._lock:
            rows = self._db.execute("SELECT id FROM items WHERE source=? AND active=1", (source,)).fetchall()
            gone = [r["id"] for r in rows if r["id"] not in live_ids]
            for item_id in gone:
                self._db.execute("UPDATE items SET active=0 WHERE id=?", (item_id,))
            self._db.commit()
        return len(gone)

    def active_items(self, source: Optional[str] = None) -> List[Item]:
        q, args = "SELECT item FROM items WHERE active=1", ()
        if source:
            q, args = q + " AND source=?", (source,)
        with self._lock:
            return [Item.from_dict(json.loads(r["item"])) for r in self._db.execute(q, args).fetchall()]

    def save_scores(self, rows: Iterable[Tuple[str, int, List[Dict[str, Any]], bool]]) -> None:
        with self._lock:
            for item_id, score, reasons, excluded in rows:
                self._db.execute(
                    "UPDATE items SET score=?, reasons=?, excluded=? WHERE id=?",
                    (score, json.dumps(reasons), int(excluded), item_id),
                )
            self._db.commit()

    def scored_rows(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute(
                "SELECT id, source, score, reasons, excluded, active FROM items ORDER BY score DESC").fetchall()]

    def prune(self, older_than_days: float = 7.0, now: Optional[float] = None) -> int:
        cutoff = (time.time() if now is None else now) - older_than_days * 86400
        with self._lock:
            cur = self._db.execute("DELETE FROM items WHERE active=0 AND updated < ?", (cutoff,))
            self._db.commit()
            return cur.rowcount

    # -------------------------------------------------------------- sources
    def health(self, source: str) -> SourceHealth:
        with self._lock:
            r = self._db.execute("SELECT * FROM sources WHERE source=?", (source,)).fetchone()
        if r is None:
            return SourceHealth(source, None, None, None, None, 0)
        return SourceHealth(source, r["cursor"], r["last_attempt"], r["last_success"], r["last_error"],
                            r["consecutive_failures"])

    def get_cursor(self, source: str) -> Optional[str]:
        return self.health(source).cursor

    def record_success(self, source: str, cursor: Optional[str], now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._db.execute(
                """INSERT INTO sources (source, cursor, last_attempt, last_success, last_error, consecutive_failures)
                   VALUES (?, ?, ?, ?, NULL, 0)
                   ON CONFLICT(source) DO UPDATE SET cursor=excluded.cursor, last_attempt=excluded.last_attempt,
                     last_success=excluded.last_success, last_error=NULL, consecutive_failures=0""",
                (source, cursor, now, now),
            )
            self._db.commit()

    def record_failure(self, source: str, error: str, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._db.execute(
                """INSERT INTO sources (source, last_attempt, last_error, consecutive_failures)
                   VALUES (?, ?, ?, 1)
                   ON CONFLICT(source) DO UPDATE SET last_attempt=excluded.last_attempt,
                     last_error=excluded.last_error, consecutive_failures=consecutive_failures+1""",
                (source, now, error[:300]),
            )
            self._db.commit()

    # ----------------------------------------------------------------- meta
    def set_meta(self, key: str, value: Any, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._db.execute(
                "INSERT INTO meta (key, value, updated) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
                (key, json.dumps(value), now),
            )
            self._db.commit()

    def get_meta(self, key: str) -> Tuple[Optional[Any], Optional[float]]:
        with self._lock:
            r = self._db.execute("SELECT value, updated FROM meta WHERE key=?", (key,)).fetchone()
        return (None, None) if r is None else (json.loads(r["value"]), r["updated"])
