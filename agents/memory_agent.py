"""
MemoryAgent - Persistent Memory Storage for VESPER

This agent manages two types of memory:
1. Short-term context: Recent commands, session state, last app opened
2. Long-term preferences: User preferences (e.g., likes VS Code, prefers dark mode)

Storage: SQLite for structured data.
Future: Prepared for vector embeddings (not implemented yet).

IMPORTANT: MemoryAgent does NOT perform reasoning.
It only stores and retrieves data. Reasoning is handled by the Brain.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, AgentCapability
from bus.event_bus import EventBus
from rag.rag_service import ChromaRAGMemoryService
from schemas.events import (
    MemoryStoreEvent,
    MemoryQueryEvent,
    MemoryQueryResultEvent,
    VoiceInputEvent,
    IntentRecognizedEvent,
    ShutdownRequestedEvent,
)
from utils.logger import get_logger


logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class MemoryType(str, Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"  # Expires after TTL
    LONG_TERM = "long_term"    # Persistent


class Category(str, Enum):
    """Categories for organizing memory."""
    # Short-term categories
    COMMAND = "command"           # Recent commands (last_command)
    APP_STATE = "app_state"       # Last opened app, focused window
    CONTEXT = "context"           # Current session context
    CONVERSATION = "conversation" # Recent conversation turns
    
    # Long-term categories  
    PREFERENCE = "preference"     # User preferences (dark_mode, default_browser)
    PATTERN = "pattern"           # Learned usage patterns (app_usage counts)
    ENTITY = "entity"             # Known entities (contacts, places)
    SETTING = "setting"           # Persistent settings


# Default TTLs
DEFAULT_SHORT_TERM_TTL = 3600      # 1 hour
COMMAND_TTL = 1800                 # 30 minutes
CONVERSATION_TTL = 7200            # 2 hours
CLEANUP_INTERVAL = 300             # 5 minutes


# =============================================================================
# SQLITE MEMORY STORE
# =============================================================================

class MemoryStore:
    """
    SQLite-based memory storage with TTL support.
    
    Schema is designed to be extensible for future vector embeddings:
    - embedding BLOB column (reserved, not used yet)
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_type TEXT NOT NULL,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL,
                access_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                
                -- Future: vector embedding for semantic search
                -- embedding BLOB,
                
                UNIQUE(memory_type, category, key)
            );
            
            CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(memory_type);
            CREATE INDEX IF NOT EXISTS idx_category ON memory(category);
            CREATE INDEX IF NOT EXISTS idx_key ON memory(key);
            CREATE INDEX IF NOT EXISTS idx_expires ON memory(expires_at);
            CREATE INDEX IF NOT EXISTS idx_updated ON memory(updated_at);
        """)
        conn.commit()
        logger.info(f"Memory store initialized at {self.db_path}")
    
    def store(
        self,
        memory_type: str,
        category: str,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Store or update a memory entry.
        
        Args:
            memory_type: "short_term" or "long_term"
            category: Category for organization
            key: Unique key within type/category
            value: Data to store (JSON-serializable)
            ttl_seconds: Time-to-live (None = no expiry)
            metadata: Additional metadata
            
        Returns:
            Row ID of stored entry
        """
        conn = self._get_conn()
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds else None
        
        value_json = json.dumps(value, default=str)
        metadata_json = json.dumps(metadata or {})
        
        cursor = conn.execute("""
            INSERT INTO memory (memory_type, category, key, value, created_at, updated_at, expires_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_type, category, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at,
                expires_at = COALESCE(excluded.expires_at, memory.expires_at),
                metadata = excluded.metadata,
                access_count = memory.access_count + 1
        """, (memory_type, category, key, value_json, now, now, expires_at, metadata_json))
        
        conn.commit()
        return cursor.lastrowid or 0
    
    def get(
        self,
        memory_type: Optional[str] = None,
        category: Optional[str] = None,
        key: Optional[str] = None,
        limit: int = 10,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query memory entries.
        
        Args:
            memory_type: Filter by type (None = all)
            category: Filter by category (None = all)
            key: Filter by exact key (None = all)
            limit: Maximum results
            include_expired: Include expired entries
            
        Returns:
            List of memory entries as dicts
        """
        conn = self._get_conn()
        
        conditions = []
        params: List[Any] = []
        
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        if key:
            conditions.append("key = ?")
            params.append(key)
        
        if not include_expired:
            conditions.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(time.time())
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT * FROM memory
            WHERE {where_clause}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(limit)
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            try:
                value = json.loads(row['value'])
            except json.JSONDecodeError:
                value = row['value']
            
            results.append({
                'id': row['id'],
                'memory_type': row['memory_type'],
                'category': row['category'],
                'key': row['key'],
                'value': value,
                'updated_at': datetime.fromtimestamp(row['updated_at']).isoformat(),
                'access_count': row['access_count'],
            })
        
        return results
    
    def delete(
        self,
        memory_type: Optional[str] = None,
        category: Optional[str] = None,
        key: Optional[str] = None,
        older_than_seconds: Optional[int] = None
    ) -> int:
        """
        Delete memory entries.
        
        Returns:
            Number of deleted entries
        """
        conn = self._get_conn()
        
        conditions = []
        params: List[Any] = []
        
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        if key:
            conditions.append("key = ?")
            params.append(key)
        
        if older_than_seconds:
            cutoff = time.time() - older_than_seconds
            conditions.append("updated_at < ?")
            params.append(cutoff)
        
        if not conditions:
            return 0  # Safety: require at least one condition
        
        where_clause = " AND ".join(conditions)
        cursor = conn.execute(f"DELETE FROM memory WHERE {where_clause}", params)
        conn.commit()
        
        return cursor.rowcount
    
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count deleted."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM memory WHERE expires_at IS NOT NULL AND expires_at < ?",
            (time.time(),)
        )
        conn.commit()
        return cursor.rowcount
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        conn = self._get_conn()
        
        stats = {'total': 0, 'by_type': {}, 'by_category': {}}
        
        cursor = conn.execute("SELECT COUNT(*) FROM memory")
        stats['total'] = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT memory_type, COUNT(*) as cnt FROM memory GROUP BY memory_type")
        stats['by_type'] = {row['memory_type']: row['cnt'] for row in cursor.fetchall()}
        
        cursor = conn.execute("SELECT category, COUNT(*) as cnt FROM memory GROUP BY category")
        stats['by_category'] = {row['category']: row['cnt'] for row in cursor.fetchall()}
        
        return stats
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# =============================================================================
# MEMORY AGENT
# =============================================================================

class MemoryAgent(BaseAgent):
    """
    Agent for persistent memory storage.
    
    Responsibilities:
    - Store/retrieve short-term context (last command, last app)
    - Store/retrieve long-term preferences (user likes, settings)
    - Track conversation history
    - Automatic cleanup of expired entries
    
    Does NOT:
    - Perform reasoning
    - Make decisions
    - Generate responses
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name or "MemoryAgent",
            event_bus=event_bus,
            config=config,
        )
        
        # Get db_path from config or use default
        memory_config = (config or {}).get("memory", {})
        db_path_str = (
            memory_config.get("db_path")
            or memory_config.get("sqlite", {}).get("database_path")
            or "data/memory.db"
        )
        self.db_path = Path(db_path_str)
        
        self.store: Optional[MemoryStore] = None
        self._rag_service: Optional[ChromaRAGMemoryService] = None
        self._rag_enabled = bool(memory_config.get("vector_store", {}).get("enabled", False))
        self._vector_provider = str(memory_config.get("vector_store", {}).get("provider", "chroma")).lower()
        self._last_cleanup = 0.0
        self._conversation_turn = 0
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities."""
        return [
            AgentCapability(
                name="memory_storage",
                description="Store and retrieve short-term and long-term memory",
                input_events=["MemoryStoreEvent", "MemoryQueryEvent"],
                output_events=["MemoryQueryResultEvent"],
            ),
            AgentCapability(
                name="context_tracking",
                description="Track conversation context and user patterns",
                input_events=["VoiceInputEvent", "IntentRecognizedEvent"],
                output_events=[],
            ),
        ]
    
    async def _setup(self) -> None:
        """Initialize memory store and subscriptions."""
        self.store = MemoryStore(self.db_path)

        # Optional semantic memory service
        if self._rag_enabled and self._vector_provider == "chroma":
            try:
                memory_config = self._config.get("memory", {})
                vs_cfg = memory_config.get("vector_store", {})
                persist_dir = vs_cfg.get("persist_directory", "data/chroma_memory")
                collection_name = vs_cfg.get("collection_name", "vesper_memory")
                embedding_model = vs_cfg.get("embedding_model")

                self._rag_service = ChromaRAGMemoryService(
                    persist_directory=persist_dir,
                    collection_name=collection_name,
                    embedding_model=embedding_model,
                    chunk_size_tokens=int(vs_cfg.get("chunk_size_tokens", 400)),
                    chunk_overlap_tokens=int(vs_cfg.get("chunk_overlap_tokens", 64)),
                )
                self._logger.info("Semantic vector memory enabled (Chroma)")
            except Exception as exc:
                self._logger.error(f"Failed to initialize semantic memory service: {exc}")
                self._rag_service = None
        
        # Subscribe to memory events
        self._subscribe(MemoryStoreEvent, self._handle_store)
        self._subscribe(MemoryQueryEvent, self._handle_query)
        
        # Subscribe for automatic context tracking
        self._subscribe(VoiceInputEvent, self._handle_voice_input)
        self._subscribe(IntentRecognizedEvent, self._handle_intent)
        
        # Initial cleanup
        if self.store:
            expired = self.store.cleanup_expired()
            if expired > 0:
                logger.info(f"Cleaned up {expired} expired memory entries")
        
        self._logger.info("Memory agent initialized")
    
    async def _teardown(self) -> None:
        """Close memory store."""
        if self._rag_service:
            self._rag_service.close()
            self._rag_service = None

        if self.store:
            self.store.cleanup_expired()
            self.store.close()
            self.store = None
        self._logger.info("Memory agent shutdown complete")
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    async def _handle_store(self, event: MemoryStoreEvent) -> None:
        """Handle memory store request."""
        if not self.store:
            return
        
        try:
            # Parse category from key if not explicit
            category = Category.CONTEXT.value
            if event.key.startswith("pref_"):
                category = Category.PREFERENCE.value
            elif event.key.startswith("app_"):
                category = Category.APP_STATE.value
            elif event.key.startswith("cmd_") or event.key == "last_command":
                category = Category.COMMAND.value
            
            # Apply default TTL for short-term
            ttl = event.ttl
            if event.memory_type == MemoryType.SHORT_TERM.value and ttl is None:
                ttl = DEFAULT_SHORT_TERM_TTL
            
            self.store.store(
                memory_type=event.memory_type,
                category=category,
                key=event.key,
                value=event.value,
                ttl_seconds=ttl
            )
            
            self._logger.debug(f"Stored: {event.memory_type}/{category}/{event.key}")
            self._maybe_cleanup()
            
        except Exception as e:
            self._logger.error(f"Store failed: {e}")
    
    async def _handle_query(self, event: MemoryQueryEvent) -> None:
        """Handle memory query request."""
        if not self.store:
            return
        
        try:
            # Determine query parameters
            memory_type = None if event.memory_type == "all" else event.memory_type
            
            # Check if query is a key lookup
            key = None
            category = None
            
            if event.query:
                # If query looks like a key, do exact match
                if not " " in event.query:
                    key = event.query
            
            results = self.store.get(
                memory_type=memory_type,
                category=category,
                key=key,
                limit=event.limit
            )
            
            # Emit result
            await self.event_bus.emit(MemoryQueryResultEvent(
                source=self.name,
                results=results,
                query=event.query,
                total_matches=len(results)
            ))
            
            self._logger.debug(f"Query '{event.query}' returned {len(results)} results")
            
        except Exception as e:
            self._logger.error(f"Query failed: {e}")
            await self.event_bus.emit(MemoryQueryResultEvent(
                source=self.name,
                results=[],
                query=event.query,
                total_matches=0
            ))
    
    async def _handle_voice_input(self, event: VoiceInputEvent) -> None:
        """Track user input in short-term memory."""
        if not self.store:
            return
        
        self._conversation_turn += 1
        
        # Store last command
        self.store.store(
            memory_type=MemoryType.SHORT_TERM.value,
            category=Category.COMMAND.value,
            key="last_command",
            value=event.text,
            ttl_seconds=COMMAND_TTL
        )
        
        # Store in conversation history
        self.store.store(
            memory_type=MemoryType.SHORT_TERM.value,
            category=Category.CONVERSATION.value,
            key=f"turn_{self._conversation_turn}_user",
            value={'role': 'user', 'text': event.text, 'turn': self._conversation_turn},
            ttl_seconds=CONVERSATION_TTL
        )

        if self._rag_service:
            try:
                await self._rag_service.ingest_text(
                    text=event.text,
                    memory_type=MemoryType.SHORT_TERM.value,
                    intent="VOICE_INPUT",
                    metadata={
                        "role": "user",
                        "turn": self._conversation_turn,
                        "source": "voice",
                    },
                    salience=0.45,
                )
            except Exception as exc:
                self._logger.warning(f"Semantic ingest failed for voice input: {exc}")
    
    async def _handle_intent(self, event: IntentRecognizedEvent) -> None:
        """Track intents and learn patterns."""
        if not self.store:
            return
        
        # Store last intent
        self.store.store(
            memory_type=MemoryType.SHORT_TERM.value,
            category=Category.CONTEXT.value,
            key="last_intent",
            value={'intent': event.intent, 'entities': event.entities},
            ttl_seconds=COMMAND_TTL
        )
        
        # Track app usage for long-term patterns
        if event.intent in ("OPEN_APP", "OPEN_APPLICATION"):
            app_name = event.entities.get("app_name", "")
            if app_name:
                self._track_app_opened(app_name)

        if self._rag_service:
            try:
                # Index intent interpretation as short-term episodic memory
                intent_text = (
                    f"User intent={event.intent}; entities={event.entities}; "
                    f"raw_text='{event.raw_text}'"
                )
                await self._rag_service.ingest_text(
                    text=intent_text,
                    memory_type=MemoryType.SHORT_TERM.value,
                    intent=event.intent,
                    metadata={
                        "source": "intent",
                        "confidence": event.confidence,
                    },
                    salience=0.60,
                )

                # Promote select intent classes to long-term semantic memory
                promotable = {
                    "OPEN_APP",
                    "OPEN_APPLICATION",
                    "SET_REMINDER",
                    "SAVE_MEMORY",
                    "RECALL_MEMORY",
                    "CONTROL_VOLUME",
                }
                if event.intent in promotable:
                    await self._rag_service.ingest_text(
                        text=intent_text,
                        memory_type=MemoryType.LONG_TERM.value,
                        intent=event.intent,
                        metadata={
                            "source": "intent",
                            "confidence": event.confidence,
                            "promoted": True,
                        },
                        salience=0.72,
                    )
            except Exception as exc:
                self._logger.warning(f"Semantic ingest failed for intent: {exc}")
    
    def _track_app_opened(self, app_name: str) -> None:
        """Track app usage for learning preferences."""
        if not self.store:
            return
        
        # Short-term: last opened app
        self.store.store(
            memory_type=MemoryType.SHORT_TERM.value,
            category=Category.APP_STATE.value,
            key="last_opened_app",
            value=app_name,
            ttl_seconds=DEFAULT_SHORT_TERM_TTL
        )
        
        # Long-term: app usage count
        key = f"app_usage_{app_name.lower().replace(' ', '_')}"
        existing = self.store.get(
            memory_type=MemoryType.LONG_TERM.value,
            category=Category.PATTERN.value,
            key=key,
            limit=1
        )
        
        count = 1
        if existing:
            val = existing[0].get('value', {})
            count = (val.get('count', 0) if isinstance(val, dict) else 0) + 1
        
        self.store.store(
            memory_type=MemoryType.LONG_TERM.value,
            category=Category.PATTERN.value,
            key=key,
            value={'app_name': app_name, 'count': count}
        )
    
    def _maybe_cleanup(self) -> None:
        """Periodic cleanup of expired entries."""
        now = time.time()
        if now - self._last_cleanup > CLEANUP_INTERVAL:
            if self.store:
                expired = self.store.cleanup_expired()
                if expired > 0:
                    self._logger.debug(f"Cleaned up {expired} expired entries")
            self._last_cleanup = now
    
    # =========================================================================
    # PUBLIC API (for direct access by other agents)
    # =========================================================================
    
    def get_last_command(self) -> Optional[str]:
        """Get the last user command."""
        if not self.store:
            return None
        results = self.store.get(
            memory_type=MemoryType.SHORT_TERM.value,
            category=Category.COMMAND.value,
            key="last_command",
            limit=1
        )
        return results[0]['value'] if results else None
    
    def get_last_app(self) -> Optional[str]:
        """Get the last opened application."""
        if not self.store:
            return None
        results = self.store.get(
            memory_type=MemoryType.SHORT_TERM.value,
            category=Category.APP_STATE.value,
            key="last_opened_app",
            limit=1
        )
        return results[0]['value'] if results else None
    
    def get_preference(self, key: str) -> Optional[Any]:
        """Get a user preference."""
        if not self.store:
            return None
        results = self.store.get(
            memory_type=MemoryType.LONG_TERM.value,
            category=Category.PREFERENCE.value,
            key=key,
            limit=1
        )
        return results[0]['value'] if results else None
    
    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference (long-term)."""
        if not self.store:
            return
        self.store.store(
            memory_type=MemoryType.LONG_TERM.value,
            category=Category.PREFERENCE.value,
            key=key,
            value=value
        )
    
    def get_recent_conversation(self, max_turns: int = 5) -> List[Dict]:
        """Get recent conversation history."""
        if not self.store:
            return []
        results = self.store.get(
            memory_type=MemoryType.SHORT_TERM.value,
            category=Category.CONVERSATION.value,
            limit=max_turns * 2
        )
        return [
            {'role': r['value'].get('role', 'unknown'), 'text': r['value'].get('text', '')}
            for r in results
            if isinstance(r.get('value'), dict)
        ]
    
    def get_frequent_apps(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get most frequently used apps."""
        if not self.store:
            return []
        
        results = self.store.get(
            memory_type=MemoryType.LONG_TERM.value,
            category=Category.PATTERN.value,
            limit=100  # Get all, then sort
        )
        
        apps = []
        for r in results:
            if r['key'].startswith('app_usage_'):
                val = r.get('value', {})
                if isinstance(val, dict):
                    apps.append((val.get('app_name', ''), val.get('count', 0)))
        
        # Sort by count descending
        apps.sort(key=lambda x: x[1], reverse=True)
        return apps[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.store:
            return {}
        stats = self.store.get_stats()
        stats["semantic_memory_enabled"] = self._rag_service is not None
        return stats

    async def semantic_retrieve(
        self,
        query: str,
        intent: str = "",
        top_k: int = 8,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve semantically relevant memory chunks."""
        if not self._rag_service:
            return []

        return await self._rag_service.retrieve(
            query=query,
            intent=intent,
            top_k=top_k,
            memory_type=memory_type,
        )

    def get_rag_service(self) -> Optional[ChromaRAGMemoryService]:
        """Get the active RAG service instance (if enabled)."""
        return self._rag_service

    async def index_memory_text(
        self,
        text: str,
        memory_type: str = MemoryType.LONG_TERM.value,
        intent: str = "MANUAL_INDEX",
        metadata: Optional[Dict[str, Any]] = None,
        salience: float = 0.5,
    ) -> int:
        """Manually index text into semantic memory."""
        if not self._rag_service:
            return 0
        return await self._rag_service.ingest_text(
            text=text,
            memory_type=memory_type,
            intent=intent,
            metadata=metadata,
            salience=salience,
        )

    async def forget(self, query: str, top_k: int = 5) -> List[str]:
        """
        Delete semantic memories matching `query` (the "forget X" tool).

        Returns the text of each deleted memory, so the caller can tell
        the user exactly what was forgotten.
        """
        if not self._rag_service:
            return []
        matches = await self._rag_service.retrieve(query=query, top_k=top_k)
        if not matches:
            return []
        ids = [m["id"] for m in matches]
        await self._rag_service.delete(ids)
        return [m["text"] for m in matches]
