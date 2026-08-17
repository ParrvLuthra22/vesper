"""
RAG memory service backed by Chroma for personal-assistant memory retrieval.

Phase 1 goals:
- Local persistent vector store (Chroma)
- Type-aware chunking
- Metadata-rich indexing
- Retrieval optimization (similarity + recency + salience + intent)
- Lightweight MMR diversification
- Context assembly under token budget
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rag.embeddings import (
    DEFAULT_MODEL_NAME,
    HashEmbeddingFunction as _HashEmbeddingFunction,  # re-exported for callers/tests
    build_embedding_function,
)
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    similarity: float
    final_score: float


class ChromaRAGMemoryService:
    """Chroma-backed semantic memory service."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "vesper_memory",
        embedding_model: Optional[str] = None,
        chunk_size_tokens: int = 400,
        chunk_overlap_tokens: int = 64,
        recency_half_life_days: float = 7.0,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        # `embedding_model=None` from a caller means "use the project default"
        # (local MiniLM). Passing an explicit empty string is how a caller
        # opts out into the hash fallback — see build_embedding_function.
        self.embedding_model = DEFAULT_MODEL_NAME if embedding_model is None else embedding_model
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.recency_half_life_days = max(1.0, recency_half_life_days)

        self._client = None
        self._collection = None
        self._embedding_fn: Any = None

        self._init_chroma()

    @property
    def _embedding_name(self) -> str:
        """
        What is actually producing vectors right now.

        Resolved live rather than at construction because the model loads
        lazily: until the first embed call we do not yet know whether it
        succeeded or quietly degraded to hashing.
        """
        fn = self._embedding_fn
        if fn is None:
            return "unknown"
        effective = getattr(fn, "effective_name", None)
        return effective if effective is not None else fn.name()

    def _init_chroma(self) -> None:
        """Initialize the Chroma client and collection."""
        try:
            import chromadb

            # Construction is cheap and never touches the model — the actual
            # load happens on the first embed, so importing torch is deferred
            # out of startup entirely.
            self._embedding_fn = build_embedding_function(self.embedding_model)

            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                f"[RAG] Chroma initialized at '{self.persist_directory}' "
                f"collection='{self.collection_name}' "
                f"embedding='{self.embedding_model or 'hash-fallback'}' (loads on first use)"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Chroma RAG service: {exc}") from exc

    async def ingest_text(
        self,
        text: str,
        memory_type: str,
        intent: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        salience: float = 0.5,
    ) -> int:
        """Ingest text into vector store using type-aware chunking."""
        if not text or not text.strip() or self._collection is None:
            return 0

        chunks = self._chunk_text(text=text, memory_type=memory_type)
        if not chunks:
            return 0

        now_ts = datetime.now(timezone.utc).timestamp()
        base_meta = dict(metadata or {})
        base_meta.update(
            {
                "memory_type": memory_type,
                "intent": intent,
                "salience": float(max(0.0, min(1.0, salience))),
                "timestamp": now_ts,
                "embedding_model": self._embedding_name,
            }
        )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            chunk_meta = dict(base_meta)
            chunk_meta["chunk_index"] = idx
            chunk_meta["chunk_count"] = len(chunks)

            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append(chunk_meta)

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(chunks)

    async def retrieve(
        self,
        query: str,
        intent: str = "",
        top_k: int = 8,
        candidate_k: int = 40,
        memory_type: Optional[str] = None,
        mmr_lambda: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """Retrieve optimized results with score fusion + MMR diversification."""
        if self._collection is None or not query.strip():
            return []

        where: Optional[Dict[str, Any]] = None
        if memory_type:
            where = {"memory_type": memory_type}

        raw = self._collection.query(
            query_texts=[query],
            n_results=max(top_k, candidate_k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        if not docs:
            return []

        now = datetime.now(timezone.utc).timestamp()
        candidates: List[RetrievedChunk] = []

        ids = (raw.get("ids") or [[]])[0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            meta = dict(meta or {})
            similarity = max(0.0, 1.0 - float(dist or 1.0))
            salience = float(meta.get("salience", 0.5))
            ts = float(meta.get("timestamp", now))
            recency = self._recency_score(now, ts)
            intent_match = 1.0 if intent and str(meta.get("intent", "")) == intent else 0.0

            final_score = (
                0.60 * similarity
                + 0.20 * recency
                + 0.15 * salience
                + 0.05 * intent_match
            )

            candidates.append(
                RetrievedChunk(
                    id=doc_id,
                    text=doc,
                    metadata=meta,
                    similarity=similarity,
                    final_score=final_score,
                )
            )

        # Score sort before MMR
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        diversified = self._mmr_select(candidates, top_k=top_k, lambda_mult=mmr_lambda)

        # One line per retrieval showing what matched and how strongly. This
        # is the evidence that recall is semantic rather than keyword-ish:
        # a high similarity between a query and a document sharing no words
        # is only possible with real embeddings.
        if diversified:
            preview = "; ".join(
                f"sim={item.similarity:.3f} score={item.final_score:.3f} :: {item.text[:60]}"
                for item in diversified
            )
            logger.info(
                f"[RAG] retrieve embedding={self._embedding_name} "
                f"query={query[:60]!r} hits={len(diversified)}/{len(candidates)} | {preview}"
            )

        return [
            {
                "id": item.id,
                "text": item.text,
                "metadata": item.metadata,
                "similarity": item.similarity,
                "score": item.final_score,
            }
            for item in diversified
        ]

    async def delete(self, ids: List[str]) -> int:
        """Delete specific chunks by id (see the "id" field `retrieve()` returns)."""
        if not ids or self._collection is None:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    async def assemble_context(
        self,
        query: str,
        retrieved: List[Dict[str, Any]],
        token_budget: int = 1400,
    ) -> str:
        """Pack retrieved chunks into a citation-friendly context window."""
        if not retrieved:
            return ""

        budget = max(200, int(token_budget))
        used = 0
        lines: List[str] = [f"Query: {query}", "Relevant memory:"]

        for i, item in enumerate(retrieved, start=1):
            txt = str(item.get("text", "")).strip()
            if not txt:
                continue
            tok = self._estimate_tokens(txt)
            if used + tok > budget:
                break

            meta = item.get("metadata", {}) or {}
            intent = meta.get("intent", "")
            ts = meta.get("timestamp", "")
            lines.append(f"[{i}] ({intent} @ {ts}) {txt}")
            used += tok

        return "\n".join(lines)

    def close(self) -> None:
        """Close service resources (no-op for current Chroma client)."""
        self._collection = None
        self._client = None

    def _chunk_text(self, text: str, memory_type: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        # Atomic short text chunks for structured memory items
        if memory_type in {"short_term", "long_term"} and len(words) <= 80:
            return [text.strip()]

        chunks: List[str] = []
        size = max(100, int(self.chunk_size_tokens * 0.75))  # word proxy
        overlap = max(0, int(self.chunk_overlap_tokens * 0.75))

        step = max(1, size - overlap)
        start = 0
        while start < len(words):
            chunk_words = words[start : start + size]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words).strip())
            start += step

        return chunks

    def _recency_score(self, now_ts: float, item_ts: float) -> float:
        age_days = max(0.0, (now_ts - item_ts) / 86400.0)
        half_life = self.recency_half_life_days
        # Exponential decay: 0.5^(age/half_life)
        return float(pow(0.5, age_days / half_life))

    def _mmr_select(
        self,
        candidates: List[RetrievedChunk],
        top_k: int,
        lambda_mult: float,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        selected: List[RetrievedChunk] = [candidates[0]]
        remaining = candidates[1:]

        while remaining and len(selected) < top_k:
            best_idx = 0
            best_score = -10.0

            for idx, cand in enumerate(remaining):
                relevance = cand.final_score
                diversity_penalty = max(
                    (self._text_similarity(cand.text, sel.text) for sel in selected),
                    default=0.0,
                )
                mmr = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx

            selected.append(remaining.pop(best_idx))

        return selected[:top_k]

    def _text_similarity(self, a: str, b: str) -> float:
        a_set = set(a.lower().split())
        b_set = set(b.lower().split())
        if not a_set or not b_set:
            return 0.0
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        return inter / union if union else 0.0

    def _estimate_tokens(self, text: str) -> int:
        # Practical approximation for English text
        return max(1, int(len(text.split()) * 1.3))
