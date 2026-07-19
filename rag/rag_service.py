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

import hashlib
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.logger import get_logger


logger = get_logger(__name__)


class _HashEmbeddingFunction:
    """
    Deterministic local fallback embedding function.

    This is not as semantically strong as transformer embeddings, but keeps the
    system operational when optional embedding backends are unavailable.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = max(64, int(dimension))

    def name(self) -> str:
        return "hash_fallback_embedding"

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in input:
            vec = [0.0] * self.dimension
            for token in (text or "").lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign

            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    similarity: float
    final_score: float


class ChromaRAGMemoryService:
    """Chroma-backed semantic memory service."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "jarvis_memory",
        embedding_model: Optional[str] = None,
        chunk_size_tokens: int = 400,
        chunk_overlap_tokens: int = 64,
        recency_half_life_days: float = 7.0,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.recency_half_life_days = max(1.0, recency_half_life_days)

        self._client = None
        self._collection = None
        self._embedding_name = "hash-fallback"

        self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialize Chroma client and collection with safe embedding fallback."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            embedding_fn = None

            if self.embedding_model:
                try:
                    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=self.embedding_model
                    )
                    self._embedding_name = self.embedding_model
                except Exception as exc:
                    logger.warning(
                        f"[RAG] Failed to init sentence-transformer embedding '{self.embedding_model}', "
                        f"falling back to hash embedding. Error: {exc}"
                    )

            if embedding_fn is None:
                embedding_fn = _HashEmbeddingFunction()

            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                f"[RAG] Chroma initialized at '{self.persist_directory}' "
                f"collection='{self.collection_name}' embedding='{self._embedding_name}'"
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

        for doc, meta, dist in zip(docs, metas, dists):
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
                    text=doc,
                    metadata=meta,
                    similarity=similarity,
                    final_score=final_score,
                )
            )

        # Score sort before MMR
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        diversified = self._mmr_select(candidates, top_k=top_k, lambda_mult=mmr_lambda)

        return [
            {
                "text": item.text,
                "metadata": item.metadata,
                "similarity": item.similarity,
                "score": item.final_score,
            }
            for item in diversified
        ]

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
