"""
Embedding functions for the Chroma-backed semantic memory.

Two implementations, and the difference between them is the difference
between memory that understands and memory that pattern-matches:

    LocalSentenceTransformerEmbedding — all-MiniLM-L6-v2 running locally on
        CPU. Real sentence embeddings, so "when should I not schedule
        meetings?" retrieves "I train at the gym at 6pm and it's
        non-negotiable" despite the two sharing no words.

    HashEmbeddingFunction — the previous behavior, kept as a fallback. It
        hashes each token into a fixed bucket, so two texts are "similar"
        only insofar as they reuse the same words. It never fails and never
        downloads anything, which is exactly what a fallback needs to be.

Loading is strictly lazy and process-wide cached. That matters more than it
looks: importing torch alone costs ~4s and ~360MB RSS on this machine, and
the model itself adds only ~12MB on top. A session that never touches
semantic memory should never pay that, and a session that does should pay
it exactly once.
"""

from __future__ import annotations

import hashlib
import math
import threading
from typing import Any, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

#: The model this project standardizes on: 6-layer MiniLM, 384-dim output,
#: ~90MB on disk, CPU-friendly. Chosen to stay comfortable on an 8GB machine.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

#: Both implementations emit 384-dim vectors. They are NOT interchangeable
#: for stored data despite that — same width, different meaning — which is
#: why switching requires a re-embed (see scripts/migrate_embeddings.py).
EMBEDDING_DIMENSION = 384

#: Process-wide model cache. Chroma may construct the embedding function
#: more than once (client re-init, a second collection); the underlying
#: SentenceTransformer must still load only once.
_model_lock = threading.Lock()
_model_cache: dict = {}


def _load_sentence_transformer(model_name: str) -> Any:
    """
    Load (and cache) a SentenceTransformer by name.

    Raises on failure — callers are responsible for degrading. Kept separate
    from the embedding class so the cache is shared across every instance.
    """
    with _model_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]

        # Imported here, not at module scope: this is the expensive line.
        from sentence_transformers import SentenceTransformer

        logger.info(f"[RAG] loading local embedding model '{model_name}' (first use)")
        model = SentenceTransformer(model_name, device="cpu")
        _model_cache[model_name] = model
        logger.info(f"[RAG] embedding model '{model_name}' ready")
        return model


class HashEmbeddingFunction:
    """
    Deterministic keyword-ish fallback embedding.

    Hashes each whitespace token into one of `dimension` buckets with a
    sign, then L2-normalizes. Two texts score as similar only when they
    literally share tokens — there is no notion of meaning here. It exists
    so that a missing/broken model degrades the quality of memory rather
    than taking the assistant down with it.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = max(64, int(dimension))

    @staticmethod
    def name() -> str:
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

            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)


class LocalSentenceTransformerEmbedding:
    """
    Local MiniLM sentence embeddings, lazily loaded and cached.

    On the first `__call__` the model is loaded (a few seconds, once per
    process). If that load fails for any reason — package missing, no
    network for the initial download, corrupt cache — the instance logs one
    warning and permanently degrades to `HashEmbeddingFunction`. It never
    raises, because an assistant that cannot embed should still be an
    assistant that runs.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        fallback: Optional[HashEmbeddingFunction] = None,
    ):
        self.model_name = model_name
        self._fallback = fallback or HashEmbeddingFunction()
        self._model: Any = None
        self._degraded = False

    @staticmethod
    def name() -> str:
        # Chroma persists this to validate that a collection is always read
        # with the embedding function it was written with.
        return "vesper_local_minilm"

    @property
    def degraded(self) -> bool:
        """True once a model load has failed and we're on the hash fallback."""
        return self._degraded

    @property
    def effective_name(self) -> str:
        """What actually produced the vectors — for logs and stored metadata."""
        return self._fallback.name() if self._degraded else self.model_name

    def _get_model(self) -> Any:
        if self._model is not None or self._degraded:
            return self._model
        try:
            self._model = _load_sentence_transformer(self.model_name)
        except Exception as exc:
            self._degraded = True
            self._model = None
            logger.warning(
                f"[RAG] could not load embedding model '{self.model_name}' "
                f"({exc}); degrading to hash embeddings. Semantic recall will be "
                "keyword-ish until this is fixed — `pip install sentence-transformers`."
            )
        return self._model

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        texts = list(input or [])
        if not texts:
            return []

        model = self._get_model()
        if model is None:
            return self._fallback.embed_documents(texts)

        try:
            vectors = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # cosine space == dot product
                show_progress_bar=False,
            )
            return [v.tolist() for v in vectors]
        except Exception as exc:
            # A runtime encode failure is treated like a load failure: one
            # warning, then permanently degraded, never an exception into
            # the caller's write or read path.
            self._degraded = True
            self._model = None
            logger.warning(f"[RAG] embedding call failed ({exc}); degrading to hash embeddings")
            return self._fallback.embed_documents(texts)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)


def build_embedding_function(
    model_name: Optional[str] = DEFAULT_MODEL_NAME,
) -> Any:
    """
    The embedding function the RAG service should use.

    `model_name=None` (or empty) explicitly selects the hash fallback —
    useful for tests and for anyone who wants zero model downloads.
    """
    if not model_name:
        logger.info("[RAG] embedding model disabled by config; using hash fallback")
        return HashEmbeddingFunction()
    return LocalSentenceTransformerEmbedding(model_name=model_name)
