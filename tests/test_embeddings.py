"""
Tests for rag/embeddings.py — the local MiniLM embedding function that
replaced the hash fallback, and the fallback's own guarantees.

Tests that need the real model are marked `integration` and skip when
sentence-transformers (or its ~90MB download) isn't available, so the
default suite stays fast and offline.
"""

from __future__ import annotations

import importlib.util

import pytest

from rag.embeddings import (
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIMENSION,
    HashEmbeddingFunction,
    LocalSentenceTransformerEmbedding,
    build_embedding_function,
)

_HAS_ST = importlib.util.find_spec("sentence_transformers") is not None
_needs_model = pytest.mark.skipif(_HAS_ST is False, reason="sentence-transformers not installed")


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / ((na * nb) or 1.0)


# ---------------------------------------------------------------------------
# The hash fallback
# ---------------------------------------------------------------------------


def test_hash_embedding_is_deterministic():
    fn = HashEmbeddingFunction()
    assert fn(["hello world"]) == fn(["hello world"])


def test_hash_embedding_has_the_expected_shape():
    vectors = HashEmbeddingFunction()(["one", "two"])
    assert len(vectors) == 2
    assert all(len(v) == EMBEDDING_DIMENSION for v in vectors)


def test_hash_embedding_is_normalized():
    (vector,) = HashEmbeddingFunction()(["some words here"])
    assert abs(sum(v * v for v in vector) ** 0.5 - 1.0) < 1e-6


def test_hash_embedding_cannot_see_meaning():
    """
    The documented weakness, pinned as a test: lexically disjoint but
    semantically related texts score zero. This is the whole reason MiniLM
    replaced it as the default.
    """
    fn = HashEmbeddingFunction()
    (a,) = fn(["when should I avoid booking calls"])
    (b,) = fn(["gym session every evening, protected"])
    assert abs(_cosine(a, b)) < 1e-9


# ---------------------------------------------------------------------------
# Degradation — the fallback must never let an embedding failure escape
# ---------------------------------------------------------------------------


def test_unloadable_model_degrades_instead_of_raising():
    fn = LocalSentenceTransformerEmbedding(model_name="definitely/not-a-real-model")
    vectors = fn(["anything at all"])

    assert fn.degraded
    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIMENSION
    assert fn.effective_name == HashEmbeddingFunction.name()


def test_degraded_instance_does_not_retry_the_failed_load(monkeypatch):
    """One warning, then stop hammering a load that is not going to work."""
    calls = {"n": 0}

    def failing_loader(model_name):
        calls["n"] += 1
        raise RuntimeError("no model here")

    monkeypatch.setattr("rag.embeddings._load_sentence_transformer", failing_loader)
    fn = LocalSentenceTransformerEmbedding(model_name="whatever")

    fn(["first"])
    fn(["second"])
    fn(["third"])

    assert calls["n"] == 1


def test_runtime_encode_failure_also_degrades(monkeypatch):
    class ExplodingModel:
        def encode(self, *args, **kwargs):
            raise RuntimeError("cuda is on fire")

    monkeypatch.setattr("rag.embeddings._load_sentence_transformer", lambda name: ExplodingModel())
    fn = LocalSentenceTransformerEmbedding(model_name="whatever")

    vectors = fn(["some text"])

    assert fn.degraded
    assert len(vectors[0]) == EMBEDDING_DIMENSION


def test_empty_input_needs_no_model(monkeypatch):
    def exploding_loader(model_name):  # pragma: no cover - must never run
        raise AssertionError("model should not load for empty input")

    monkeypatch.setattr("rag.embeddings._load_sentence_transformer", exploding_loader)
    assert LocalSentenceTransformerEmbedding()([]) == []


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_build_defaults_to_the_local_model():
    assert isinstance(build_embedding_function(), LocalSentenceTransformerEmbedding)


def test_build_with_empty_model_name_selects_the_hash_fallback():
    assert isinstance(build_embedding_function(""), HashEmbeddingFunction)


def test_model_is_not_loaded_at_construction(monkeypatch):
    """
    Importing torch costs ~4s and ~360MB RSS, so a session that never touches
    semantic memory must never pay it.
    """
    def exploding_loader(model_name):  # pragma: no cover - must never run
        raise AssertionError("model loaded eagerly at construction")

    monkeypatch.setattr("rag.embeddings._load_sentence_transformer", exploding_loader)
    LocalSentenceTransformerEmbedding()  # must not raise


# ---------------------------------------------------------------------------
# The real model
# ---------------------------------------------------------------------------


@pytest.mark.integration
@_needs_model
def test_real_model_produces_normalized_384d_vectors():
    fn = LocalSentenceTransformerEmbedding(DEFAULT_MODEL_NAME)
    vectors = fn(["a sentence about scheduling"])

    assert not fn.degraded
    assert len(vectors[0]) == EMBEDDING_DIMENSION
    assert abs(sum(v * v for v in vectors[0]) ** 0.5 - 1.0) < 1e-3


@pytest.mark.integration
@_needs_model
def test_real_model_ranks_by_meaning_where_hashing_cannot():
    """
    The acceptance case: five queries against a memory bank, each phrased to
    share as little vocabulary as possible with its target. Measured on this
    machine, hashing gets 2/5 (both by stopword accident) and MiniLM 5/5.
    """
    memories = [
        "I train at the gym at 6pm and it's non-negotiable",
        "Deep work happens between 10 and 12, no interruptions",
        "I dislike vocals in music while concentrating",
        "My sister's birthday is the 3rd of March",
        "Coffee: dark roast, no sugar",
        "The car insurance renews every September",
    ]
    cases = [
        ("when should I not schedule meetings?", 0),
        ("what should I put on while I'm focusing?", 2),
        ("any important family dates coming up?", 3),
        ("how do I take my espresso?", 4),
        ("when is the vehicle policy due for renewal?", 5),
    ]

    fn = LocalSentenceTransformerEmbedding(DEFAULT_MODEL_NAME)
    memory_vectors = fn(memories)

    correct = 0
    for query, expected in cases:
        (query_vector,) = fn([query])
        scores = [_cosine(query_vector, mv) for mv in memory_vectors]
        if max(range(len(scores)), key=lambda i: scores[i]) == expected:
            correct += 1

    assert correct == len(cases), f"semantic ranking regressed: {correct}/{len(cases)}"


@pytest.mark.integration
@_needs_model
def test_gym_memory_beats_hashing_on_the_acceptance_query():
    query = "when should I not schedule meetings?"
    target = "I train at the gym at 6pm and it's non-negotiable"
    unrelated = "the wifi password is stored in the notes app"

    mini = LocalSentenceTransformerEmbedding(DEFAULT_MODEL_NAME)
    (q, t, u) = mini([query, target, unrelated])

    assert _cosine(q, t) > _cosine(q, u)
    assert _cosine(q, t) > 0.2
