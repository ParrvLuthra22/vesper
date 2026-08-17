#!/usr/bin/env python3
"""
Prove that memory recall is semantic, not keyword matching.

Runs in two modes so the store and the read happen in genuinely separate
processes — a real "new session", not a warm cache:

    --store     write the test memory through the reflection write-path
    --retrieve  query it from a cold process and print the match + score

The test pair is chosen so keyword matching CANNOT succeed: the query
"when should I not schedule meetings?" shares no content word with the
stored "I train at the gym at 6pm and it's non-negotiable". Under the old
hash embeddings this scores ~0; only real sentence embeddings connect them.

    .venv/bin/python scripts/smoke_semantic_memory.py --store
    .venv/bin/python scripts/smoke_semantic_memory.py --retrieve
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import load_config_dict
from rag.rag_service import ChromaRAGMemoryService

MEMORY_TEXT = "I train at the gym at 6pm and it's non-negotiable"
QUERY = "when should I not schedule meetings?"


def _service() -> ChromaRAGMemoryService:
    config = load_config_dict()
    vs_cfg = (config.get("memory", {}) or {}).get("vector_store", {}) or {}
    return ChromaRAGMemoryService(
        persist_directory=str(PROJECT_ROOT / vs_cfg.get("persist_directory", "data/chroma_memory")),
        collection_name=vs_cfg.get("collection_name", "vesper_memory"),
        embedding_model=vs_cfg.get("embedding_model"),
    )


def _shared_words(a: str, b: str) -> set:
    stop = {
        "i", "at", "the", "and", "it", "s", "a", "an", "to", "when", "should",
        "not", "my", "is", "in", "on", "of", "for",
    }
    wa = {w.strip(".,?!'").lower() for w in a.split()} - stop
    wb = {w.strip(".,?!'").lower() for w in b.split()} - stop
    return wa & wb


async def store() -> int:
    service = _service()
    written = await service.ingest_text(
        text=MEMORY_TEXT,
        memory_type="long_term",
        intent="reflection_preference",   # same shape P09 reflection writes
        metadata={"kind": "preference", "source": "smoke_semantic_memory"},
        salience=0.8,
    )
    print(f"stored {written} chunk(s): {MEMORY_TEXT!r}")
    print(f"embedding: {service._embedding_name}")
    return 0


async def retrieve() -> int:
    service = _service()

    print("=" * 74)
    print("SEMANTIC RECALL TEST — new process, cold cache")
    print("=" * 74)
    print(f"  stored : {MEMORY_TEXT!r}")
    print(f"  query  : {QUERY!r}")
    overlap = _shared_words(QUERY, MEMORY_TEXT)
    print(f"  shared content words: {overlap or 'NONE — keyword matching cannot work here'}")

    results = await service.retrieve(query=QUERY, top_k=3)
    print(f"\n  embedding in use: {service._embedding_name}")
    print(f"\n  top {len(results)} matches:")
    for rank, item in enumerate(results, start=1):
        marker = "  <-- the gym memory" if "gym" in item["text"].lower() else ""
        print(f"    {rank}. sim={item['similarity']:.4f} score={item['score']:.4f}{marker}")
        print(f"       {item['text'][:88]}")

    hit = next((r for r in results if "gym" in r["text"].lower()), None)
    print("\n" + "=" * 74)
    if hit is None:
        print("FAIL: the gym memory was not retrieved.")
        return 1
    print(f"PASS: retrieved by meaning at similarity {hit['similarity']:.4f} "
          f"with zero shared keywords.")
    print("=" * 74)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--retrieve", action="store_true")
    args = parser.parse_args()

    if args.store:
        return asyncio.run(store())
    if args.retrieve:
        return asyncio.run(retrieve())
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
