#!/usr/bin/env python3
"""Phase 1 RAG smoke test: ingest and retrieve from Chroma service."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.rag_service import ChromaRAGMemoryService


async def main() -> None:
    persist_dir = Path("data/chroma_smoke")
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    svc = ChromaRAGMemoryService(
        persist_directory=str(persist_dir),
        collection_name="vesper_smoke",
        embedding_model=None,
    )

    await svc.ingest_text(
        text="User prefers concise summaries before 10 AM meetings.",
        memory_type="long_term",
        intent="PREFERENCE",
        metadata={"source": "smoke_test"},
        salience=0.9,
    )
    await svc.ingest_text(
        text="Last command was open Safari and play lo-fi music.",
        memory_type="short_term",
        intent="VOICE_INPUT",
        metadata={"source": "smoke_test"},
        salience=0.6,
    )

    hits = await svc.retrieve(
        query="What communication style does the user prefer in morning meetings?",
        intent="GENERAL_QUESTION",
        top_k=3,
    )

    context = await svc.assemble_context(
        query="user style",
        retrieved=hits,
        token_budget=400,
    )

    print(f"Retrieved hits: {len(hits)}")
    for i, h in enumerate(hits, start=1):
        print(f"{i}. score={h['score']:.4f} text={h['text'][:90]}")

    print("\nContext:\n" + context)


if __name__ == "__main__":
    asyncio.run(main())
