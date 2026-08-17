#!/usr/bin/env python3
"""
One-time migration: re-embed the Chroma memory store with local MiniLM
embeddings (PF4), replacing the hash-fallback vectors written before it.

Why a migration is unavoidable: the old and new vectors are the same width
(384) but mean entirely different things. Hash vectors encode "which words
appear"; MiniLM vectors encode meaning. Leaving the old ones in place would
put both in one space, where a query embedded by MiniLM is compared against
hash vectors it has no relationship to — silently worse than either method
alone. So every stored document is re-encoded with the new model.

The documents and metadata themselves are preserved exactly; only the
vectors change. A timestamped backup of the whole store is taken first, and
the migration writes into a fresh collection and swaps it in only on
success, so an interrupted run cannot leave a half-migrated store.

    .venv/bin/python scripts/migrate_embeddings.py            # migrate
    .venv/bin/python scripts/migrate_embeddings.py --dry-run  # report only
    .venv/bin/python scripts/migrate_embeddings.py --force    # re-run anyway
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import load_config_dict
from rag.embeddings import DEFAULT_MODEL_NAME, build_embedding_function

#: Written into every migrated document's metadata so a second run can tell
#: what has already been converted.
MIGRATION_MARKER = "embedding_model"

BATCH_SIZE = 64


def _backup(persist_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = persist_dir.parent / f"{persist_dir.name}.backup-{stamp}"
    shutil.copytree(persist_dir, destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    parser.add_argument("--force", action="store_true", help="re-embed even documents already migrated")
    args = parser.parse_args()

    config = load_config_dict()
    vs_cfg = (config.get("memory", {}) or {}).get("vector_store", {}) or {}
    persist_dir = PROJECT_ROOT / vs_cfg.get("persist_directory", "data/chroma_memory")
    collection_name = vs_cfg.get("collection_name", "vesper_memory")
    model_name = vs_cfg.get("embedding_model") or DEFAULT_MODEL_NAME

    print("=" * 70)
    print("Chroma embedding migration — hash fallback -> local MiniLM")
    print("=" * 70)
    print(f"  store      : {persist_dir}")
    print(f"  collection : {collection_name}")
    print(f"  new model  : {model_name}")

    if not persist_dir.exists():
        print("\nNothing to migrate: the store does not exist yet. It will be")
        print("created with the new embeddings on first use.")
        return 0

    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        source = client.get_collection(collection_name)
    except Exception:
        print(f"\nNothing to migrate: no collection named '{collection_name}'.")
        return 0

    total = source.count()
    if total == 0:
        print("\nNothing to migrate: the collection is empty.")
        return 0

    existing = source.get(include=["documents", "metadatas"])
    ids: List[str] = existing.get("ids") or []
    documents: List[str] = existing.get("documents") or []
    metadatas: List[Dict[str, Any]] = [dict(m or {}) for m in (existing.get("metadatas") or [])]

    by_model = Counter(str(m.get(MIGRATION_MARKER, "unknown")) for m in metadatas)
    print(f"\n  documents  : {total}")
    for name, count in by_model.most_common():
        print(f"     {count:>4} embedded with {name!r}")

    already = sum(count for name, count in by_model.items() if name == model_name)
    if already == total and not args.force:
        print(f"\nAll {total} documents are already on '{model_name}'. Nothing to do.")
        print("(Use --force to re-embed them anyway.)")
        return 0

    if args.dry_run:
        print(f"\n[dry-run] would re-embed {total} document(s) with '{model_name}'.")
        print("[dry-run] would back up the store first. No changes written.")
        return 0

    print("\n  backing up the store before touching it...")
    backup_path = _backup(persist_dir)
    print(f"  backup     : {backup_path}")

    # Load the model up front so a failure happens before anything is written.
    print(f"\n  loading '{model_name}' (first run downloads ~90MB)...")
    embedding_fn = build_embedding_function(model_name)
    probe = embedding_fn(["warmup"])
    if getattr(embedding_fn, "degraded", False):
        print("\nABORTED: the embedding model could not be loaded, so migrating")
        print("would rewrite the store with hash vectors — exactly what we are")
        print("trying to move away from. Install it and retry:")
        print("    .venv/bin/pip install sentence-transformers")
        return 1
    print(f"  model ready (dim={len(probe[0])})")

    # Write into a temporary collection, then swap. An interrupted run leaves
    # the original untouched rather than half-converted.
    staging_name = f"{collection_name}__migrating"
    try:
        client.delete_collection(staging_name)
    except Exception:
        pass

    staging = client.create_collection(
        name=staging_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\n  re-embedding {total} document(s)...")
    migrated = 0
    for start in range(0, len(ids), BATCH_SIZE):
        stop = start + BATCH_SIZE
        batch_meta = []
        for meta in metadatas[start:stop]:
            updated = dict(meta)
            updated[MIGRATION_MARKER] = model_name
            batch_meta.append(updated)

        staging.upsert(
            ids=ids[start:stop],
            documents=documents[start:stop],
            metadatas=batch_meta,
        )
        migrated += len(ids[start:stop])
        print(f"    {migrated}/{total}")

    if staging.count() != total:
        print(f"\nABORTED: staging holds {staging.count()} of {total} documents.")
        print(f"The original collection is untouched; backup at {backup_path}")
        return 1

    # Swap: the staging collection cannot be renamed onto an existing name,
    # so drop the original only after staging is verified complete above.
    client.delete_collection(collection_name)
    staging.modify(name=collection_name)

    print("\n" + "=" * 70)
    print(f"Migrated {migrated} document(s) to '{model_name}'.")
    print(f"Backup retained at: {backup_path}")
    print("Delete the backup once you're satisfied recall looks right.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
