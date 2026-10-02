"""Briefing engine: background collectors -> deterministic scorer -> SQLite cache ->
compact, spoken-friendly briefing. See docs/BRIEFING.md.

Read-only by construction: collectors only ever call allow-listed, safe-tier,
read tools (briefing/readonly.py). Nothing here sends, modifies, archives or labels.
"""
