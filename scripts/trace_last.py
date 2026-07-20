#!/usr/bin/env python3
"""
`trace last` — prints the last turn's run tree to the terminal.

Stub for now: the P08 CLI will expose this as the /trace command. Reads
data/traces/traces.jsonl directly (the file tracing.Tracer always writes,
regardless of whether LangSmith is configured).

Usage:
    python scripts/trace_last.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracing.last_trace import print_last_turn

if __name__ == "__main__":
    print_last_turn()
