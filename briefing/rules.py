"""
Feedback rules: `vesper briefing --mark <item> priority|ignore`.

A rule is one line of plain data — (scope, value, action) — applied by the scorer:

    scope   sender  an exact address     |  domain  everyone at a domain (and subdomains)
    action  priority  +weights.rule_priority AND the bulk/newsletter penalties are waived
            ignore    weights.rule_ignore (a large negative): never priority

Stored in data/briefing_rules.json — human-readable, git-ignored, survives a cache wipe
(the cache is disposable; your decisions are not). There is no model and no learning:
the file is the whole behaviour. If an address rule and a domain rule both match, the
address rule wins.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from email.utils import parseaddr
from pathlib import Path
from typing import Dict, List, Optional

SCOPES = ("sender", "domain")
ACTIONS = ("priority", "ignore")


@dataclass
class Rule:
    id: int
    scope: str       # "sender" | "domain"
    value: str       # lowercase address or domain
    action: str      # "priority" | "ignore"
    created: float

    def describe(self) -> str:
        who = f"mail from {self.value}" if self.scope == "sender" else f"all mail from @{self.value} (and its subdomains)"
        return f"#{self.id}: {who} -> {self.action.upper()}"


class RuleSet:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rules: List[Rule] = []
        self.reload()

    def reload(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.rules = [Rule(**r) for r in data.get("rules", [])]
        except (OSError, ValueError, TypeError, AttributeError):   # unreadable, not JSON, or the wrong shape
            self.rules = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1,
                   "note": "Written by `vesper briefing --mark`. Plain data; edit or delete freely.",
                   "rules": [asdict(r) for r in self.rules]}
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".rules-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def add(self, scope: str, value: str, action: str, now: Optional[float] = None) -> Rule:
        if scope not in SCOPES or action not in ACTIONS:
            raise ValueError(f"scope must be one of {SCOPES}, action one of {ACTIONS}")
        value = value.strip().lower().lstrip("@")
        if not value:
            raise ValueError("empty sender/domain")
        # one rule per (scope, value): marking again replaces the old decision
        self.rules = [r for r in self.rules if not (r.scope == scope and r.value == value)]
        rule = Rule(id=max([r.id for r in self.rules], default=0) + 1, scope=scope, value=value,
                    action=action, created=time.time() if now is None else now)
        self.rules.append(rule)
        self._save()
        return rule

    def remove(self, rule_id: int) -> Optional[Rule]:
        for r in self.rules:
            if r.id == rule_id:
                self.rules = [x for x in self.rules if x.id != rule_id]
                self._save()
                return r
        return None

    def match(self, sender: str) -> Optional[Rule]:
        """The rule that applies to a From header, or None. Address beats domain."""
        address = parseaddr(sender or "")[1].lower()
        if not address:
            return None
        domain = address.rsplit("@", 1)[1] if "@" in address else ""
        by_sender = [r for r in self.rules if r.scope == "sender" and r.value == address]
        if by_sender:
            return by_sender[-1]
        by_domain = [r for r in self.rules if r.scope == "domain" and domain and (domain == r.value or domain.endswith("." + r.value))]
        return by_domain[-1] if by_domain else None
