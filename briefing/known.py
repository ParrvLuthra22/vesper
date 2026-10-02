"""
Known correspondents: who I have written to, derived from SENT mail.

Input is the Gmail `sent_summary` read tool's output — recipient addresses (To/Cc headers)
of recent sent messages, with a per-address message count. Output is just two small
maps, addresses and domains, each with a count. **No message body, subject or snippet is
read or stored** — only header addresses.

Rules of derivation:
  * an address is dropped if it looks automated (noreply patterns): replying to a
    notification address does not make a service a correspondent;
  * a DOMAIN is only recorded for non-public mailbox providers: having written to
    someone@gmail.com must not make every Gmail sender "known";
  * the window is `collect.sent_days` (default 365). A 60-day window was empty on the
    real account (it sends ~1 mail a fortnight), which is why the default is longer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from email.utils import parseaddr
from typing import Any, Dict, Iterable, List, Optional, Set

from briefing.config import BriefingConfig

META_KEY = "known_correspondents"


@dataclass
class KnownSet:
    addresses: Dict[str, int] = field(default_factory=dict)   # address -> messages I sent it
    domains: Dict[str, int] = field(default_factory=dict)     # non-public domain -> messages sent there
    window_days: int = 0
    sent_messages: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"addresses": self.addresses, "domains": self.domains, "window_days": self.window_days,
                "sent_messages": self.sent_messages}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "KnownSet":
        d = data or {}
        return cls(addresses={str(k): int(v) for k, v in (d.get("addresses") or {}).items()},
                   domains={str(k): int(v) for k, v in (d.get("domains") or {}).items()},
                   window_days=int(d.get("window_days") or 0), sent_messages=int(d.get("sent_messages") or 0))


def domain_of(address: str) -> str:
    return address.rsplit("@", 1)[1].lower() if "@" in address else ""


def is_public_domain(domain: str, public: Iterable[str]) -> bool:
    return any(domain == p or domain.endswith("." + p) for p in public)


def derive_known(sent_summary: Optional[Dict[str, Any]], cfg: BriefingConfig) -> KnownSet:
    """Filter + aggregate a sent_summary into a KnownSet."""
    summary = sent_summary or {}
    counts: Dict[str, int] = {}
    raw_counts = summary.get("recipient_counts")
    if isinstance(raw_counts, dict):
        counts = {str(a): int(n) for a, n in raw_counts.items()}
    else:  # an older cache entry: addresses only
        counts = {str(a): 1 for a in (summary.get("recipients") or [])}

    noreply = [re.compile(p, re.IGNORECASE) for p in cfg.pattern_noreply]
    known = KnownSet(window_days=cfg.sent_days, sent_messages=int(summary.get("message_count") or 0))
    for raw, n in counts.items():
        address = parseaddr(raw)[1].lower() or raw.lower()
        if "@" not in address or any(rx.search(address) for rx in noreply):
            continue
        known.addresses[address] = known.addresses.get(address, 0) + max(1, n)
        domain = domain_of(address)
        if domain and not is_public_domain(domain, cfg.public_domains):
            known.domains[domain] = known.domains.get(domain, 0) + max(1, n)
    return known


def domain_matches(domain: str, known_domains: Iterable[str]) -> bool:
    """A sender at mail.college.edu matches a known college.edu."""
    return any(domain == k or domain.endswith("." + k) for k in known_domains)
