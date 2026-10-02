"""The Item schema every collector emits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

#: Everything a collector returns was written by someone else (a sender, an
#: invite organizer). It is data, never instructions — see briefing/sanitize.py.
RAW_TRUST = "third_party"


@dataclass
class Item:
    id: str                      # "gmail:<message id>" | "calendar:<hash>"
    source: str                  # "gmail" | "calendar"
    sender: str = ""             # raw From (name + address) | calendar name
    title: str = ""              # subject | event title
    snippet: str = ""            # truncated at collection time
    timestamp: float = 0.0       # epoch seconds: received time | event start
    thread_id: str = ""
    is_reply: bool = False       # subject starts "Re:" or the message has In-Reply-To
    start_at: Optional[float] = None   # events: epoch start
    end_at: Optional[float] = None     # events: epoch end
    all_day: bool = False
    unread: bool = True
    #: Source-specific scoring signals (mail: labels, list_id, has_list_unsubscribe,
    #: precedence, auto_submitted; calendar: calendar name).
    signals: Dict[str, Any] = field(default_factory=dict)
    raw_trust: str = RAW_TRUST

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Item":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
