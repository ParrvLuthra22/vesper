"""The normalized inbound message every channel adapter produces."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

TRUST_USER = "user"
TRUST_THIRD_PARTY = "third_party"
TRUSTS = (TRUST_USER, TRUST_THIRD_PARTY)

_CHANNEL_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
#: Attachment kinds an adapter may report. Content is never fetched for these (only voice notes are
#: downloaded, and the adapter turns them into typed text before this object exists).
ATTACHMENT_KINDS = ("photo", "document", "video", "audio", "animation", "sticker", "video_note", "contact",
                    "location", "poll", "other")
MAX_TEXT_CHARS = 16000


class InvalidMessage(ValueError):
    """The message does not satisfy the contract; the gateway answers 422 and processes nothing."""


@dataclass(frozen=True)
class Attachment:
    """Metadata only. No file name (it is attacker-controlled text) and no content."""

    kind: str = "other"
    mime: str = ""
    size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "mime": self.mime, "size": self.size}

    @classmethod
    def from_dict(cls, data: Any) -> "Attachment":
        if not isinstance(data, dict):
            raise InvalidMessage("attachment must be an object")
        kind = str(data.get("kind", "other"))
        if kind not in ATTACHMENT_KINDS:
            kind = "other"
        try:
            size = max(0, int(data.get("size", 0) or 0))
        except (TypeError, ValueError):
            size = 0
        return cls(kind=kind, mime=str(data.get("mime", ""))[:64], size=size)


@dataclass(frozen=True)
class InboundMessage:
    """One message from a channel, normalized.

    text          everything the planner will be shown (typed text plus any framed third-party content)
    channel       adapter name, e.g. "telegram"
    user_id       the provider's id for the sender, as a string
    trust         "user" (the owner typed or spoke all of `text`) | "third_party"
    is_forward    the message was forwarded
    attachments   metadata of files that came with it (never their content)
    trusted_text  the part the owner actually typed; only this counts as "said by the user" for the
                  Guardian's tainted-input rule. None means "all of `text`" (only valid for trust=user).
    """

    text: str
    channel: str
    user_id: str
    trust: str = TRUST_USER
    is_forward: bool = False
    attachments: Tuple[Attachment, ...] = field(default_factory=tuple)
    trusted_text: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise InvalidMessage("empty text")
        if len(self.text) > MAX_TEXT_CHARS:
            raise InvalidMessage("text too long")
        if not _CHANNEL_RE.match(str(self.channel or "")):
            raise InvalidMessage("bad channel name")
        if not str(self.user_id or "").strip():
            raise InvalidMessage("missing user id")
        if self.trust not in TRUSTS:
            raise InvalidMessage("unknown trust value")  # fail closed: never default an unknown value to trusted

    @property
    def effective_trust(self) -> str:
        """Never more trusting than the content allows, whatever the adapter claimed: a forward, or
        any attachment, makes the turn third-party even if the adapter said "user"."""
        if self.trust != TRUST_USER or self.is_forward or self.attachments:
            return TRUST_THIRD_PARTY
        return TRUST_USER

    @property
    def supplied_text(self) -> str:
        """What counts as said by the owner for the tainted-input rule."""
        if self.effective_trust == TRUST_USER:
            return self.text
        return self.trusted_text or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "channel": self.channel, "user_id": str(self.user_id), "trust": self.trust,
            "is_forward": self.is_forward, "attachments": [a.to_dict() for a in self.attachments],
            "trusted_text": self.trusted_text,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "InboundMessage":
        if not isinstance(data, dict):
            raise InvalidMessage("message must be an object")
        atts = data.get("attachments") or []
        if not isinstance(atts, list):
            raise InvalidMessage("attachments must be a list")
        trusted = data.get("trusted_text")
        return cls(
            text=data.get("text") if isinstance(data.get("text"), str) else "",
            channel=str(data.get("channel", "")),
            user_id=str(data.get("user_id", "")),
            trust=str(data.get("trust", "")),          # a missing trust is "" -> rejected, not defaulted
            is_forward=bool(data.get("is_forward", False)),
            attachments=tuple(Attachment.from_dict(a) for a in atts[:16]),
            trusted_text=trusted if isinstance(trusted, str) else None,
        )
