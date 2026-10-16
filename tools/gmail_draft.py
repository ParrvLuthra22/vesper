"""
gmail_create_draft — prepare an email DRAFT in Gmail. It can only PREPARE: nothing is ever delivered to anyone.

What makes this safe, in layers:
  * the backend interface (`DraftBackend`) has exactly one method, `create_draft`. There is no other call this
    module can make, and a test fails the build if the module's source ever mentions the verbs for delivering mail;
  * tier "draft": an explicit approval every time, from a local surface when the request came over a remote channel
    (guardian/gate.py), and a recipient the user did not say raises the tier one rung (ToolSpec.recipient_args);
  * the approval card shows the COMPLETE draft — every recipient, the subject and the whole body — and the arguments
    are normalised in place, so what is approved is byte-for-byte what is saved;
  * hard caps and header-injection checks run before anything reaches Gmail.

"Draft a reply to the one from Priya": the planner finds the thread with the read-only mail tools, composes the
body, and calls this tool; the confirmation reads the draft back and asks "Want me to save this as a draft?".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from email.utils import parseaddr
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

from tools.registry import ToolRegistry, ToolSpec
from utils.logger import get_logger

logger = get_logger(__name__)

TOOL_NAME = "gmail_create_draft"
MAX_SUBJECT_CHARS = 200
MAX_BODY_CHARS = 8000
MAX_RECIPIENTS = 5                  # To + Cc together
MAX_ADDRESS_CHARS = 254

_ADDRESS = re.compile(r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+$")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LINEBREAK = re.compile(r"[\r\n  ]")


class DraftRejected(ValueError):
    """The draft breaks a rule (cap, header injection, bad address). The message is safe to show the user."""


@dataclass
class Draft:
    to: List[str]
    subject: str
    body: str
    cc: List[str] = field(default_factory=list)
    thread_id: str = ""

    @property
    def recipients(self) -> List[str]:
        return self.to + self.cc

    def as_arguments(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"to": list(self.to), "subject": self.subject, "body": self.body}
        if self.cc:
            out["cc"] = list(self.cc)
        if self.thread_id:
            out["thread_id"] = self.thread_id
        return out


class DraftBackend(Protocol):
    """The only thing this tool can ask of Gmail: store a draft. It returns the draft's id."""

    async def create_draft(self, draft: Draft) -> str: ...


class DraftUnavailable(RuntimeError):
    """Gmail is not connected (or the draft tool is not exposed by the mail server)."""


# ------------------------------------------------------------------ validation

def _as_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [p for p in re.split(r"[;,]", value) if p.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    raise DraftRejected("recipients must be a list of email addresses")


def _address(raw: str) -> str:
    if _CONTROL.search(raw) or _LINEBREAK.search(raw):
        raise DraftRejected("a recipient contains a line break or control character")
    _, addr = parseaddr(raw.strip())
    addr = addr.strip().lower()
    if not addr or len(addr) > MAX_ADDRESS_CHARS or not _ADDRESS.match(addr):
        raise DraftRejected(f"'{raw.strip()[:60]}' is not a valid email address")
    return addr


def validate_draft(arguments: Dict[str, Any]) -> Draft:
    """Normalise and check the arguments. Raises DraftRejected with a reason fit to show the user."""
    to = [_address(a) for a in _as_list(arguments.get("to"))]
    cc = [_address(a) for a in _as_list(arguments.get("cc"))]
    seen: set = set()
    to = [a for a in to if not (a in seen or seen.add(a))]
    cc = [a for a in cc if not (a in seen or seen.add(a))]
    if not to:
        raise DraftRejected("a draft needs at least one recipient in 'to'")
    if len(to) + len(cc) > MAX_RECIPIENTS:
        raise DraftRejected(f"too many recipients ({len(to) + len(cc)}; the limit is {MAX_RECIPIENTS})")

    subject = str(arguments.get("subject") or "")
    if _LINEBREAK.search(subject) or _CONTROL.search(subject):
        raise DraftRejected("the subject contains a line break or control character")
    subject = " ".join(subject.split())
    if not subject:
        raise DraftRejected("a draft needs a subject")
    if len(subject) > MAX_SUBJECT_CHARS:
        raise DraftRejected(f"the subject is too long ({len(subject)} characters; the limit is {MAX_SUBJECT_CHARS})")

    body = _CONTROL.sub("", str(arguments.get("body") or "")).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not body:
        raise DraftRejected("a draft needs a body")
    if len(body) > MAX_BODY_CHARS:
        raise DraftRejected(f"the body is too long ({len(body)} characters; the limit is {MAX_BODY_CHARS})")

    thread_id = str(arguments.get("thread_id") or "").strip()
    if thread_id and not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", thread_id):
        raise DraftRejected("the thread id is not valid")
    return Draft(to=to, cc=cc, subject=subject, body=body, thread_id=thread_id)


# ------------------------------------------------------------------ the approval card

def render_confirmation(draft: Draft) -> str:
    words = len(draft.body.split())
    lines = [
        f"Save an email DRAFT in Gmail — nothing is delivered to anyone. {words} words.",
        f"To: {', '.join(draft.to)}",
    ]
    if draft.cc:
        lines.append(f"Cc: {', '.join(draft.cc)}")
    if draft.thread_id:
        lines.append("In reply to an existing thread.")
    lines += [f"Subject: {draft.subject}", "Body:", draft.body, "", "Want me to save this as a draft?"]
    return "\n".join(lines)


def spoken_readback(draft: Draft) -> str:
    """The two-sentence version for a speaker: who it is to, what it is about, how long — then the question."""
    who = draft.to[0] + (f" and {len(draft.recipients) - 1} more" if len(draft.recipients) > 1 else "")
    return (f"A draft to {who}, subject {draft.subject}, {len(draft.body.split())} words. "
            "Want me to save this as a draft?")


async def confirm_summary(arguments: Dict[str, Any]) -> str:
    """Shown by the Guardian. Normalises `arguments` in place so the approved text is exactly what is saved."""
    try:
        draft = validate_draft(arguments)
    except DraftRejected as exc:
        return f"This draft will be refused: {exc}. Nothing will be saved."
    arguments.clear()
    arguments.update(draft.as_arguments())
    return render_confirmation(draft)


# ------------------------------------------------------------------ the tool

def make_handler(backend: DraftBackend) -> Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[str]]:
    async def handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        try:
            draft = validate_draft(arguments)
        except DraftRejected as exc:
            return f"Error: {exc}"
        try:
            draft_id = await backend.create_draft(draft)
        except DraftUnavailable as exc:
            return f"Error: {exc}"
        return (f"Draft saved in Gmail (id={draft_id}) for {', '.join(draft.recipients)}. "
                "It is only a draft: it has NOT been delivered, and you would finish it from Gmail yourself.")
    return handler


def make_tool(backend: DraftBackend) -> ToolSpec:
    return ToolSpec(
        name=TOOL_NAME,
        description=(
            "Save an email DRAFT in Gmail (to, subject, body; optional cc and thread_id to reply inside a thread). "
            "It only prepares a draft for the user to review — it cannot deliver anything. Use for \"draft a reply to "
            "the one from X\": find the thread with the read-only mail tools first, compose the full reply body, then "
            "call this. The user is shown the whole draft and must approve saving it."),
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "array", "items": {"type": "string"}, "description": "recipient email addresses"},
                "cc": {"type": "array", "items": {"type": "string"}, "description": "optional cc addresses"},
                "subject": {"type": "string"},
                "body": {"type": "string", "description": "the complete reply text, ready to read"},
                "thread_id": {"type": "string", "description": "optional Gmail thread id to reply inside"},
            },
            "required": ["to", "subject", "body"],
        },
        tier="draft",
        handler=make_handler(backend),
        category="mail-draft",
        confirm_summary=confirm_summary,
        recipient_args=("to", "cc"),
    )


class RegistryDraftBackend:
    """Stores the draft through the Gmail MCP server's hidden `create_draft` tool. That tool is not offered to the
    planner (settings.yaml hidden_tools): the only caller is this module, after the Guardian's approval."""

    HIDDEN_TOOL = "create_draft"

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def create_draft(self, draft: Draft) -> str:
        spec = self._registry.get(self.HIDDEN_TOOL)
        if spec is None or spec.handler is None:
            raise DraftUnavailable("Gmail isn't connected, so I can't save a draft right now")
        result = await spec.handler(draft.as_arguments(), {})
        return str(result).strip()


def register(registry: ToolRegistry, backend: Optional[DraftBackend] = None) -> ToolSpec:
    """Register gmail_create_draft (idempotent)."""
    existing = registry.get(TOOL_NAME)
    if existing is not None:
        return existing
    return registry.register(make_tool(backend or RegistryDraftBackend(registry)))
