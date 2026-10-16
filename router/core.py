"""
The skill router: normalised phrases + anchored patterns, no model.

    normalize(text)       lowercase, no punctuation/apostrophes, filler words ("uh", "so", "can you", "please") dropped
    SkillRouter.match()   every skill's patterns are tried with re.fullmatch on the normalised text. Exactly ONE skill
                          may claim an utterance; zero claims or more than one -> None and the utterance falls
                          through to the planner untouched. Any extra word the patterns do not expect also falls through.
    SkillRouter.explain() the same walk, narrated: what was stripped, which rule matched, or why nothing did.

Matching is deliberately narrow (full-string, whole-phrase): the cost of a miss is one ordinary LLM turn, the cost of a
wrong match is an action nobody asked for. Skills that DO something (open/close an app, volume) additionally validate
their arguments (a known app, a volume 0-100) before they may claim an utterance, and every skill runs through the
Guardian at max(skill tier, tool tier).
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Pattern, Sequence, Tuple

# ----------------------------------------------------------------------------- normalisation

#: Stripped from the START, repeatedly. Politeness and disfluency only — nothing that carries meaning.
_LEAD = re.compile(
    r"^(?:uh+|um+|er+|erm|ah+|hmm+|hm+|so|well|like|okay|ok|alright|hey|hi|hello|yo|please|pls|plz|kindly|vesper|jarvis|"
    r"friday|can you|could you|would you|will you|can u|could u|would you mind|do me a favou?r and|just|quickly|"
    r"go ahead and|i want you to|i need you to|i would like you to|id like you to|yaar|bhai|arre|arey|acha|accha|haan|"
    r"toh|zara|ek baar|bas)\b\s*")
#: Stripped from the END, repeatedly.
_TRAIL = re.compile(
    r"\s+(?:please|pls|plz|thanks|thank you|thankyou|for me|sir|vesper|jarvis|yaar|bhai|na|ji|now|right now|quickly|"
    r"real quick|if you can|will you|can you)$")

_APOSTROPHES = re.compile(r"['’`´]")
_NON_WORD = re.compile(r"[^a-z0-9% ]+")
_SPACES = re.compile(r"\s+")


def normalize(text: object) -> Tuple[str, List[str]]:
    """(normalised text, the filler fragments that were stripped, in order)."""
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    s = s.replace("&", " and ")
    s = _APOSTROPHES.sub("", s)
    s = _NON_WORD.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    stripped: List[str] = []
    while True:
        m = _LEAD.match(s)
        if not m or not m.group(0).strip():
            break
        stripped.append(m.group(0).strip())
        s = s[m.end():]
    while True:
        m = _TRAIL.search(s)
        if not m:
            break
        stripped.append(m.group(0).strip())
        s = s[: m.start()]
    return s.strip(), stripped


# ----------------------------------------------------------------------------- model

TIER_ORDER = ("safe", "draft", "confirm", "dangerous")


def max_tier(a: str, b: str) -> str:
    return a if TIER_ORDER.index(a) >= TIER_ORDER.index(b) else b


@dataclass
class SkillRuntime:
    """What a skill handler may use. It reaches tools ONLY through `call`, which runs the Guardian first."""

    call: Callable[[str, Dict[str, Any]], Awaitable[str]]
    briefing: Any = None                     # BriefingService or None
    clock: Callable[[], datetime] = datetime.now
    address_as: str = "Sir"


@dataclass
class SkillResult:
    text: str
    #: True if the text quotes third-party content (event titles, senders): the turn is then marked tainted.
    tainted: bool = False


class SkillUnavailable(RuntimeError):
    """Raised by a handler BEFORE it has done anything: the data or tool it needs is not there. The runner then lets
    the utterance fall through to the planner, which may have another way to answer."""


Handler = Callable[["Match", SkillRuntime], Awaitable[SkillResult]]
#: validator(groups, ctx) -> args dict to accept, or None to reject this match (the utterance then falls through).
Validator = Callable[[Dict[str, str], "RouterContext"], Optional[Dict[str, Any]]]


@dataclass
class RouterContext:
    """What validators may know without side effects: the clock and the installed-app list."""

    now: datetime = field(default_factory=datetime.now)
    resolve_app: Callable[[str], Optional[str]] = lambda name: None
    briefing_phrases: Sequence[str] = ()
    morning_until_hour: int = 12


@dataclass(frozen=True)
class Rule:
    id: str
    regex: Pattern[str]
    validate: Optional[Validator] = None


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    rules: Tuple[Rule, ...]
    handler: Handler
    #: The permission floor for everything this skill does (the Guardian uses max(skill, tool)).
    tier: str = "safe"
    #: The reply quotes third-party text.
    untrusted: bool = False
    #: The skill changes something (opens/closes an app, sets the volume). If a mutating skill fails part-way the user
    #: gets an honest error; a read-only skill that fails simply falls through to the planner.
    mutates: bool = False


@dataclass
class Match:
    skill: Skill
    rule: Rule
    args: Dict[str, Any]
    normalized: str
    stripped: List[str]


@dataclass
class Explanation:
    utterance: str
    normalized: str
    stripped: List[str]
    matched: Optional[Match]
    verdict: str                              # "matched" | "no skill matched" | "ambiguous" | "disabled"
    candidates: List[Tuple[str, str]] = field(default_factory=list)       # (skill, rule) that claimed it
    rejected: List[Tuple[str, str]] = field(default_factory=list)         # (skill, rule) matched but failed validation

    def render(self) -> str:
        lines = [f'utterance:   "{self.utterance}"',
                 f'normalised:  "{self.normalized}"' + (f"   (dropped: {', '.join(self.stripped)})" if self.stripped else "")]
        if self.matched is not None:
            m = self.matched
            lines += [f"verdict:     MATCHED -> skill '{m.skill.name}' (rule '{m.rule.id}', tier floor {m.skill.tier})",
                      f"arguments:   {m.args if m.args else '(none)'}",
                      f"why:         the whole normalised phrase fits the pattern  {m.rule.regex.pattern[:110]}"]
        elif self.verdict == "ambiguous":
            lines += ["verdict:     FALL THROUGH to the planner: more than one skill claims it -> "
                      + ", ".join(f"{s}/{r}" for s, r in self.candidates)]
        else:
            lines.append(f"verdict:     FALL THROUGH to the planner ({self.verdict})")
        if self.rejected:
            lines.append("rejected:    " + ", ".join(f"{s}/{r} (matched the words, failed validation)" for s, r in self.rejected))
        return "\n".join(lines)


# ----------------------------------------------------------------------------- the router

class SkillRouter:
    def __init__(self, skills: Sequence[Skill], ctx_factory: Callable[[], RouterContext],
                 disabled: Sequence[str] = ()):
        names = [s.name for s in skills]
        if len(set(names)) != len(names):
            raise ValueError("skill names must be unique")
        self._skills = list(skills)
        self._ctx_factory = ctx_factory
        self._disabled = set(disabled)

    @property
    def skills(self) -> List[Skill]:
        return list(self._skills)

    def explain(self, text: str, ctx: Optional[RouterContext] = None) -> Explanation:
        norm, stripped = normalize(text)
        ctx = ctx or self._ctx_factory()
        if not norm:
            return Explanation(str(text), norm, stripped, None, "empty after normalisation")
        claimed: List[Tuple[Skill, Rule, Dict[str, Any]]] = []
        rejected: List[Tuple[str, str]] = []
        for skill in self._skills:
            if skill.name in self._disabled:
                continue
            for rule in skill.rules:
                m = rule.regex.fullmatch(norm)
                if not m:
                    continue
                groups = {k: v for k, v in m.groupdict().items() if v is not None}
                args = rule.validate(groups, ctx) if rule.validate else dict(groups)
                if args is None:
                    rejected.append((skill.name, rule.id))
                else:
                    claimed.append((skill, rule, args))
        owners = {s.name for s, _, _ in claimed}
        if len(owners) > 1:
            return Explanation(str(text), norm, stripped, None, "ambiguous",
                               [(s.name, r.id) for s, r, _ in claimed], rejected)
        if not claimed:
            return Explanation(str(text), norm, stripped, None, "no skill matched", [], rejected)
        skill, rule, args = claimed[0]
        return Explanation(str(text), norm, stripped, Match(skill, rule, args, norm, stripped), "matched",
                           [(skill.name, rule.id)], rejected)

    def match(self, text: str, ctx: Optional[RouterContext] = None) -> Optional[Match]:
        return self.explain(text, ctx).matched

    def measure(self, text: str, repeat: int = 200) -> float:
        """Median wall time of match() for one utterance, in microseconds."""
        samples = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            self.match(text)
            samples.append((time.perf_counter() - t0) * 1e6)
        samples.sort()
        return samples[len(samples) // 2]
