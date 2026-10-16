"""Deterministic skill router: the common voice commands, answered without the LLM. See docs/ROUTER.md."""

from router.core import Explanation, Match, Skill, SkillResult, SkillRouter, SkillRuntime, normalize  # noqa: F401
from router.skills import build_router  # noqa: F401
