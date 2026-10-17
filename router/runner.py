"""Runs a matched skill. Everything a skill does to the machine goes through the Guardian first, at
max(skill tier, tool tier) — the router never has more authority than the planner would have had."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from guardian.gate import Guardian, VerdictType
from router.core import Handoff, Match, SkillResult, SkillRouter, SkillRuntime, SkillUnavailable, max_tier
from tools.registry import ToolRegistry, ToolSpec
from utils.logger import get_logger

logger = get_logger(__name__)


class SkillDenied(RuntimeError):
    """The Guardian (or the user) said no."""


@dataclass
class SkillOutcome:
    text: str
    skill: str
    tainted: bool = False
    #: When set, `text` is empty: the planner answers, restricted to these tools (router/core.py Handoff).
    handoff: Optional[Handoff] = None


class SkillRunner:
    def __init__(self, router: SkillRouter, registry: ToolRegistry, guardian: Guardian,
                 run_tool: Callable[[ToolSpec, Dict[str, Any]], Awaitable[str]],
                 briefing: Any = None, address_as: str = "Sir", clock: Callable[[], datetime] = datetime.now):
        self._router, self._registry, self._guardian = router, registry, guardian
        self._run_tool, self._briefing, self._address, self._clock = run_tool, briefing, address_as, clock

    def _caller(self, match: Match) -> Callable[[str, Dict[str, Any]], Awaitable[str]]:
        async def call(tool: str, args: Dict[str, Any]) -> str:
            spec = self._registry.get(tool)
            if spec is None or not spec.enabled:
                raise SkillUnavailable(f"tool '{tool}' is not available")
            effective = dataclasses.replace(spec, tier=max_tier(spec.tier, match.skill.tier))
            verdict = await self._guardian.check(effective, args, context={})
            if verdict.outcome == VerdictType.NEEDS_CONFIRMATION:
                verdict = await self._guardian.await_resolution(verdict.request_id)
            if verdict.outcome != VerdictType.ALLOW:
                raise SkillDenied(verdict.reason)
            return await self._run_tool(spec, args)
        return call

    async def try_handle(self, text: str) -> Optional[SkillOutcome]:
        """The skill's answer, or None to let the planner take the utterance (no match, ambiguous, or the skill could
        not do its job without having changed anything)."""
        match = self._router.match(text)
        if match is None:
            return None
        rt = SkillRuntime(call=self._caller(match), briefing=self._briefing, clock=self._clock, address_as=self._address)
        try:
            result: SkillResult = await match.skill.handler(match, rt)
        except SkillUnavailable as exc:
            logger.info(f"[Router] {match.skill.name}: unavailable ({exc}); falling through to the planner")
            return None
        except SkillDenied as exc:
            return SkillOutcome(f"Understood, {self._address} — I won't do that. ({exc})", match.skill.name)
        except Exception as exc:                                  # noqa: BLE001
            logger.warning(f"[Router] {match.skill.name} failed: {type(exc).__name__}: {exc}")
            if match.skill.mutates:
                return SkillOutcome(f"That didn't work, {self._address}: {str(exc)[:120] or type(exc).__name__}.",
                                    match.skill.name)
            return None
        return SkillOutcome(result.text, match.skill.name, tainted=result.tainted or match.skill.untrusted,
                            handoff=result.handoff)
