"""
Tool Registry — the schema the planner (LLM router) plans over.

A ToolSpec describes one capability the planner may call: its LLM-facing
schema (name, description, JSON Schema parameters) plus how it actually
runs — either a direct async `handler`, or a `target_agent`/`action` pair
routed over the event bus. ToolRegistry stores ToolSpecs and renders them
into the OpenAI-style tool schema the router hands to a model.

Registration happens two ways:
    - `@tool(...)` decorates a plain async function as a directly-invoked
      tool (used for local/self-contained tools).
    - Explicit `registry.register(ToolSpec(...))` calls, used for
      bus-routed tools that have no Python function to decorate (see
      tools/builtin.py).

The Guardian (guardian/gate.py) decides whether a given ToolSpec call may
run without confirmation; this module only describes tools, it does not
gate or execute them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

VALID_TIERS = ("safe", "confirm", "dangerous")

ToolHandler = Callable[..., Awaitable[Any]]


@dataclass
class ToolSpec:
    """Describes one tool the planner can call.

    Exactly one of `handler` or (`target_agent` + `action`) must be set:
    a tool either runs a local async callable directly, or is routed to
    an agent over the event bus.
    """

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    tier: str = "safe"
    handler: Optional[ToolHandler] = None
    target_agent: str = ""
    action: str = ""
    category: str = "general"
    enabled: bool = True
    slow: bool = False
    #: Optional async builder for the Guardian's confirmation summary. Given the
    #: call's `arguments`, returns the exact one-line summary the user sees (and
    #: may enrich/mutate `arguments` in place — e.g. git_commit generating a
    #: message from the diff so it appears in the confirmation before running).
    confirm_summary: Optional[Callable[[Dict[str, Any]], Awaitable[str]]] = None

    def __post_init__(self) -> None:
        if self.tier not in VALID_TIERS:
            raise ValueError(
                f"Invalid tier {self.tier!r} for tool {self.name!r}; must be one of {VALID_TIERS}"
            )

        has_handler = self.handler is not None
        has_bus_route = bool(self.target_agent) and bool(self.action)
        if has_handler == has_bus_route:
            raise ValueError(
                f"Tool {self.name!r} must set exactly one of `handler` or "
                f"(`target_agent` + `action`), not "
                f"{'both' if has_handler else 'neither'}"
            )

    @property
    def is_bus_routed(self) -> bool:
        """Whether this tool is invoked by routing a request to an agent over the bus."""
        return bool(self.target_agent and self.action)

    def to_openai_function(self) -> Dict[str, Any]:
        """OpenAI/Groq-style `{"type": "function", "function": {...}}` schema entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Central registry of tools the planner can call."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        """Register a ToolSpec. Raises ValueError if the name is already taken."""
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered")

        self._tools[spec.name] = spec
        route = f"handler:{spec.handler.__name__}" if spec.handler else f"{spec.target_agent}.{spec.action}"
        logger.debug(f"[ToolRegistry] registered '{spec.name}' tier={spec.tier} route={route}")
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        """Look up a tool by name, or None if it isn't registered."""
        return self._tools.get(name)

    def list_all(self, enabled_only: bool = True) -> List[ToolSpec]:
        """List every registered tool, optionally excluding disabled ones."""
        tools = list(self._tools.values())
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        return tools

    def list_by_tier(self, tier: str, enabled_only: bool = True) -> List[ToolSpec]:
        """List registered tools matching a permission tier."""
        if tier not in VALID_TIERS:
            raise ValueError(f"Invalid tier {tier!r}; must be one of {VALID_TIERS}")
        return [t for t in self.list_all(enabled_only=enabled_only) if t.tier == tier]

    def to_llm_schema(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """The OpenAI-style `tools` array for the router (`llm.router.ModelRouter.complete`)."""
        return [t.to_openai_function() for t in self.list_all(enabled_only=enabled_only)]


_default_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """The process-wide default tool registry, populated by tools.builtin on import."""
    return _default_registry


def tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    tier: str = "safe",
    category: str = "general",
    enabled: bool = True,
    slow: bool = False,
    registry: Optional[ToolRegistry] = None,
):
    """Decorator: register an async function as a directly-invoked tool.

    Example:
        @tool(name="echo", description="Echo the input back.",
              parameters={"type": "object", "properties": {"text": {"type": "string"}}})
        async def echo(arguments, context):
            return arguments["text"]
    """

    def decorator(func: ToolHandler) -> ToolHandler:
        target_registry = registry if registry is not None else get_registry()
        target_registry.register(
            ToolSpec(
                name=name,
                description=description,
                parameters=parameters or {"type": "object", "properties": {}},
                tier=tier,
                handler=func,
                category=category,
                enabled=enabled,
                slow=slow,
            )
        )
        return func

    return decorator
