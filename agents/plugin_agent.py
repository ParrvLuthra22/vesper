"""PluginAgent - dynamically loads and executes intent plugins."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agents.base_agent import AgentCapability, BaseAgent
from schemas.events import IntentRecognizedEvent, VoiceOutputEvent


PluginHandler = Callable[[IntentRecognizedEvent], Awaitable[Any]]


class PluginAgent(BaseAgent):
    """Loads plugins from `plugins/` and routes matching intents to them."""

    def __init__(
        self,
        name: Optional[str] = None,
        event_bus=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name or "PluginAgent", event_bus=event_bus, config=config)

        plugins_cfg = (config or {}).get("plugins", {})
        configured_dir = plugins_cfg.get("directory") or (config or {}).get("plugins_dir")

        default_plugins_dir = Path(__file__).resolve().parents[1] / "plugins"
        self._plugins_dir = Path(configured_dir) if configured_dir else default_plugins_dir

        self._loaded_plugins: Dict[str, ModuleType] = {}
        self._trigger_registry: Dict[str, str] = {}  # trigger -> plugin_name

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="plugin_routing",
                description="Routes recognized intents to dynamically loaded plugins",
                input_events=["IntentRecognizedEvent"],
                output_events=["VoiceOutputEvent"],
            )
        ]

    @property
    def loaded_plugins(self) -> List[str]:
        return sorted(self._loaded_plugins.keys())

    async def _setup(self) -> None:
        self._load_plugins()
        # D7: IntentRecognizedEvent is dead — the Planner no longer emits it, and
        # plugins are not (yet) exposed as registered tools, so this agent is no
        # longer wired to the bus. _handle_intent is retained as the dispatch entry
        # point for whenever plugins are surfaced as tools; it is not subscribed.
        self._logger.info(
            f"Plugin agent ready with {len(self._loaded_plugins)} plugin(s) from {self._plugins_dir}"
        )

    async def _teardown(self) -> None:
        self._loaded_plugins.clear()
        self._trigger_registry.clear()

    def _load_plugins(self) -> None:
        self._loaded_plugins.clear()
        self._trigger_registry.clear()

        if not self._plugins_dir.exists():
            self._logger.warning(f"Plugins directory not found: {self._plugins_dir}")
            return

        for plugin_file in sorted(self._plugins_dir.glob("*.py")):
            if plugin_file.name.startswith("_"):
                continue

            module = self._import_plugin_module(plugin_file)
            if module is None:
                continue

            plugin_name = getattr(module, "PLUGIN_NAME", None)
            triggers = getattr(module, "TRIGGERS", None)
            handler = getattr(module, "handle", None)

            if not isinstance(plugin_name, str) or not plugin_name.strip():
                self._logger.warning(f"Skipping plugin {plugin_file.name}: invalid PLUGIN_NAME")
                continue

            if not isinstance(triggers, list) or not all(isinstance(t, str) for t in triggers):
                self._logger.warning(f"Skipping plugin {plugin_name}: invalid TRIGGERS")
                continue

            if not callable(handler) or not inspect.iscoroutinefunction(handler):
                self._logger.warning(f"Skipping plugin {plugin_name}: async handle(event) missing")
                continue

            self._loaded_plugins[plugin_name] = module
            for trigger in triggers:
                normalized = trigger.strip().lower()
                if normalized:
                    self._trigger_registry[normalized] = plugin_name

            self._logger.info(f"Loaded plugin: {plugin_name} ({len(triggers)} trigger(s))")

    def _import_plugin_module(self, plugin_file: Path) -> Optional[ModuleType]:
        module_name = f"vesper_plugin_{plugin_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(plugin_file))
        if spec is None or spec.loader is None:
            self._logger.warning(f"Failed to create import spec for plugin: {plugin_file}")
            return None

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            self._logger.error(f"Failed to load plugin {plugin_file.name}: {exc}")
            return None

        return module

    async def _handle_intent(self, event: IntentRecognizedEvent) -> None:
        trigger = (event.intent or "").strip().lower()
        plugin_name = self._trigger_registry.get(trigger)

        if plugin_name is None:
            raw = (event.raw_text or "").strip().lower()
            plugin_name = self._trigger_registry.get(raw)

        if plugin_name is None:
            return

        module = self._loaded_plugins.get(plugin_name)
        if module is None:
            return

        handler: PluginHandler = getattr(module, "handle")

        try:
            result = await handler(event)
        except Exception as exc:
            self._logger.error(f"Plugin '{plugin_name}' failed: {exc}", exc_info=True)
            await self._emit(
                VoiceOutputEvent(
                    text=f"Plugin {plugin_name} failed: {exc}",
                    source=self.name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )
            return

        if result is None:
            return

        text = str(result)
        await self._emit(
            VoiceOutputEvent(
                text=text,
                source=self.name,
                correlation_id=event.correlation_id or event.event_id,
            )
        )
