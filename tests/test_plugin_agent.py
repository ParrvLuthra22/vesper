import asyncio
import pytest

from agents.plugin_agent import PluginAgent
from schemas.events import IntentRecognizedEvent, VoiceOutputEvent


@pytest.mark.asyncio
async def test_plugin_agent_loads_and_handles(tmp_path, monkeypatch):
    # Create a temporary plugins directory and copy the sample plugin into it
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()

    # locate the sample plugin in the repo
    repo_plugin = __import__("plugins.calculator_plugin", fromlist=["*"])
    plugin_source = repo_plugin.__file__

    # copy file content
    dest = plugins_dir / "calculator_plugin.py"
    dest.write_bytes(open(plugin_source, "rb").read())

    # Build a minimal brain/config and inject the plugins path
    config = {
        "plugins": {"path": str(plugins_dir)}
    }

    # Create agent
    agent = PluginAgent(config={"plugins": {"path": str(plugins_dir)}})

    # Start agent (runs _setup and transitions to RUNNING)
    await agent.start()

    # Create a fake event that should trigger the calculator plugin
    event = IntentRecognizedEvent(
        intent="calculate",
        raw_text="what is 2 plus 2",
        slots={},
        correlation_id="test-1",
    )

    # Prepare a future to capture emitted VoiceOutputEvent
    received = asyncio.Future()

    async def capture(event_obj):
        if isinstance(event_obj, VoiceOutputEvent):
            if not received.done():
                received.set_result(event_obj)

    # Subscribe capture to the event bus used by the agent
    agent.event_bus.subscribe(VoiceOutputEvent, capture)

    # D7: PluginAgent is no longer subscribed to IntentRecognizedEvent (the
    # Planner doesn't emit it). Invoke the dispatch handler directly — this still
    # verifies plugin loading + handling + VoiceOutputEvent emission.
    await agent._handle_intent(event)

    # Wait for the plugin to handle and emit voice output
    voice_event = await asyncio.wait_for(received, timeout=2.0)

    assert isinstance(voice_event, VoiceOutputEvent)
    assert "result" in voice_event.text.lower() or "4" in voice_event.text

    # Teardown
    await agent.stop()
