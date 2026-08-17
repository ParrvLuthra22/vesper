"""
Wraps existing SystemAgent/WebSearchAgent capabilities as bus-routed tools.

Each ToolSpec here is descriptive metadata only, registered via explicit
register() calls (these tools have no local Python handler to decorate with
@tool — they are invoked by routing target_agent/action over the event bus).

Tier assignment follows the P02 rules:
    - reads/queries, plus open/focus/volume/brightness/notify actions = safe
    - close_app, lock_screen = confirm (they change state in a way that's
      hard to casually undo)

`search_web` targets WebSearchAgent (not SystemAgent) — SystemAgent's
search_web handler only opens a browser search URL; the real Tavily+OpenRouter
search/summarize pipeline lives on WebSearchAgent.
"""

from __future__ import annotations

from tools.registry import ToolSpec, get_registry

registry = get_registry()


def _register_builtin_tools() -> None:
    tools = [
        ToolSpec(
            name="open_app",
            description=(
                "Open (launch) a macOS application by name. Use when the user wants to "
                "start or switch to an application that may not be running yet."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name of the application, e.g. 'Safari', 'Notes', 'Terminal'.",
                    },
                },
                "required": ["app_name"],
            },
            tier="safe",
            target_agent="SystemAgent",
            action="open_app",
            category="system",
        ),
        ToolSpec(
            name="close_app",
            description=(
                "Quit a running macOS application by name. Use only when the user "
                "explicitly asks to close or quit an app — this can discard unsaved "
                "state in that app."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Name of the application to close."},
                    "force": {
                        "type": "boolean",
                        "description": "Force-quit instead of a graceful quit. Defaults to false.",
                    },
                },
                "required": ["app_name"],
            },
            tier="confirm",
            target_agent="SystemAgent",
            action="close_app",
            category="system",
        ),
        ToolSpec(
            name="focus_app",
            description="Bring a running application to the foreground. Use for 'switch to X' style requests.",
            parameters={
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Name of the application to focus."},
                },
                "required": ["app_name"],
            },
            tier="safe",
            target_agent="SystemAgent",
            action="focus_app",
            category="system",
        ),
        ToolSpec(
            name="list_apps",
            description="List currently running applications. Use to answer 'what's open' style questions.",
            parameters={"type": "object", "properties": {}},
            tier="safe",
            target_agent="SystemAgent",
            action="list_apps",
            category="system",
        ),
        ToolSpec(
            name="set_volume",
            description=(
                "Set the system output volume to an exact level. Use when the user "
                "gives a specific percentage rather than 'up'/'down'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "Volume level from 0 to 100.",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "required": ["level"],
            },
            tier="safe",
            target_agent="SystemAgent",
            action="control_volume",
            category="system",
        ),
        ToolSpec(
            name="get_volume",
            description="Read the current system volume level and mute state.",
            parameters={"type": "object", "properties": {}},
            tier="safe",
            target_agent="SystemAgent",
            action="get_volume",
            category="system",
        ),
        ToolSpec(
            name="mute",
            description="Mute system audio output.",
            parameters={"type": "object", "properties": {}},
            tier="safe",
            target_agent="SystemAgent",
            action="mute",
            category="system",
        ),
        ToolSpec(
            name="set_brightness",
            description="Set the display brightness to an exact level.",
            parameters={
                "type": "object",
                "properties": {
                    "level": {
                        "type": "number",
                        "description": "Brightness from 0 to 100 (percent).",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "required": ["level"],
            },
            tier="safe",
            target_agent="SystemAgent",
            action="set_brightness",
            category="system",
        ),
        ToolSpec(
            name="get_time",
            description="Get the current local time.",
            parameters={"type": "object", "properties": {}},
            tier="safe",
            target_agent="SystemAgent",
            action="get_time",
            category="system",
        ),
        ToolSpec(
            name="get_date",
            description="Get the current local date.",
            parameters={"type": "object", "properties": {}},
            tier="safe",
            target_agent="SystemAgent",
            action="get_date",
            category="system",
        ),
        ToolSpec(
            name="get_battery",
            description="Get battery percentage and charging status.",
            parameters={"type": "object", "properties": {}},
            tier="safe",
            target_agent="SystemAgent",
            action="get_battery",
            category="system",
        ),
        ToolSpec(
            name="system_info",
            description="Get a summary of CPU, memory, disk, and battery usage.",
            parameters={"type": "object", "properties": {}},
            tier="safe",
            target_agent="SystemAgent",
            action="system_info",
            category="system",
        ),
        ToolSpec(
            name="take_screenshot",
            description="Capture a screenshot of the current screen.",
            parameters={"type": "object", "properties": {}},
            tier="safe",
            target_agent="SystemAgent",
            action="screenshot",
            category="system",
        ),
        ToolSpec(
            name="show_notification",
            description="Show a macOS notification banner with a title and message.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Notification title."},
                    "message": {"type": "string", "description": "Notification body text."},
                },
                "required": ["title", "message"],
            },
            tier="safe",
            target_agent="SystemAgent",
            action="notify",
            category="system",
        ),
        ToolSpec(
            name="open_url",
            description="Open a URL in the default web browser.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to open, e.g. 'https://github.com'."},
                },
                "required": ["url"],
            },
            tier="safe",
            target_agent="SystemAgent",
            action="open_url",
            category="web",
        ),
        ToolSpec(
            name="search_web",
            description=(
                "Search the web for up-to-date information and return a summarized "
                "answer with sources. Use for factual questions, current events, or "
                "anything requiring information beyond training data."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
            tier="safe",
            target_agent="WebSearchAgent",
            action="search_web",
            category="web",
        ),
        ToolSpec(
            name="lock_screen",
            description="Lock the screen immediately. Use only when the user explicitly asks to lock/secure the Mac.",
            parameters={"type": "object", "properties": {}},
            tier="confirm",
            target_agent="SystemAgent",
            action="lock_screen",
            category="system",
        ),
    ]

    for spec in tools:
        registry.register(spec)


_register_builtin_tools()
