"""MacOSControlAgent - macOS automation and app control via AppleScript and system tools."""

from __future__ import annotations

import asyncio
import base64
import getpass
import json
import os
import re
import shutil
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import parse, request

from agents.base_agent import AgentCapability, BaseAgent
from schemas.events import IntentRecognizedEvent, MacOSCommandEvent, VoiceOutputEvent
from utils.applescript import run_applescript, run_applescript_file


class MacOSControlAgent(BaseAgent):
    """Agent handling direct macOS controls and app automations."""

    def __init__(
        self,
        name: Optional[str] = None,
        event_bus=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name or "MacOSControlAgent", event_bus=event_bus, config=config)
        self._username = getpass.getuser()

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="macos_control",
                description="iMessage, Safari, Finder, Calendar, Spotify, and system settings control",
                input_events=["IntentRecognizedEvent", "MacOSCommandEvent"],
                output_events=["VoiceOutputEvent"],
            )
        ]

    async def _setup(self) -> None:
        # D7: IntentRecognizedEvent is dead — the Planner routes tool calls, and
        # macOS control is exposed through SystemAgent's registered tools
        # (open_app/close_app/set_volume/set_brightness/...). Only the direct
        # MacOSCommandEvent path remains.
        self._subscribe(MacOSCommandEvent, self._handle_macos_command)

    async def _teardown(self) -> None:
        return

    async def _handle_intent(self, event: IntentRecognizedEvent) -> None:
        text = self._event_text(event)
        if not text:
            return

        text_lower = text.lower()

        if self._is_imessage_intent(text_lower):
            await self._handle_imessage(text, event)
            return

        if self._is_safari_intent(text_lower):
            await self._handle_safari(text, event)
            return

        if self._is_finder_intent(text_lower):
            await self._handle_finder(text, event)
            return

        if self._is_calendar_intent(text_lower):
            await self._handle_calendar(text, event)
            return

        if self._is_spotify_intent(text_lower):
            await self._handle_spotify(text, event)
            return

        if self._is_system_settings_intent(text_lower):
            await self._handle_system_settings(text, event)

    async def _handle_macos_command(self, event: MacOSCommandEvent) -> None:
        command = (event.command_type or "").strip().lower()
        payload = event.payload or {}

        if command == "applescript":
            script = str(payload.get("script", ""))
            if not script:
                await self._emit_voice("No AppleScript payload provided.", event)
                return
            await self._safe_osascript(
                script=script,
                event=event,
                success_message="AppleScript command completed.",
                failure_prefix="AppleScript command failed",
            )
            return

        if command == "applescript_file":
            path = str(payload.get("path", ""))
            try:
                _ = run_applescript_file(path)
                await self._emit_voice("AppleScript file executed successfully.", event)
            except Exception as exc:
                await self._emit_voice(f"AppleScript file execution failed: {exc}", event)
            return

        await self._emit_voice(f"Unsupported MacOS command type: {command}", event)

    # ---------------------------------------------------------------------
    # iMessage
    # ---------------------------------------------------------------------

    async def _handle_imessage(self, text: str, event: IntentRecognizedEvent) -> None:
        recipient = self.extract_name(text)
        message = self.extract_message(text)

        if not recipient:
            await self._emit_voice("I couldn't determine who to message.", event)
            return
        if not message:
            await self._emit_voice("I couldn't determine the message content.", event)
            return

        script = (
            "tell application \"Messages\"\n"
            f"  set targetBuddy to buddy \"{self._escape_applescript(recipient)}\" of service \"SMS\"\n"
            f"  send \"{self._escape_applescript(message)}\" to targetBuddy\n"
            "end tell"
        )

        await self._safe_osascript(
            script=script,
            event=event,
            success_message=f"Message sent to {recipient}",
            failure_prefix=f"I couldn't send the message to {recipient}",
        )

    # ---------------------------------------------------------------------
    # Safari
    # ---------------------------------------------------------------------

    async def _handle_safari(self, text: str, event: IntentRecognizedEvent) -> None:
        query = self._extract_safari_query(text)
        if not query:
            await self._emit_voice("I couldn't determine what to search in Safari.", event)
            return

        url = f"https://www.google.com/search?q={parse.quote_plus(query)}"
        script = (
            "tell application \"Safari\"\n"
            "  activate\n"
            f"  open location \"{url}\"\n"
            "end tell"
        )

        await self._safe_osascript(
            script=script,
            event=event,
            success_message=f"Opening Safari search for {query}",
            failure_prefix="I couldn't open Safari",
        )

    # ---------------------------------------------------------------------
    # Finder
    # ---------------------------------------------------------------------

    async def _handle_finder(self, text: str, event: IntentRecognizedEvent) -> None:
        lower = text.lower()

        if "open downloads" in lower:
            await self._open_finder_path(Path(f"/Users/{self._username}/Downloads"), event)
            return

        if lower.startswith("create folder"):
            await self._create_folder(text, event)
            return

        if lower.startswith("move file"):
            await self._move_file(text, event)
            return

        if lower.startswith("delete file"):
            await self._delete_file(text, event)
            return

        if lower.startswith("find file"):
            await self._find_file(text, event)
            return

        folder_path = self._extract_folder_path(text)
        if folder_path:
            await self._open_finder_path(folder_path, event)
            return

    async def _open_finder_path(self, path: Path, event: IntentRecognizedEvent) -> None:
        script = (
            "tell application \"Finder\"\n"
            f"  open POSIX file \"{self._escape_applescript(str(path))}\"\n"
            "  activate\n"
            "end tell"
        )
        await self._safe_osascript(
            script=script,
            event=event,
            success_message=f"Opened {path}",
            failure_prefix=f"I couldn't open {path}",
        )

    async def _create_folder(self, text: str, event: IntentRecognizedEvent) -> None:
        match = re.search(r"create folder\s+(.+?)\s+in\s+(.+)$", text, flags=re.IGNORECASE)
        if not match:
            await self._emit_voice("I couldn't parse the folder creation request.", event)
            return

        folder_name = match.group(1).strip().strip("\"'")
        parent_rel = match.group(2).strip().strip("\"'")
        parent = self._resolve_user_path(parent_rel)
        target = parent / folder_name

        try:
            target.mkdir(parents=True, exist_ok=True)
            await self._open_finder_path(target, event)
            await self._emit_voice(f"Created folder {folder_name} in {parent_rel}", event)
        except Exception as exc:
            await self._emit_voice(f"I couldn't create that folder: {exc}", event)

    async def _move_file(self, text: str, event: IntentRecognizedEvent) -> None:
        match = re.search(r"move file\s+(.+?)\s+to\s+(.+)$", text, flags=re.IGNORECASE)
        if not match:
            await self._emit_voice("I couldn't parse the move file request.", event)
            return

        source_name = match.group(1).strip().strip("\"'")
        destination_rel = match.group(2).strip().strip("\"'")

        source_path = self._find_file_path(source_name)
        if not source_path:
            await self._emit_voice(f"I couldn't find file {source_name}", event)
            return

        destination_dir = self._resolve_user_path(destination_rel)
        destination_dir.mkdir(parents=True, exist_ok=True)
        target = destination_dir / source_path.name

        try:
            shutil.move(str(source_path), str(target))
            await self._emit_voice(f"Moved {source_path.name} to {destination_rel}", event)
        except Exception as exc:
            await self._emit_voice(f"I couldn't move the file: {exc}", event)

    async def _delete_file(self, text: str, event: IntentRecognizedEvent) -> None:
        match = re.search(r"delete file\s+(.+)$", text, flags=re.IGNORECASE)
        if not match:
            await self._emit_voice("I couldn't parse which file to delete.", event)
            return

        source_name = match.group(1).strip().strip("\"'")
        source_path = self._find_file_path(source_name)
        if not source_path:
            await self._emit_voice(f"I couldn't find file {source_name}", event)
            return

        await self._emit_voice(f"Confirmation: deleting file {source_path.name}.", event)

        try:
            if source_path.is_dir():
                shutil.rmtree(source_path)
            else:
                source_path.unlink()
            await self._emit_voice(f"Deleted {source_path.name}", event)
        except Exception as exc:
            await self._emit_voice(f"I couldn't delete the file: {exc}", event)

    async def _find_file(self, text: str, event: IntentRecognizedEvent) -> None:
        match = re.search(r"find file\s+(.+)$", text, flags=re.IGNORECASE)
        if not match:
            await self._emit_voice("I couldn't parse which file to find.", event)
            return

        target_name = match.group(1).strip().strip("\"'")
        file_path = self._find_file_path(target_name)
        if not file_path:
            await self._emit_voice(f"I couldn't find {target_name}", event)
            return

        script = (
            "tell application \"Finder\"\n"
            f"  reveal POSIX file \"{self._escape_applescript(str(file_path))}\"\n"
            "  activate\n"
            "end tell"
        )

        await self._safe_osascript(
            script=script,
            event=event,
            success_message=f"Found {target_name}",
            failure_prefix=f"I couldn't reveal {target_name}",
        )

    # ---------------------------------------------------------------------
    # Calendar
    # ---------------------------------------------------------------------

    async def _handle_calendar(self, text: str, event: IntentRecognizedEvent) -> None:
        lower = text.lower()
        if "what's on my calendar" in lower or "whats on my calendar" in lower:
            await self._read_calendar_today(event)
            return

        await self._create_calendar_event(text, event)

    async def _read_calendar_today(self, event: IntentRecognizedEvent) -> None:
        loop = asyncio.get_running_loop()
        try:
            events = await loop.run_in_executor(None, self._fetch_today_events_eventkit)
        except Exception as exc:
            await self._emit_voice(f"I couldn't read your calendar: {exc}", event)
            return

        if not events:
            await self._emit_voice("You have no events on your calendar today.", event)
            return

        lines = [f"{title} at {start_time}" for title, start_time in events[:5]]
        await self._emit_voice("Today's calendar: " + "; ".join(lines), event)

    async def _create_calendar_event(self, text: str, event: IntentRecognizedEvent) -> None:
        title, start_dt = self._parse_calendar_request(text)
        end_dt = start_dt + timedelta(hours=1)

        start_str = start_dt.strftime("%m/%d/%Y %I:%M:%S %p")
        end_str = end_dt.strftime("%m/%d/%Y %I:%M:%S %p")

        script = (
            "tell application \"Calendar\"\n"
            "  tell first calendar whose writable is true\n"
            "    make new event at end of events with properties "
            f"{{summary:\"{self._escape_applescript(title)}\", start date:date \"{start_str}\", end date:date \"{end_str}\"}}\n"
            "  end tell\n"
            "end tell"
        )

        await self._safe_osascript(
            script=script,
            event=event,
            success_message=f"Scheduled {title}",
            failure_prefix="I couldn't create that calendar event",
        )

    # ---------------------------------------------------------------------
    # Spotify
    # ---------------------------------------------------------------------

    async def _handle_spotify(self, text: str, event: IntentRecognizedEvent) -> None:
        lower = text.lower()

        if "pause music" in lower:
            await self._safe_osascript(
                script='tell application "Spotify"\n  pause\nend tell',
                event=event,
                success_message="Paused Spotify",
                failure_prefix="I couldn't pause Spotify",
            )
            return

        if "next song" in lower:
            await self._safe_osascript(
                script='tell application "Spotify"\n  next track\nend tell',
                event=event,
                success_message="Playing next song",
                failure_prefix="I couldn't skip to the next song",
            )
            return

        if "previous song" in lower:
            await self._safe_osascript(
                script='tell application "Spotify"\n  previous track\nend tell',
                event=event,
                success_message="Playing previous song",
                failure_prefix="I couldn't go to the previous song",
            )
            return

        if "what's playing" in lower or "whats playing" in lower:
            output = await self._safe_osascript(
                script=(
                    'tell application "Spotify"\n'
                    '  if player state is stopped then\n'
                    '    return "Nothing is playing"\n'
                    '  end if\n'
                    '  return name of current track & " by " & artist of current track\n'
                    'end tell'
                ),
                event=event,
                success_message=None,
                failure_prefix="I couldn't read Spotify playback state",
            )
            if output is not None:
                await self._emit_voice(f"Now playing: {output}", event)
            return

        playlist_match = re.search(r"play playlist\s+(.+)$", text, flags=re.IGNORECASE)
        if playlist_match:
            query = playlist_match.group(1).strip()
            await self._play_spotify_uri(query, "playlist", event)
            return

        track_match = re.search(r"play\s+(.+?)\s+on spotify$", text, flags=re.IGNORECASE)
        if track_match:
            query = track_match.group(1).strip()
            await self._play_spotify_uri(query, "track", event)
            return

    async def _play_spotify_uri(self, query: str, item_type: str, event: IntentRecognizedEvent) -> None:
        loop = asyncio.get_running_loop()
        try:
            uri = await loop.run_in_executor(None, self._spotify_search_uri, query, item_type)
        except Exception as exc:
            await self._emit_voice(f"Spotify search failed: {exc}", event)
            return

        if not uri:
            await self._emit_voice(f"I couldn't find {query} on Spotify.", event)
            return

        script = (
            "tell application \"Spotify\"\n"
            "  activate\n"
            f"  play track \"{self._escape_applescript(uri)}\"\n"
            "end tell"
        )

        await self._safe_osascript(
            script=script,
            event=event,
            success_message=f"Playing {query} on Spotify",
            failure_prefix="I couldn't play that on Spotify",
        )

    # ---------------------------------------------------------------------
    # System settings
    # ---------------------------------------------------------------------

    async def _handle_system_settings(self, text: str, event: IntentRecognizedEvent) -> None:
        lower = text.lower()

        volume_match = re.search(r"set volume to\s+(\d{1,3})", lower)
        if volume_match:
            level = max(0, min(100, int(volume_match.group(1))))
            await self._safe_osascript(
                script=f"set volume output volume {level}",
                event=event,
                success_message=f"Set volume to {level}",
                failure_prefix="I couldn't set the volume",
            )
            return

        brightness_match = re.search(r"set brightness to\s+([0-9]{1,3})", lower)
        if brightness_match:
            await self._set_brightness(int(brightness_match.group(1)), event)
            return

        if "turn off wifi" in lower:
            await self._set_wifi(False, event)
            return

        if "turn on wifi" in lower:
            await self._set_wifi(True, event)
            return

        if "enable dark mode" in lower:
            await self._safe_osascript(
                script=(
                    'tell application "System Events"\n'
                    '  tell appearance preferences\n'
                    '    set dark mode to true\n'
                    '  end tell\n'
                    'end tell'
                ),
                event=event,
                success_message="Dark mode enabled",
                failure_prefix="I couldn't enable dark mode",
            )
            return

        if "what's my battery" in lower or "whats my battery" in lower:
            await self._read_battery(event)
            return

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    async def _emit_voice(self, text: str, event: Any) -> None:
        await self._emit(
            VoiceOutputEvent(
                text=text,
                source=self._name,
                correlation_id=getattr(event, "correlation_id", None) or getattr(event, "event_id", None),
            )
        )

    async def _safe_osascript(
        self,
        script: str,
        event: Any,
        success_message: Optional[str],
        failure_prefix: str,
    ) -> Optional[str]:
        try:
            output = run_applescript(script)
        except Exception as exc:
            await self._emit_voice(f"{failure_prefix}: {exc}", event)
            return None

        if success_message:
            await self._emit_voice(success_message, event)

        return output

    @staticmethod
    def _escape_applescript(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _event_text(self, event: IntentRecognizedEvent) -> str:
        text_parts = [
            getattr(event, "raw_text", ""),
            getattr(event, "intent", ""),
            getattr(event, "text", ""),
        ]
        return " ".join(part.strip() for part in text_parts if part and part.strip())

    def extract_name(self, text: str) -> str:
        patterns = [
            r"message\s+(.+?)\s+saying\s+",
            r"send message to\s+(.+?)(?:\s+saying\s+|$)",
            r"text\s+(.+?)(?:\s+saying\s+|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().strip("\"'")
        return ""

    def extract_message(self, text: str) -> str:
        match = re.search(r"saying\s+(.+)$", text, flags=re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).strip().strip("\"'")

    def _extract_safari_query(self, text: str) -> str:
        patterns = [
            r"open\s+(.+?)\s+in safari",
            r"search for\s+(.+?)\s+in safari$",
            r"go to\s+(.+?)\s+website",
            r"browse\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().strip("\"'")
        return ""

    def _extract_folder_path(self, text: str) -> Optional[Path]:
        open_match = re.search(r"open folder\s+(.+)$", text, flags=re.IGNORECASE)
        if open_match:
            return self._resolve_user_path(open_match.group(1).strip().strip("\"'"))

        show_match = re.search(r"show me\s+(.+?)\s+folder$", text, flags=re.IGNORECASE)
        if show_match:
            return self._resolve_user_path(show_match.group(1).strip().strip("\"'"))

        return None

    def _resolve_user_path(self, relative_path: str) -> Path:
        cleaned = relative_path.strip().lstrip("/")
        return Path(f"/Users/{self._username}") / cleaned

    def _find_file_path(self, name_or_path: str) -> Optional[Path]:
        candidate = Path(name_or_path).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return candidate

        home = Path(f"/Users/{self._username}")
        direct = home / name_or_path
        if direct.exists():
            return direct

        try:
            result = subprocess.run(
                ["mdfind", f'kMDItemFSName == "{name_or_path}"c'],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.splitlines():
                path = Path(line.strip())
                if path.exists():
                    return path
        except Exception:
            return None

        return None

    def _is_imessage_intent(self, text_lower: str) -> bool:
        return (
            "send message to" in text_lower
            or text_lower.startswith("text ")
            or re.search(r"message\s+.+\s+saying\s+.+", text_lower) is not None
        )

    def _is_safari_intent(self, text_lower: str) -> bool:
        return (
            " in safari" in text_lower
            or " website" in text_lower
            or text_lower.startswith("browse ")
        )

    def _is_finder_intent(self, text_lower: str) -> bool:
        return (
            text_lower.startswith("open folder")
            or text_lower.startswith("find file")
            or text_lower.startswith("show me")
            or "open downloads" in text_lower
            or text_lower.startswith("create folder")
            or text_lower.startswith("move file")
            or text_lower.startswith("delete file")
        )

    def _is_calendar_intent(self, text_lower: str) -> bool:
        return (
            text_lower.startswith("add event")
            or "what's on my calendar" in text_lower
            or "whats on my calendar" in text_lower
            or text_lower.startswith("schedule ")
        )

    def _is_spotify_intent(self, text_lower: str) -> bool:
        return (
            "on spotify" in text_lower
            or "pause music" in text_lower
            or "next song" in text_lower
            or "play playlist" in text_lower
            or "what's playing" in text_lower
            or "whats playing" in text_lower
            or "previous song" in text_lower
        )

    def _is_system_settings_intent(self, text_lower: str) -> bool:
        return (
            text_lower.startswith("set volume to")
            or text_lower.startswith("set brightness to")
            or "turn off wifi" in text_lower
            or "turn on wifi" in text_lower
            or "enable dark mode" in text_lower
            or "what's my battery" in text_lower
            or "whats my battery" in text_lower
        )

    def _parse_calendar_request(self, text: str) -> Tuple[str, datetime]:
        add_match = re.search(r"add event\s+(.+?)\s+on\s+(.+?)\s+at\s+(.+)$", text, flags=re.IGNORECASE)
        if add_match:
            title = add_match.group(1).strip().strip("\"'")
            date_part = add_match.group(2).strip()
            time_part = add_match.group(3).strip()
            return title, self._parse_date_time(date_part, time_part)

        schedule_match = re.search(r"schedule\s+(.+)$", text, flags=re.IGNORECASE)
        if schedule_match:
            title = schedule_match.group(1).strip().strip("\"'")
            start = datetime.now() + timedelta(hours=1)
            start = start.replace(minute=0, second=0, microsecond=0)
            return title, start

        return "Untitled Event", datetime.now() + timedelta(hours=1)

    def _parse_date_time(self, date_part: str, time_part: str) -> datetime:
        now = datetime.now()
        date_lower = date_part.lower().strip()

        if date_lower == "today":
            base_date = now.date()
        elif date_lower == "tomorrow":
            base_date = (now + timedelta(days=1)).date()
        else:
            base_date = None
            for fmt in ("%Y-%m-%d", "%B %d %Y", "%B %d", "%b %d %Y", "%b %d", "%m/%d/%Y", "%m/%d"):
                try:
                    parsed = datetime.strptime(date_part, fmt)
                    if "%Y" in fmt:
                        base_date = parsed.date()
                    else:
                        base_date = parsed.replace(year=now.year).date()
                    break
                except ValueError:
                    continue
            if base_date is None:
                base_date = now.date()

        parsed_time = None
        for fmt in ("%I:%M %p", "%I %p", "%H:%M"):
            try:
                parsed_time = datetime.strptime(time_part.upper(), fmt).time()
                break
            except ValueError:
                continue

        if parsed_time is None:
            parsed_time = datetime.strptime("09:00", "%H:%M").time()

        return datetime.combine(base_date, parsed_time)

    def _fetch_today_events_eventkit(self) -> List[Tuple[str, str]]:
        try:
            import EventKit  # type: ignore
            import Foundation  # type: ignore
        except Exception as exc:
            raise RuntimeError("EventKit (PyObjC) is not installed") from exc

        store = EventKit.EKEventStore.alloc().init()

        access_granted = {"granted": False, "done": False, "error": None}
        done_event = threading.Event()

        def completion(granted, error):
            access_granted["granted"] = bool(granted)
            access_granted["error"] = error
            access_granted["done"] = True
            done_event.set()

        if hasattr(store, "requestFullAccessToEventsWithCompletion_"):
            store.requestFullAccessToEventsWithCompletion_(completion)
            done_event.wait(timeout=10)
        elif hasattr(store, "requestAccessToEntityType_completion_"):
            store.requestAccessToEntityType_completion_(EventKit.EKEntityTypeEvent, completion)
            done_event.wait(timeout=10)

        if access_granted["done"] and not access_granted["granted"]:
            raise RuntimeError("Calendar access denied")

        now = datetime.now()
        start_of_day = datetime(now.year, now.month, now.day)
        end_of_day = start_of_day + timedelta(days=1)

        start_date = Foundation.NSDate.dateWithTimeIntervalSince1970_(start_of_day.timestamp())
        end_date = Foundation.NSDate.dateWithTimeIntervalSince1970_(end_of_day.timestamp())

        predicate = store.predicateForEventsWithStartDate_endDate_calendars_(
            start_date,
            end_date,
            None,
        )

        events = list(store.eventsMatchingPredicate_(predicate) or [])
        events.sort(key=lambda item: item.startDate().timeIntervalSince1970())

        output: List[Tuple[str, str]] = []
        for item in events:
            title = str(item.title() or "Untitled")
            start = datetime.fromtimestamp(item.startDate().timeIntervalSince1970())
            output.append((title, start.strftime("%I:%M %p")))

        return output

    def _spotify_search_uri(self, query: str, item_type: str) -> Optional[str]:
        client_id = os.getenv("SPOTIFY_CLIENT_ID") or self._get_config("system.spotify.client_id")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET") or self._get_config("system.spotify.client_secret")

        if not client_id or not client_secret:
            raise RuntimeError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are required")

        token = self._spotify_client_credentials_token(client_id, client_secret)
        search_url = (
            "https://api.spotify.com/v1/search?"
            + parse.urlencode({"q": query, "type": item_type, "limit": 1})
        )

        req = request.Request(
            search_url,
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )

        with request.urlopen(req, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        bucket = payload.get(f"{item_type}s", {})
        items = bucket.get("items", [])
        if not items:
            return None

        return items[0].get("uri")

    def _spotify_client_credentials_token(self, client_id: str, client_secret: str) -> str:
        basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
        data = parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")

        req = request.Request(
            "https://accounts.spotify.com/api/token",
            data=data,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )

        with request.urlopen(req, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        access_token = payload.get("access_token")
        if not access_token:
            raise RuntimeError("Failed to retrieve Spotify access token")

        return access_token

    async def _set_brightness(self, level: int, event: IntentRecognizedEvent) -> None:
        normalized = max(0, min(100, level)) / 100.0
        try:
            result = subprocess.run(
                ["brightness", str(normalized)],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            await self._emit_voice(f"I couldn't set brightness: {exc}", event)
            return

        if result.returncode != 0:
            err = result.stderr.strip() or "brightness CLI failed"
            await self._emit_voice(f"I couldn't set brightness: {err}", event)
            return

        await self._emit_voice(f"Set brightness to {max(0, min(100, level))}", event)

    async def _set_wifi(self, enabled: bool, event: IntentRecognizedEvent) -> None:
        state = "on" if enabled else "off"
        try:
            result = subprocess.run(
                ["networksetup", "-setairportpower", "en0", state],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            await self._emit_voice(f"I couldn't change WiFi state: {exc}", event)
            return

        if result.returncode != 0:
            err = result.stderr.strip() or "networksetup failed"
            await self._emit_voice(f"I couldn't change WiFi state: {err}", event)
            return

        await self._emit_voice(f"WiFi turned {state}", event)

    async def _read_battery(self, event: IntentRecognizedEvent) -> None:
        try:
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            await self._emit_voice(f"I couldn't read battery status: {exc}", event)
            return

        if result.returncode != 0:
            err = result.stderr.strip() or "pmset failed"
            await self._emit_voice(f"I couldn't read battery status: {err}", event)
            return

        match = re.search(r"(\d+%)", result.stdout)
        if not match:
            await self._emit_voice("I couldn't parse battery percentage.", event)
            return

        await self._emit_voice(f"Battery is at {match.group(1)}", event)
