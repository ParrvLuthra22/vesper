"""
SystemAgent - Comprehensive macOS System Control Agent

This agent handles all system-level operations on macOS including:
- Application control (open, close, focus, list running apps)
- Volume control (get, set, mute, unmute)
- Brightness control (get, set)
- System stats (CPU, RAM, disk, battery)
- Screen control (sleep, lock)
- Notifications
- Clipboard operations

All operations use AppleScript (osascript) for reliable macOS integration.
The agent never interacts with voice or UI directly - it emits events.
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents.base_agent import AgentCapability, BaseAgent
from bus.event_bus import EventBus
from schemas.events import (
    ActionRequestEvent,
    ActionResultEvent,
    AgentErrorEvent,
    ApplicationLaunchedEvent,
    IntentRecognizedEvent,
    ShutdownRequestedEvent,
    SystemCommandEvent,
    SystemCommandResultEvent,
    VoiceOutputEvent,
)


# =============================================================================
# DATA CLASSES & ENUMS
# =============================================================================

# Same convention as agents.vision_agent.SCREENSHOT_PATH.
SCREENSHOT_PATH = Path("/tmp/vesper_screen.png")


class CommandStatus(Enum):
    """Status of a system command execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class CommandResult:
    """Result of a system command execution."""
    status: CommandStatus
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    command_type: str = ""
    
    @property
    def success(self) -> bool:
        return self.status == CommandStatus.SUCCESS


@dataclass
class SystemStats:
    """System statistics snapshot."""
    cpu_percent: float
    memory_total_gb: float
    memory_used_gb: float
    memory_percent: float
    disk_total_gb: float
    disk_used_gb: float
    disk_percent: float
    battery_percent: Optional[int]
    battery_charging: Optional[bool]
    uptime_seconds: float
    timestamp: datetime


@dataclass
class ApplicationInfo:
    """Information about a running application."""
    name: str
    bundle_id: str
    pid: int
    is_frontmost: bool


# =============================================================================
# APPLESCRIPT EXECUTOR
# =============================================================================

class AppleScriptExecutor:
    """
    Executes AppleScript commands for macOS system control.
    Thread-safe and with proper error handling.
    """
    
    OSASCRIPT_PATH = "/usr/bin/osascript"
    DEFAULT_TIMEOUT = 10.0  # seconds
    
    def __init__(self, logger=None):
        self._logger = logger
        self._check_osascript()
    
    def _check_osascript(self) -> None:
        """Verify osascript is available."""
        if not os.path.exists(self.OSASCRIPT_PATH):
            raise RuntimeError("osascript not found - this agent requires macOS")
    
    def _log(self, level: str, msg: str) -> None:
        """Log a message if logger is available."""
        if self._logger:
            log_func = getattr(self._logger, level, self._logger.info)
            log_func(msg)
    
    def run(
        self,
        script: str,
        timeout: float = DEFAULT_TIMEOUT,
        as_admin: bool = False
    ) -> Tuple[bool, str, str]:
        """
        Execute an AppleScript and return (success, stdout, stderr).
        
        Args:
            script: The AppleScript code to execute
            timeout: Maximum execution time in seconds
            as_admin: Whether to run with admin privileges (shows dialog)
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        start_time = time.perf_counter()
        
        try:
            cmd = [self.OSASCRIPT_PATH, "-e", script]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            elapsed = (time.perf_counter() - start_time) * 1000
            
            success = result.returncode == 0
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            if success:
                self._log("debug", f"AppleScript executed in {elapsed:.1f}ms")
            else:
                self._log("warning", f"AppleScript failed: {stderr}")
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            self._log("error", f"AppleScript timed out after {timeout}s")
            return False, "", f"Script timed out after {timeout} seconds"
        except Exception as e:
            self._log("error", f"AppleScript error: {e}")
            return False, "", str(e)
    
    def run_shell(
        self,
        command: str,
        timeout: float = DEFAULT_TIMEOUT
    ) -> Tuple[bool, str, str]:
        """
        Execute a shell command via AppleScript's 'do shell script'.
        This is useful for commands that need proper shell environment.
        """
        # Escape quotes in the command
        escaped = command.replace("\\", "\\\\").replace('"', '\\"')
        script = f'do shell script "{escaped}"'
        return self.run(script, timeout)
    
    # -------------------------------------------------------------------------
    # APPLICATION CONTROL
    # -------------------------------------------------------------------------
    
    def open_application(self, app_name: str) -> Tuple[bool, str]:
        """
        Open an application by name.
        
        Args:
            app_name: Name of the application (e.g., "Safari", "Finder")
            
        Returns:
            Tuple of (success, message)
        """
        # Clean the app name
        app_name = app_name.strip().rstrip(".")
        
        # Try different methods in order of reliability
        methods = [
            # Method 1: tell application to activate
            f'tell application "{app_name}" to activate',
            # Method 2: open via Finder
            f'tell application "Finder" to open application file id "{app_name}"',
            # Method 3: open command
            f'do shell script "open -a \\"{app_name}\\""',
        ]
        
        for i, script in enumerate(methods):
            success, stdout, stderr = self.run(script, timeout=15.0)
            if success:
                return True, f"Opened {app_name}"
            elif i == 0:
                # First method failed, log and try next
                self._log("debug", f"Method 1 failed for {app_name}, trying alternatives")
        
        return False, f"Failed to open {app_name}: {stderr}"
    
    def close_application(self, app_name: str, force: bool = False) -> Tuple[bool, str]:
        """
        Close an application by name.
        
        Args:
            app_name: Name of the application
            force: If True, force quit the application
            
        Returns:
            Tuple of (success, message)
        """
        app_name = app_name.strip().rstrip(".")
        
        if force:
            script = f'do shell script "killall \\"{app_name}\\" 2>/dev/null || true"'
        else:
            script = f'tell application "{app_name}" to quit'
        
        success, stdout, stderr = self.run(script)
        
        if success:
            return True, f"Closed {app_name}"
        else:
            return False, f"Failed to close {app_name}: {stderr}"
    
    def focus_application(self, app_name: str) -> Tuple[bool, str]:
        """Bring an application to the foreground."""
        app_name = app_name.strip().rstrip(".")
        script = f'tell application "{app_name}" to activate'
        success, stdout, stderr = self.run(script)
        
        if success:
            return True, f"Focused {app_name}"
        else:
            return False, f"Failed to focus {app_name}: {stderr}"
    
    def get_running_applications(self) -> List[str]:
        """Get list of running application names."""
        script = '''
        tell application "System Events"
            set appList to name of every process whose background only is false
            set AppleScript's text item delimiters to "|||"
            return appList as text
        end tell
        '''
        success, stdout, stderr = self.run(script)
        
        if success and stdout:
            return [app.strip() for app in stdout.split("|||") if app.strip()]
        return []
    
    def get_frontmost_application(self) -> Optional[str]:
        """Get the name of the frontmost application."""
        script = '''
        tell application "System Events"
            return name of first process whose frontmost is true
        end tell
        '''
        success, stdout, stderr = self.run(script)
        return stdout if success else None
    
    def is_application_running(self, app_name: str) -> bool:
        """Check if an application is currently running."""
        app_name = app_name.strip().rstrip(".")
        script = f'''
        tell application "System Events"
            return exists (process "{app_name}")
        end tell
        '''
        success, stdout, stderr = self.run(script)
        return success and stdout.lower() == "true"
    
    # -------------------------------------------------------------------------
    # VOLUME CONTROL
    # -------------------------------------------------------------------------
    
    def get_volume(self) -> Optional[int]:
        """Get current system volume (0-100)."""
        script = "output volume of (get volume settings)"
        success, stdout, stderr = self.run(script)
        
        if success and stdout.isdigit():
            return int(stdout)
        return None
    
    def set_volume(self, level: int) -> Tuple[bool, str]:
        """
        Set system volume.
        
        Args:
            level: Volume level (0-100)
            
        Returns:
            Tuple of (success, message)
        """
        level = max(0, min(100, level))
        script = f"set volume output volume {level}"
        success, stdout, stderr = self.run(script)
        
        if success:
            return True, f"Volume set to {level}%"
        else:
            return False, f"Failed to set volume: {stderr}"
    
    def mute(self) -> Tuple[bool, str]:
        """Mute system audio."""
        script = "set volume output muted true"
        success, stdout, stderr = self.run(script)
        
        if success:
            return True, "Audio muted"
        else:
            return False, f"Failed to mute: {stderr}"
    
    def unmute(self) -> Tuple[bool, str]:
        """Unmute system audio."""
        script = "set volume output muted false"
        success, stdout, stderr = self.run(script)
        
        if success:
            return True, "Audio unmuted"
        else:
            return False, f"Failed to unmute: {stderr}"
    
    def is_muted(self) -> Optional[bool]:
        """Check if system audio is muted."""
        script = "output muted of (get volume settings)"
        success, stdout, stderr = self.run(script)
        
        if success:
            return stdout.lower() == "true"
        return None
    
    def volume_up(self, step: int = 10) -> Tuple[bool, str]:
        """Increase volume by a step amount."""
        current = self.get_volume()
        if current is not None:
            new_level = min(100, current + step)
            return self.set_volume(new_level)
        return False, "Could not get current volume"
    
    def volume_down(self, step: int = 10) -> Tuple[bool, str]:
        """Decrease volume by a step amount."""
        current = self.get_volume()
        if current is not None:
            new_level = max(0, current - step)
            return self.set_volume(new_level)
        return False, "Could not get current volume"
    
    # -------------------------------------------------------------------------
    # BRIGHTNESS CONTROL
    # -------------------------------------------------------------------------
    
    def get_brightness(self) -> Optional[float]:
        """
        Get current display brightness (0.0-1.0).
        Note: On Apple Silicon Macs, brightness API access is limited.
        """
        # Try using brightness command if available
        if shutil.which("brightness"):
            script = r'do shell script "brightness -l 2>/dev/null | grep -o \"brightness [0-9.]*\" | head -1 | cut -d\" \" -f2"'
            success, stdout, stderr = self.run(script)
            
            if success and stdout:
                try:
                    return float(stdout)
                except ValueError:
                    pass
        
        # Brightness API access is limited on modern macOS
        return None
    
    def set_brightness(self, level: float) -> Tuple[bool, str]:
        """
        Set display brightness.
        
        Args:
            level: Brightness level (0.0-1.0 or 0-100)
            
        Returns:
            Tuple of (success, message)
        """
        # Normalize level to 0.0-1.0
        if level > 1.0:
            level = level / 100.0
        level = max(0.0, min(1.0, level))
        
        # Check if brightness command is available
        if shutil.which("brightness"):
            script = f'do shell script "brightness {level}"'
            success, stdout, stderr = self.run(script)
            
            if success:
                return True, f"Brightness set to {int(level * 100)}%"
            else:
                return False, f"Failed to set brightness: {stderr}"
        else:
            return False, "Brightness control requires 'brightness' tool (brew install brightness)"
    
    def brightness_up(self, step: int = 2) -> Tuple[bool, str]:
        """
        Increase brightness by simulating brightness key presses.
        Uses AppleScript to simulate F2 (brightness up) key.
        Requires Accessibility permissions for Terminal/osascript.
        """
        # Simulate pressing brightness up key (F2)
        script = '''
        tell application "System Events"
            repeat ''' + str(step) + ''' times
                key code 144 -- brightness up key
                delay 0.1
            end repeat
        end tell
        '''
        success, stdout, stderr = self.run(script)
        
        if success:
            return True, "Brightness increased"
        elif "1002" in stderr or "not allowed" in stderr.lower():
            return False, "I need Accessibility permissions to control brightness. Please add Terminal to System Settings, Privacy and Security, Accessibility"
        else:
            return False, f"Could not increase brightness: {stderr}"
    
    def brightness_down(self, step: int = 2) -> Tuple[bool, str]:
        """
        Decrease brightness by simulating brightness key presses.
        Uses AppleScript to simulate F1 (brightness down) key.
        Requires Accessibility permissions for Terminal/osascript.
        """
        # Simulate pressing brightness down key (F1)
        script = '''
        tell application "System Events"
            repeat ''' + str(step) + ''' times
                key code 145 -- brightness down key
                delay 0.1
            end repeat
        end tell
        '''
        success, stdout, stderr = self.run(script)
        
        if success:
            return True, "Brightness decreased"
        elif "1002" in stderr or "not allowed" in stderr.lower():
            return False, "I need Accessibility permissions to control brightness. Please add Terminal to System Settings, Privacy and Security, Accessibility"
        else:
            return False, f"Could not decrease brightness: {stderr}"
    
    # -------------------------------------------------------------------------
    # BATTERY & POWER
    # -------------------------------------------------------------------------
    
    def get_battery_level(self) -> Optional[int]:
        """Get battery percentage (0-100) or None if no battery."""
        script = 'do shell script "pmset -g batt | grep -Eo \\"[0-9]+%\\" | tr -d \\"%\\""'
        success, stdout, stderr = self.run(script)
        
        if success and stdout.isdigit():
            return int(stdout)
        return None
    
    def is_charging(self) -> Optional[bool]:
        """Check if the device is currently charging."""
        script = 'do shell script "pmset -g batt | head -1"'
        success, stdout, stderr = self.run(script)
        
        if success:
            return "AC Power" in stdout or "charging" in stdout.lower()
        return None
    
    def get_power_source(self) -> str:
        """Get current power source (AC Power / Battery Power)."""
        script = 'do shell script "pmset -g batt | head -1"'
        success, stdout, stderr = self.run(script)
        
        if success:
            if "AC Power" in stdout:
                return "AC Power"
            elif "Battery Power" in stdout:
                return "Battery Power"
        return "Unknown"
    
    # -------------------------------------------------------------------------
    # SYSTEM INFO & STATS
    # -------------------------------------------------------------------------
    
    def get_cpu_usage(self) -> Optional[float]:
        """Get current CPU usage percentage."""
        script = '''do shell script "top -l 1 | head -n 10 | grep 'CPU usage' | awk '{print $3}' | tr -d '%'"'''
        success, stdout, stderr = self.run(script)
        
        if success and stdout:
            try:
                return float(stdout)
            except ValueError:
                pass
        return None
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        script = '''do shell script "vm_stat | head -10"'''
        success, stdout, stderr = self.run(script)
        
        result = {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "percent": 0.0}
        
        if success:
            # Parse vm_stat output
            try:
                page_size = 16384  # Default for Apple Silicon
                
                # Get total physical memory
                total_script = 'do shell script "sysctl -n hw.memsize"'
                t_success, t_stdout, _ = self.run(total_script)
                if t_success and t_stdout.isdigit():
                    total_bytes = int(t_stdout)
                    result["total_gb"] = total_bytes / (1024**3)
                
                # Parse used memory from vm_stat
                lines = stdout.split("\n")
                stats = {}
                for line in lines:
                    if ":" in line:
                        key, val = line.split(":", 1)
                        val = val.strip().rstrip(".")
                        if val.isdigit():
                            stats[key.strip()] = int(val)
                
                # Calculate used memory
                wired = stats.get("Pages wired down", 0) * page_size
                active = stats.get("Pages active", 0) * page_size
                compressed = stats.get("Pages occupied by compressor", 0) * page_size
                
                used_bytes = wired + active + compressed
                result["used_gb"] = used_bytes / (1024**3)
                result["free_gb"] = result["total_gb"] - result["used_gb"]
                
                if result["total_gb"] > 0:
                    result["percent"] = (result["used_gb"] / result["total_gb"]) * 100
                    
            except Exception as e:
                self._log("error", f"Failed to parse memory info: {e}")
        
        return result
    
    def get_disk_info(self, path: str = "/") -> Dict[str, Any]:
        """Get disk usage information for a path."""
        script = f'do shell script "df -H \\"{path}\\" | tail -1"'
        success, stdout, stderr = self.run(script)
        
        result = {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "percent": 0.0}
        
        if success and stdout:
            try:
                parts = stdout.split()
                if len(parts) >= 5:
                    # Parse sizes (e.g., "500G", "250G")
                    def parse_size(s: str) -> float:
                        s = s.strip()
                        if s.endswith("T"):
                            return float(s[:-1]) * 1024
                        elif s.endswith("G"):
                            return float(s[:-1])
                        elif s.endswith("M"):
                            return float(s[:-1]) / 1024
                        elif s.endswith("K"):
                            return float(s[:-1]) / (1024**2)
                        return 0.0
                    
                    result["total_gb"] = parse_size(parts[1])
                    result["used_gb"] = parse_size(parts[2])
                    result["free_gb"] = parse_size(parts[3])
                    
                    # Parse percentage
                    pct = parts[4].rstrip("%")
                    if pct.isdigit():
                        result["percent"] = float(pct)
                        
            except Exception as e:
                self._log("error", f"Failed to parse disk info: {e}")
        
        return result
    
    def get_uptime(self) -> Optional[float]:
        """Get system uptime in seconds."""
        # Use a simpler command that doesn't require awk quoting
        script = 'do shell script "sysctl -n kern.boottime"'
        success, stdout, stderr = self.run(script)
        
        if success and stdout:
            # Parse output like: "{ sec = 1737700000, usec = 0 } ..."
            import re
            match = re.search(r'sec\s*=\s*(\d+)', stdout)
            if match:
                boot_time = int(match.group(1))
                return time.time() - boot_time
        return None
    
    def get_system_stats(self) -> SystemStats:
        """Get comprehensive system statistics."""
        cpu = self.get_cpu_usage() or 0.0
        memory = self.get_memory_info()
        disk = self.get_disk_info()
        battery = self.get_battery_level()
        charging = self.is_charging()
        uptime = self.get_uptime() or 0.0
        
        return SystemStats(
            cpu_percent=cpu,
            memory_total_gb=memory["total_gb"],
            memory_used_gb=memory["used_gb"],
            memory_percent=memory["percent"],
            disk_total_gb=disk["total_gb"],
            disk_used_gb=disk["used_gb"],
            disk_percent=disk["percent"],
            battery_percent=battery,
            battery_charging=charging,
            uptime_seconds=uptime,
            timestamp=datetime.now()
        )
    
    # -------------------------------------------------------------------------
    # SCREEN & DISPLAY
    # -------------------------------------------------------------------------
    
    def sleep_display(self) -> Tuple[bool, str]:
        """Put display(s) to sleep."""
        script = 'do shell script "pmset displaysleepnow"'
        success, stdout, stderr = self.run(script)
        
        if success:
            return True, "Display sleeping"
        else:
            return False, f"Failed to sleep display: {stderr}"
    
    def lock_screen(self) -> Tuple[bool, str]:
        """Lock the screen."""
        # Method 1: Activation Lock
        script = '''
        tell application "System Events"
            keystroke "q" using {command down, control down}
        end tell
        '''
        success, stdout, stderr = self.run(script)
        
        if not success:
            # Method 2: pmset
            success, stdout, stderr = self.run_shell("pmset displaysleepnow")
        
        if success:
            return True, "Screen locked"
        else:
            return False, f"Failed to lock screen: {stderr}"
    
    def get_screen_resolution(self) -> Optional[Tuple[int, int]]:
        """Get main display resolution."""
        script = 'do shell script "system_profiler SPDisplaysDataType | grep Resolution | head -1"'
        success, stdout, stderr = self.run(script)
        
        if success and stdout:
            # Parse "Resolution: 2560 x 1440" or similar
            match = re.search(r"(\d+)\s*x\s*(\d+)", stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
        return None
    
    # -------------------------------------------------------------------------
    # CLIPBOARD
    # -------------------------------------------------------------------------
    
    def get_clipboard(self) -> Optional[str]:
        """Get text from clipboard."""
        script = 'the clipboard as text'
        success, stdout, stderr = self.run(script)
        return stdout if success else None
    
    def set_clipboard(self, text: str) -> Tuple[bool, str]:
        """Set text to clipboard."""
        # Escape special characters
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        script = f'set the clipboard to "{escaped}"'
        success, stdout, stderr = self.run(script)
        
        if success:
            return True, "Text copied to clipboard"
        else:
            return False, f"Failed to set clipboard: {stderr}"
    
    # -------------------------------------------------------------------------
    # NOTIFICATIONS
    # -------------------------------------------------------------------------
    
    def show_notification(
        self,
        title: str,
        message: str,
        subtitle: str = "",
        sound: str = "default"
    ) -> Tuple[bool, str]:
        """
        Show a system notification.
        
        Args:
            title: Notification title
            message: Notification body
            subtitle: Optional subtitle
            sound: Sound name or "default"/"none"
        """
        # Escape quotes
        title = title.replace('"', '\\"')
        message = message.replace('"', '\\"')
        subtitle = subtitle.replace('"', '\\"')
        
        script = f'''
        display notification "{message}" with title "{title}"'''
        
        if subtitle:
            script += f' subtitle "{subtitle}"'
        
        if sound and sound != "none":
            script += f' sound name "{sound}"'
        
        success, stdout, stderr = self.run(script)
        
        if success:
            return True, "Notification shown"
        else:
            return False, f"Failed to show notification: {stderr}"
    
    # -------------------------------------------------------------------------
    # DIALOGS
    # -------------------------------------------------------------------------
    
    def show_alert(
        self,
        title: str,
        message: str,
        buttons: List[str] = None,
        default_button: int = 1
    ) -> Tuple[bool, str]:
        """
        Show an alert dialog and return the button clicked.
        Note: This blocks until user responds!
        """
        buttons = buttons or ["OK"]
        button_str = ", ".join(f'"{b}"' for b in buttons)
        
        title = title.replace('"', '\\"')
        message = message.replace('"', '\\"')
        
        script = f'''
        display alert "{title}" message "{message}" buttons {{{button_str}}} default button {default_button}
        return button returned of result
        '''
        
        success, stdout, stderr = self.run(script, timeout=60.0)
        
        if success:
            return True, stdout
        else:
            return False, f"Dialog failed: {stderr}"


# =============================================================================
# SYSTEM AGENT
# =============================================================================

class SystemAgent(BaseAgent):
    """
    System Agent for macOS system control.
    
    Handles intents and system commands by delegating to AppleScriptExecutor.
    Emits events for all results - never interacts with voice or UI directly.
    """
    
    # Intent to handler mapping (both upper and lower case for flexibility)
    INTENT_HANDLERS = {
        # Time & Date
        "get_time": "_handle_get_time",
        "GET_TIME": "_handle_get_time",
        "get_date": "_handle_get_date",
        "GET_DATE": "_handle_get_date",
        
        # Applications
        "open_app": "_handle_open_app",
        "OPEN_APP": "_handle_open_app",
        "open_application": "_handle_open_app",
        "OPEN_APPLICATION": "_handle_open_app",
        "close_app": "_handle_close_app",
        "CLOSE_APP": "_handle_close_app",
        "close_application": "_handle_close_app",
        "CLOSE_APPLICATION": "_handle_close_app",
        "focus_app": "_handle_focus_app",
        "FOCUS_APP": "_handle_focus_app",
        "list_apps": "_handle_list_apps",
        "LIST_APPS": "_handle_list_apps",
        
        # Volume
        "control_volume": "_handle_volume",
        "CONTROL_VOLUME": "_handle_volume",
        "set_volume": "_handle_volume",
        "SET_VOLUME": "_handle_volume",
        "get_volume": "_handle_get_volume",
        "GET_VOLUME": "_handle_get_volume",
        "volume_up": "_handle_volume_up",
        "VOLUME_UP": "_handle_volume_up",
        "volume_down": "_handle_volume_down",
        "VOLUME_DOWN": "_handle_volume_down",
        "mute": "_handle_mute",
        "MUTE": "_handle_mute",
        "unmute": "_handle_unmute",
        "UNMUTE": "_handle_unmute",
        
        # Brightness
        "control_brightness": "_handle_brightness",
        "CONTROL_BRIGHTNESS": "_handle_brightness",
        "set_brightness": "_handle_brightness",
        "SET_BRIGHTNESS": "_handle_brightness",
        "get_brightness": "_handle_get_brightness",
        "GET_BRIGHTNESS": "_handle_get_brightness",
        "brightness_up": "_handle_brightness_up",
        "BRIGHTNESS_UP": "_handle_brightness_up",
        "brightness_down": "_handle_brightness_down",
        "BRIGHTNESS_DOWN": "_handle_brightness_down",
        
        # System Info
        "system_info": "_handle_system_info",
        "SYSTEM_INFO": "_handle_system_info",
        "get_cpu": "_handle_get_cpu",
        "GET_CPU": "_handle_get_cpu",
        "get_memory": "_handle_get_memory",
        "GET_MEMORY": "_handle_get_memory",
        "get_battery": "_handle_get_battery",
        "GET_BATTERY": "_handle_get_battery",
        "get_disk": "_handle_get_disk",
        "GET_DISK": "_handle_get_disk",
        
        # Screen Control
        "sleep_display": "_handle_sleep_display",
        "SLEEP_DISPLAY": "_handle_sleep_display",
        "lock_screen": "_handle_lock_screen",
        "LOCK_SCREEN": "_handle_lock_screen",
        
        # Web
        "search_web": "_handle_search_web",
        "SEARCH_WEB": "_handle_search_web",
        "open_url": "_handle_open_url",
        "OPEN_URL": "_handle_open_url",

        # Notifications & screen capture
        "notify": "_handle_notify",
        "NOTIFY": "_handle_notify",
        "screenshot": "_handle_screenshot",
        "SCREENSHOT": "_handle_screenshot",
        
        # Social
        "greeting": "_handle_greeting",
        "GREETING": "_handle_greeting",
        "goodbye": "_handle_goodbye",
        "GOODBYE": "_handle_goodbye",
        "help": "_handle_help",
        "HELP": "_handle_help",
    }
    
    def __init__(self, event_bus: Optional[EventBus] = None, config: Any = None):
        super().__init__(name="SystemAgent", event_bus=event_bus, config=config)
        self.executor: Optional[AppleScriptExecutor] = None
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """List of capabilities this agent provides."""
        return [
            AgentCapability(
                name="application_control",
                description="Open, close, and manage applications",
                input_events=["IntentRecognizedEvent", "SystemCommandEvent"],
                output_events=["ApplicationLaunchedEvent", "SystemCommandResultEvent"],
            ),
            AgentCapability(
                name="volume_control",
                description="Control system volume (get, set, mute, unmute)",
                input_events=["IntentRecognizedEvent"],
                output_events=["SystemCommandResultEvent"],
            ),
            AgentCapability(
                name="brightness_control",
                description="Control screen brightness",
                input_events=["IntentRecognizedEvent"],
                output_events=["SystemCommandResultEvent"],
            ),
            AgentCapability(
                name="system_stats",
                description="Get CPU, memory, disk, and battery information",
                input_events=["IntentRecognizedEvent"],
                output_events=["SystemCommandResultEvent"],
            ),
            AgentCapability(
                name="screen_control",
                description="Lock screen, sleep display",
                input_events=["IntentRecognizedEvent"],
                output_events=["SystemCommandResultEvent"],
            ),
            AgentCapability(
                name="time_date",
                description="Get current time and date",
                input_events=["IntentRecognizedEvent"],
                output_events=["SystemCommandResultEvent", "VoiceOutputEvent"],
            ),
            AgentCapability(
                name="web_search",
                description="Search the web and open URLs",
                input_events=["IntentRecognizedEvent"],
                output_events=["SystemCommandResultEvent"],
            ),
        ]
    
    async def _setup(self) -> None:
        """Initialize the agent."""
        # Initialize AppleScript executor
        self.executor = AppleScriptExecutor(logger=self._logger)
        
        # In LangGraph mode, ToolAgent calls handlers directly to avoid duplicate responses.
        if bool(self._get_config("system.listen_intent_events", False)):
            self._subscribe(IntentRecognizedEvent, self._handle_intent)
        self._subscribe(SystemCommandEvent, self._handle_command)
        self._subscribe(ActionRequestEvent, self._handle_action_request)

        self._logger.info("System agent initialized")
    
    async def _teardown(self) -> None:
        """Clean up the agent."""
        self._logger.info("System agent shutdown complete")
    
    async def _handle_shutdown(self, event: ShutdownRequestedEvent) -> None:
        """Handle shutdown request."""
        self._logger.info(f"Received shutdown request: {event.payload.get('reason', 'unknown')}")
        await self.stop(reason=event.payload.get("reason", "shutdown"))
    
    async def _handle_intent(self, event: IntentRecognizedEvent) -> None:
        """Handle recognized intent event."""
        intent = event.payload.get("intent", "")
        entities = event.payload.get("entities", {})
        raw_text = event.payload.get("raw_text", "")
        
        # Find handler for this intent
        handler_name = self.INTENT_HANDLERS.get(intent)
        
        if handler_name:
            handler = getattr(self, handler_name, None)
            if handler and callable(handler):
                start_time = time.perf_counter()
                try:
                    result = handler(entities, raw_text)
                    execution_time = (time.perf_counter() - start_time) * 1000
                    
                    # Emit result event
                    await self._emit_result(
                        success=True,
                        result=result,
                        intent=intent,
                        execution_time=execution_time
                    )
                    
                    # Emit voice output
                    await self._speak(result)
                    
                except Exception as e:
                    self._logger.error(f"Handler error for {intent}: {e}")
                    execution_time = (time.perf_counter() - start_time) * 1000
                    await self._emit_result(
                        success=False,
                        result=None,
                        error=str(e),
                        intent=intent,
                        execution_time=execution_time
                    )
                    await self._speak(f"Sorry, I encountered an error: {e}")
    
    async def _handle_command(self, event: SystemCommandEvent) -> None:
        """Handle direct system command event."""
        command = event.payload.get("command", "")
        parameters = event.payload.get("parameters", {})
        
        # Execute command directly
        handler_name = self.INTENT_HANDLERS.get(command)
        
        if handler_name:
            handler = getattr(self, handler_name, None)
            if handler and callable(handler):
                start_time = time.perf_counter()
                try:
                    result = handler(parameters, "")
                    execution_time = (time.perf_counter() - start_time) * 1000
                    
                    await self._emit_result(
                        success=True,
                        result=result,
                        intent=command,
                        execution_time=execution_time
                    )
                except Exception as e:
                    self._logger.error(f"Command error for {command}: {e}")
                    execution_time = (time.perf_counter() - start_time) * 1000
                    await self._emit_result(
                        success=False,
                        result=None,
                        error=str(e),
                        intent=command,
                        execution_time=execution_time
                    )

    async def _handle_action_request(self, event: ActionRequestEvent) -> None:
        """
        Handle a Planner tool-call routed over the bus.

        Looks up event.action in INTENT_HANDLERS (the same table
        _handle_intent/_handle_command use), runs it, and emits an
        ActionResultEvent carrying the same correlation_id so the Planner's
        waiting future resolves.
        """
        if event.target_agent != self.name:
            return

        start_time = time.perf_counter()
        handler_name = self.INTENT_HANDLERS.get(event.action)
        handler = getattr(self, handler_name, None) if handler_name else None

        if not handler or not callable(handler):
            await self._emit_action_result(
                event, success=False, error=f"Unknown action '{event.action}'"
            )
            return

        try:
            result = handler(event.parameters, "")
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            await self._emit_action_result(
                event, success=True, result=result, execution_time_ms=execution_time_ms
            )
        except Exception as exc:
            self._logger.error(f"Action error for {event.action}: {exc}")
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            await self._emit_action_result(
                event, success=False, error=str(exc), execution_time_ms=execution_time_ms
            )

    async def _emit_action_result(
        self,
        event: ActionRequestEvent,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        execution_time_ms: float = 0.0,
    ) -> None:
        await self.event_bus.emit(ActionResultEvent(
            source=self.name,
            action=event.action,
            success=success,
            result=result,
            error=error or "",
            execution_time_ms=execution_time_ms,
            plan_id=event.plan_id,
            step_number=event.step_number,
            correlation_id=event.correlation_id,
        ))

    async def _emit_result(
        self,
        success: bool,
        result: Any,
        intent: str = "",
        error: Optional[str] = None,
        execution_time: float = 0.0
    ) -> None:
        """Emit a SystemCommandResultEvent."""
        await self.event_bus.emit(SystemCommandResultEvent(
            source=self.name,
            success=success,
            result=result,
            error=error,
            execution_time=execution_time,
            original_command=intent,
        ))
    
    async def _speak(self, text: str) -> None:
        """Emit a VoiceOutputEvent to speak text."""
        await self.event_bus.emit(VoiceOutputEvent(
            source=self.name,
            text=text,
            voice_id="default",
            speed=1.0,
            wait_for_completion=False,
        ))
    
    def _clean_app_name(self, app_name: str) -> str:
        """Clean up application name from entities."""
        if not app_name:
            return ""
        # Remove trailing punctuation and whitespace
        return app_name.strip().rstrip(".,!?")
    
    # =========================================================================
    # INTENT HANDLERS
    # =========================================================================
    
    def _handle_get_time(self, entities: Dict, raw_text: str) -> str:
        """Get current time."""
        now = datetime.now()
        time_str = now.strftime("%I:%M %p")
        return f"It's {time_str}"
    
    def _handle_get_date(self, entities: Dict, raw_text: str) -> str:
        """Get current date."""
        now = datetime.now()
        date_str = now.strftime("%A, %B %d, %Y")
        return f"Today is {date_str}"
    
    def _handle_open_app(self, entities: Dict, raw_text: str) -> str:
        """Open an application."""
        # Try multiple entity keys
        app_name = entities.get("app_name") or entities.get("app") or entities.get("application") or ""
        app_name = self._clean_app_name(app_name)
        
        if not app_name:
            return "I didn't catch which application you want to open"
        
        self._logger.info(f"Opening application: {app_name}")
        
        success, message = self.executor.open_application(app_name)
        
        if success:
            # Emit app launched event (fire and forget with asyncio.create_task)
            asyncio.create_task(self.event_bus.emit(ApplicationLaunchedEvent(
                source=self.name,
                app_name=app_name,
                app_bundle_id="",
                success=True,
            )))
            return f"Opening {app_name}"
        else:
            return f"I couldn't open {app_name}. {message}"
    
    def _handle_close_app(self, entities: Dict, raw_text: str) -> str:
        """Close an application."""
        app_name = entities.get("app_name") or entities.get("app") or entities.get("application") or ""
        app_name = self._clean_app_name(app_name)
        force = entities.get("force", False)
        
        if not app_name:
            return "I didn't catch which application you want to close"
        
        self._logger.info(f"Closing application: {app_name} (force={force})")
        
        success, message = self.executor.close_application(app_name, force=force)
        
        if success:
            return f"Closed {app_name}"
        else:
            return f"I couldn't close {app_name}. {message}"
    
    def _handle_focus_app(self, entities: Dict, raw_text: str) -> str:
        """Focus/activate an application."""
        app_name = entities.get("app_name") or entities.get("app") or ""
        app_name = self._clean_app_name(app_name)
        
        if not app_name:
            return "I didn't catch which application you want to focus"
        
        success, message = self.executor.focus_application(app_name)
        
        if success:
            return f"Focused {app_name}"
        else:
            return message
    
    def _handle_list_apps(self, entities: Dict, raw_text: str) -> str:
        """List running applications."""
        apps = self.executor.get_running_applications()
        
        if apps:
            count = len(apps)
            # Limit spoken list to first 5
            sample = apps[:5]
            sample_str = ", ".join(sample)
            
            if count <= 5:
                return f"You have {count} apps running: {sample_str}"
            else:
                return f"You have {count} apps running. Here are some: {sample_str}, and {count - 5} more"
        else:
            return "I couldn't get the list of running apps"
    
    def _handle_volume(self, entities: Dict, raw_text: str) -> str:
        """Control volume (set to specific level)."""
        level = entities.get("level") or entities.get("volume")
        action = entities.get("action", "").lower()
        
        # Infer action from raw_text if not in entities
        raw_lower = raw_text.lower()
        if not action:
            if any(word in raw_lower for word in ["mute", "silence", "quiet"]):
                if "unmute" in raw_lower or "un-mute" in raw_lower:
                    action = "unmute"
                else:
                    action = "mute"
            elif any(word in raw_lower for word in ["increase", "raise", "up", "higher", "louder", "turn up"]):
                action = "up"
            elif any(word in raw_lower for word in ["decrease", "lower", "down", "reduce", "softer", "turn down", "minimize"]):
                action = "down"
            elif any(word in raw_lower for word in ["max", "maximum", "full", "100"]):
                level = 100
            elif any(word in raw_lower for word in ["min", "minimum", "zero", "0"]):
                level = 0
        
        # Handle different actions
        if action == "mute":
            return self._handle_mute(entities, raw_text)
        elif action == "unmute":
            return self._handle_unmute(entities, raw_text)
        elif action == "up":
            return self._handle_volume_up(entities, raw_text)
        elif action == "down":
            return self._handle_volume_down(entities, raw_text)
        
        if level is not None:
            try:
                level = int(level)
            except (ValueError, TypeError):
                return "I didn't understand the volume level"
            
            success, message = self.executor.set_volume(level)
            return message
        
        # If no level specified, return current volume
        return self._handle_get_volume(entities, raw_text)
    
    def _handle_get_volume(self, entities: Dict, raw_text: str) -> str:
        """Get current volume level."""
        volume = self.executor.get_volume()
        muted = self.executor.is_muted()
        
        if volume is not None:
            if muted:
                return f"Volume is at {volume}% but muted"
            else:
                return f"Volume is at {volume}%"
        else:
            return "I couldn't get the volume level"
    
    def _handle_volume_up(self, entities: Dict, raw_text: str) -> str:
        """Increase volume."""
        step = entities.get("step", 10)
        try:
            step = int(step)
        except (ValueError, TypeError):
            step = 10
        
        success, message = self.executor.volume_up(step)
        if success:
            new_volume = self.executor.get_volume()
            return f"Volume increased to {new_volume}%"
        return message
    
    def _handle_volume_down(self, entities: Dict, raw_text: str) -> str:
        """Decrease volume."""
        step = entities.get("step", 10)
        try:
            step = int(step)
        except (ValueError, TypeError):
            step = 10
        
        success, message = self.executor.volume_down(step)
        if success:
            new_volume = self.executor.get_volume()
            return f"Volume decreased to {new_volume}%"
        return message
    
    def _handle_mute(self, entities: Dict, raw_text: str) -> str:
        """Mute system audio."""
        success, message = self.executor.mute()
        return message
    
    def _handle_unmute(self, entities: Dict, raw_text: str) -> str:
        """Unmute system audio."""
        success, message = self.executor.unmute()
        return message
    
    def _handle_brightness(self, entities: Dict, raw_text: str) -> str:
        """Set brightness level."""
        level = entities.get("level") or entities.get("brightness")
        action = entities.get("action", "").lower()
        
        # Infer action from raw_text if not in entities
        raw_lower = raw_text.lower()
        if not action:
            if any(word in raw_lower for word in ["increase", "raise", "up", "higher", "brighter", "turn up"]):
                action = "up"
            elif any(word in raw_lower for word in ["decrease", "lower", "down", "reduce", "dimmer", "dim", "turn down"]):
                action = "down"
            elif any(word in raw_lower for word in ["max", "maximum", "full", "100"]):
                level = 100
            elif any(word in raw_lower for word in ["min", "minimum", "zero", "0"]):
                level = 0
        
        if action == "up":
            return self._handle_brightness_up(entities, raw_text)
        elif action == "down":
            return self._handle_brightness_down(entities, raw_text)
        
        if level is not None:
            try:
                level = float(level)
            except (ValueError, TypeError):
                return "I didn't understand the brightness level"
            
            # Normalize to 0-100 if given as percentage
            if level > 1:
                level = level / 100.0
            
            success, message = self.executor.set_brightness(level)
            return message
        
        return self._handle_get_brightness(entities, raw_text)
    
    def _handle_get_brightness(self, entities: Dict, raw_text: str) -> str:
        """Get current brightness level."""
        brightness = self.executor.get_brightness()
        
        if brightness is not None:
            return f"Brightness is at {int(brightness * 100)}%"
        else:
            return "I can't read the exact brightness level on this Mac, but I can still adjust it for you"
    
    def _handle_brightness_up(self, entities: Dict, raw_text: str) -> str:
        """Increase brightness."""
        success, message = self.executor.brightness_up()
        if success:
            return "Brightness increased"
        return message
    
    def _handle_brightness_down(self, entities: Dict, raw_text: str) -> str:
        """Decrease brightness."""
        success, message = self.executor.brightness_down()
        if success:
            return "Brightness decreased"
        return message
    
    def _handle_system_info(self, entities: Dict, raw_text: str) -> str:
        """Get system information summary."""
        stats = self.executor.get_system_stats()
        
        parts = []
        parts.append(f"CPU usage is at {stats.cpu_percent:.1f}%")
        parts.append(f"Memory: {stats.memory_used_gb:.1f} of {stats.memory_total_gb:.1f} GB used ({stats.memory_percent:.0f}%)")
        parts.append(f"Disk: {stats.disk_used_gb:.0f} of {stats.disk_total_gb:.0f} GB used")
        
        if stats.battery_percent is not None:
            charging = " and charging" if stats.battery_charging else ""
            parts.append(f"Battery at {stats.battery_percent}%{charging}")
        
        # Format uptime
        uptime_hours = stats.uptime_seconds / 3600
        if uptime_hours < 1:
            uptime_str = f"{int(uptime_hours * 60)} minutes"
        elif uptime_hours < 24:
            uptime_str = f"{uptime_hours:.1f} hours"
        else:
            uptime_str = f"{uptime_hours / 24:.1f} days"
        parts.append(f"System uptime is {uptime_str}")
        
        return ". ".join(parts)
    
    def _handle_get_cpu(self, entities: Dict, raw_text: str) -> str:
        """Get CPU usage."""
        cpu = self.executor.get_cpu_usage()
        if cpu is not None:
            return f"CPU usage is at {cpu:.1f}%"
        return "I couldn't get CPU usage"
    
    def _handle_get_memory(self, entities: Dict, raw_text: str) -> str:
        """Get memory usage."""
        memory = self.executor.get_memory_info()
        return f"Memory: {memory['used_gb']:.1f} of {memory['total_gb']:.1f} GB used ({memory['percent']:.0f}%)"
    
    def _handle_get_battery(self, entities: Dict, raw_text: str) -> str:
        """Get battery status."""
        battery = self.executor.get_battery_level()
        charging = self.executor.is_charging()
        power_source = self.executor.get_power_source()
        
        if battery is not None:
            status = f"Battery is at {battery}%"
            if charging:
                status += " and charging"
            status += f" ({power_source})"
            return status
        else:
            return "No battery detected - you may be on a desktop Mac"
    
    def _handle_get_disk(self, entities: Dict, raw_text: str) -> str:
        """Get disk usage."""
        disk = self.executor.get_disk_info()
        return f"Disk: {disk['used_gb']:.0f} of {disk['total_gb']:.0f} GB used ({disk['percent']:.0f}% full)"
    
    def _handle_sleep_display(self, entities: Dict, raw_text: str) -> str:
        """Put display to sleep."""
        success, message = self.executor.sleep_display()
        return message
    
    def _handle_lock_screen(self, entities: Dict, raw_text: str) -> str:
        """Lock the screen."""
        success, message = self.executor.lock_screen()
        return message
    
    def _handle_search_web(self, entities: Dict, raw_text: str) -> str:
        """Search the web."""
        query = entities.get("query", "")
        engine = entities.get("engine", "google").lower()
        
        if not query:
            # Try to extract query from raw text
            query = raw_text.replace("search for", "").replace("search", "").strip()
        
        if not query:
            return "What would you like me to search for?"
        
        # URL encode the query
        from urllib.parse import quote_plus
        encoded_query = quote_plus(query)
        
        # Build search URL
        if engine == "duckduckgo":
            url = f"https://duckduckgo.com/?q={encoded_query}"
        elif engine == "bing":
            url = f"https://www.bing.com/search?q={encoded_query}"
        else:  # default to google
            url = f"https://www.google.com/search?q={encoded_query}"
        
        # Open in default browser
        success, stdout, stderr = self.executor.run_shell(f'open "{url}"')
        
        if success:
            return f"Searching for {query}"
        else:
            return f"I couldn't open the search. {stderr}"
    
    def _handle_open_url(self, entities: Dict, raw_text: str) -> str:
        """Open a URL in the browser."""
        url = entities.get("url", "")
        
        if not url:
            return "What URL would you like me to open?"
        
        # Add https if no protocol specified
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"
        
        success, stdout, stderr = self.executor.run_shell(f'open "{url}"')

        if success:
            return f"Opening {url}"
        else:
            return f"I couldn't open that URL. {stderr}"

    def _handle_notify(self, entities: Dict, raw_text: str) -> str:
        """Show a system notification."""
        title = entities.get("title", "Vesper")
        message = entities.get("message", "")

        if not message:
            return "I didn't catch what the notification should say"

        success, msg = self.executor.show_notification(title=title, message=message)
        return msg

    def _handle_screenshot(self, entities: Dict, raw_text: str) -> str:
        """Capture a screenshot of the current screen via macOS's screencapture CLI."""
        success, stdout, stderr = self.executor.run_shell(
            f'screencapture -x "{SCREENSHOT_PATH}"'
        )

        if success and SCREENSHOT_PATH.exists():
            return f"Screenshot saved to {SCREENSHOT_PATH}"
        else:
            return f"I couldn't take a screenshot. {stderr}"

    def _handle_greeting(self, entities: Dict, raw_text: str) -> str:
        """Respond to greeting."""
        hour = datetime.now().hour
        
        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"
        
        return f"{greeting}! How can I help you?"
    
    def _handle_goodbye(self, entities: Dict, raw_text: str) -> str:
        """Respond to goodbye."""
        return "Goodbye! Have a great day!"
    
    def _handle_help(self, entities: Dict, raw_text: str) -> str:
        """Provide help information."""
        return (
            "I can help you with many things! Try asking me to: "
            "open applications, control volume or brightness, "
            "get system information like CPU, memory, or battery, "
            "search the web, tell you the time or date, "
            "and more!"
        )
