"""
macOS HUD overlay for VESPER.

Design goals:
- Runs in its own thread (non-blocking to orchestrator/event loop)
- Subscribes to EventBus via existing pub/sub handlers
- Shows listening waveform, last command, latest response, event log, and agent health
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import platform
import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional

from bus.event_bus import EventBus, SubscriptionToken, get_event_bus
from schemas.events import (
    AgentErrorEvent,
    AgentHealthCheckEvent,
    AgentStartedEvent,
    AgentStoppedEvent,
    BaseEvent,
    HUDGraphStateEvent,
    HUDSearchResultsEvent,
    HUDUpdateEvent,
    ListeningStateChangedEvent,
    ResponseGeneratedEvent,
    VoiceInputEvent,
    VoiceOutputEvent,
)
from utils.logger import get_logger


logger = get_logger(__name__)


TARGET_AGENTS = [
    "VoiceAgent",
    "IntentAgent",
    "SystemAgent",
    "MemoryAgent",
    "VisionAgent",
    "WebSearchAgent",
    "MacOSControlAgent",
]


def _hud_process_main(config: Dict[str, Any], ipc_queue: "mp.Queue") -> None:
    """Run Tk HUD in a dedicated process (required on macOS)."""
    try:
        import tkinter as tk
    except Exception as exc:
        logger.error(f"HUD process failed to import tkinter: {exc}")
        return

    model = HUDModel()
    phase = 0.0

    width = int(config.get("width", 680))
    height = int(config.get("height", 380))
    collapsed_size = int(config.get("collapsed_size", 112))
    x = int(config.get("x", 24))
    y = int(config.get("y", 24))
    alpha = float(config.get("alpha", 0.88))
    background = str(config.get("background", "#0a0e1a"))

    root = tk.Tk()
    root.title("VESPER HUD")
    root.geometry(f"{collapsed_size}x{collapsed_size}+{x}+{y}")
    root.configure(bg=background)
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", alpha)

    try:
        root.focusmodel("passive")
        root.tk.call(
            "::tk::unsupported::MacWindowStyle",
            "style",
            root._w,
            "floating",
            "noActivates",
        )
    except Exception:
        pass

    wave_canvas = tk.Canvas(root, width=88, height=88, bg=background, highlightthickness=0, bd=0, cursor="hand2")
    wave_canvas.place(x=(collapsed_size - 88) // 2, y=(collapsed_size - 88) // 2)

    text_left_x = 120
    text_width = max(260, width - text_left_x - 150)

    command_label = tk.Label(
        root,
        text="Command: ",
        anchor="w",
        justify="left",
        fg="#ffffff",
        bg=background,
        font=("Menlo", 13),
        wraplength=text_width,
    )

    response_label = tk.Label(
        root,
        text="Response: ",
        anchor="w",
        justify="left",
        fg="#00e5ff",
        bg=background,
        font=("Menlo", 13),
        wraplength=text_width,
    )

    screenshot_label = tk.Label(
        root,
        text="",
        bg=background,
        bd=1,
        relief="solid",
    )

    transcript_label = tk.Label(
        root,
        text="Transcript",
        anchor="w",
        justify="left",
        fg="#9eb2d0",
        bg=background,
        font=("Menlo", 11),
    )

    transcript_text = tk.Text(
        root,
        fg="#d7dff0",
        bg="#111827",
        insertbackground="#d7dff0",
        highlightthickness=0,
        borderwidth=1,
        relief="solid",
        font=("Menlo", 10),
        wrap="word",
        state="disabled",
    )
    transcript_scrollbar = tk.Scrollbar(root, orient="vertical", command=transcript_text.yview)
    transcript_text.config(yscrollcommand=transcript_scrollbar.set)

    status_canvas = tk.Canvas(root, width=width - 36, height=30, bg=background, highlightthickness=0, bd=0)

    screenshot_photo = None
    generated_photo = None
    transcript_lines: Deque[str] = deque(maxlen=220)
    ui_expanded = False
    drag_state = {"start_x": 0, "start_y": 0, "win_x": x, "win_y": y, "moved": False}

    def render_status() -> None:
        status_canvas.delete("all")
        status_canvas.create_text(2, 15, anchor="w", text="Agents:", fill="#9eb2d0", font=("Menlo", 11))
        sx = 70
        for name in TARGET_AGENTS:
            healthy = model.agent_health.get(name, False)
            color = "#00ff85" if healthy else "#ff4b4b"
            status_canvas.create_oval(sx, 10, sx + 10, 20, fill=color, outline=color)
            status_canvas.create_text(sx + 16, 15, anchor="w", text=name, fill="#d7dff0", font=("Menlo", 10))
            sx += 112

    def render_transcript() -> None:
        transcript_text.config(state="normal")
        transcript_text.delete("1.0", "end")
        if transcript_lines:
            transcript_text.insert("end", "\n".join(transcript_lines))
        transcript_text.config(state="disabled")
        transcript_text.see("end")

    def append_transcript(prefix: str, value: str) -> None:
        text = (value or "").strip()
        if not text:
            return
        transcript_lines.append(f"{prefix}: {text}")
        if ui_expanded:
            render_transcript()

    def set_expanded(expanded: bool) -> None:
        nonlocal ui_expanded
        ui_expanded = expanded
        if ui_expanded:
            root.geometry(f"{width}x{height}+{x}+{y}")
            wave_canvas.place(x=18, y=18)
            command_label.place(x=text_left_x, y=18, width=text_width)
            response_label.place(x=text_left_x, y=60, width=text_width)
            screenshot_label.place(x=width - 130, y=18, width=110, height=76)
            transcript_label.place(x=18, y=108)
            transcript_text.place(x=18, y=128, width=width - 46, height=height - 178)
            transcript_scrollbar.place(x=width - 28, y=128, width=10, height=height - 178)
            status_canvas.place(x=18, y=height - 38)
            render_status()
            render_transcript()
        else:
            root.geometry(f"{collapsed_size}x{collapsed_size}+{x}+{y}")
            wave_canvas.place(x=(collapsed_size - 88) // 2, y=(collapsed_size - 88) // 2)
            command_label.place_forget()
            response_label.place_forget()
            screenshot_label.place_forget()
            transcript_label.place_forget()
            transcript_text.place_forget()
            transcript_scrollbar.place_forget()
            status_canvas.place_forget()

    def _on_press(event: Any) -> None:
        drag_state["start_x"] = event.x_root
        drag_state["start_y"] = event.y_root
        drag_state["win_x"] = root.winfo_x()
        drag_state["win_y"] = root.winfo_y()
        drag_state["moved"] = False

    def _on_motion(event: Any) -> None:
        nonlocal x, y
        dx = event.x_root - drag_state["start_x"]
        dy = event.y_root - drag_state["start_y"]
        if abs(dx) > 2 or abs(dy) > 2:
            drag_state["moved"] = True
        x = int(drag_state["win_x"] + dx)
        y = int(drag_state["win_y"] + dy)
        w = width if ui_expanded else collapsed_size
        h = height if ui_expanded else collapsed_size
        root.geometry(f"{w}x{h}+{x}+{y}")

    def _on_wave_release(_event: Any) -> None:
        if not drag_state["moved"]:
            set_expanded(not ui_expanded)

    def draw_wave() -> None:
        nonlocal phase
        phase = (phase + 0.22) % (2 * math.pi)

        wave_canvas.delete("wave")
        cx, cy = 44, 44
        pulse = 4 if model.is_listening else 1.2
        wobble = 0.5 + 0.5 * math.sin(phase)
        outer = 32 + pulse * wobble
        middle = 22 + (pulse * 0.7) * wobble
        inner = 12 + (pulse * 0.45) * wobble
        ring_color = "#00e5ff" if model.is_listening else "#3a5278"
        core_color = "#1a2a45" if model.is_listening else "#152036"

        wave_canvas.create_oval(
            cx - outer,
            cy - outer,
            cx + outer,
            cy + outer,
            outline=ring_color,
            width=2,
            tags="wave",
        )
        wave_canvas.create_oval(
            cx - middle,
            cy - middle,
            cx + middle,
            cy + middle,
            outline=ring_color,
            width=2,
            tags="wave",
        )
        wave_canvas.create_oval(
            cx - inner,
            cy - inner,
            cx + inner,
            cy + inner,
            fill=core_color,
            outline=ring_color,
            width=1,
            tags="wave",
        )

    root.bind("<ButtonPress-1>", _on_press, add="+")
    root.bind("<B1-Motion>", _on_motion, add="+")
    wave_canvas.bind("<ButtonRelease-1>", _on_wave_release, add="+")

    def apply_message(msg: Dict[str, Any]) -> bool:
        nonlocal screenshot_photo
        kind = msg.get("kind", "")
        if kind == "shutdown":
            return False
        if kind == "listening":
            model.is_listening = bool(msg.get("value", False))
        elif kind == "command":
            model.last_command = str(msg.get("value", "")).strip()
            command_label.config(text=f"Command: {model.last_command}")
            append_transcript("Command", model.last_command)
        elif kind == "response":
            model.last_response = str(msg.get("value", "")).strip()
            response_label.config(text=f"Response: {model.last_response}")
            append_transcript("Response", model.last_response)
        elif kind == "event_log":
            model.event_lines.append(str(msg.get("value", "")))
            append_transcript("Event", str(msg.get("value", "")))
        elif kind == "agent_health":
            name = str(msg.get("agent_name", ""))
            if name in TARGET_AGENTS:
                model.agent_health[name] = bool(msg.get("healthy", False))
                render_status()
        elif kind == "hud_image":
            image_path = str(msg.get("image_path", ""))
            if image_path and os.path.exists(image_path):
                try:
                    from PIL import Image, ImageTk

                    image = Image.open(image_path).convert("RGB")
                    image.thumbnail((110, 76))
                    screenshot_photo = ImageTk.PhotoImage(image)
                    screenshot_label.config(image=screenshot_photo)
                except Exception as exc:
                    logger.warning(f"Failed to render HUD thumbnail: {exc}")
        elif kind == "hud_generated_image":
            image_path = str(msg.get("image_path", "")).strip()
            prompt = str(msg.get("prompt", "")).strip()
            if image_path and os.path.exists(image_path):
                try:
                    from PIL import Image, ImageTk

                    image = Image.open(image_path).convert("RGB")
                    image.thumbnail((110, 76))
                    screenshot_photo = ImageTk.PhotoImage(image)
                    screenshot_label.config(image=screenshot_photo)
                    append_transcript("Image", f"Generated: {prompt}")
                except Exception as exc:
                    logger.warning(f"Failed to render generated image: {exc}")
        return True

    set_expanded(False)
    render_status()

    def tick() -> None:
        keep_running = True
        while True:
            try:
                msg = ipc_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            keep_running = apply_message(msg)
            if not keep_running:
                break

        if not keep_running:
            root.quit()
            root.destroy()
            return

        draw_wave()
        root.after(33, tick)

    tick()
    root.mainloop()


@dataclass
class HUDModel:
    """Thread-shared HUD model (mutated only on UI thread)."""

    is_listening: bool = False
    last_command: str = ""
    last_response: str = ""
    event_lines: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    search_query: str = ""
    search_summary: str = ""
    search_sources: List[Dict[str, str]] = field(default_factory=list)
    generated_image_path: str = ""
    generated_prompt: str = ""
    agent_health: Dict[str, bool] = field(
        default_factory=lambda: {name: False for name in TARGET_AGENTS}
    )


class VesperHUDOverlay:
    """Tkinter-based always-on-top HUD overlay running in a dedicated thread."""

    def __init__(
        self,
        width: int = 680,
        height: int = 380,
        x: int = 24,
        y: int = 24,
        alpha: float = 0.88,
        background: str = "#0a0e1a",
    ):
        self._width = width
        self._height = height
        self._x = x
        self._y = y
        self._alpha = alpha
        self._background = background

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._stop_requested = threading.Event()
        self._ops: "queue.Queue[Callable[[], None]]" = queue.Queue()

        self._root = None
        self._canvas = None
        self._command_label = None
        self._response_label = None
        self._event_labels: List[Any] = []
        self._status_canvas = None
        self._screenshot_label = None
        self._screenshot_image = None
        self._search_listbox = None
        self._search_summary_label = None
        self._generated_image_label = None
        self._generated_image_caption = None
        self._generated_image_photo = None

        self._pulse_phase = 0.0
        self._model = HUDModel()

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        """Start HUD UI in its own thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_requested.clear()
        self._thread = threading.Thread(target=self._run_ui, name="VesperHUD", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop HUD thread gracefully."""
        self._stop_requested.set()
        self._enqueue(self._shutdown_ui)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._running.clear()

    def set_listening(self, is_listening: bool) -> None:
        self._enqueue(lambda: self._set_listening_ui(is_listening))

    def set_last_command(self, text: str) -> None:
        self._enqueue(lambda: self._set_last_command_ui(text))

    def set_last_response(self, text: str) -> None:
        self._enqueue(lambda: self._set_last_response_ui(text))

    def add_event_log_line(self, timestamp: datetime, event_type: str, source: str) -> None:
        line = f"{timestamp.strftime('%H:%M:%S')} {event_type} {source}"
        self._enqueue(lambda: self._add_event_line_ui(line))

    def set_agent_health(self, agent_name: str, healthy: bool) -> None:
        if agent_name not in TARGET_AGENTS:
            return
        self._enqueue(lambda: self._set_agent_health_ui(agent_name, healthy))

    def set_screenshot(self, image_path: str) -> None:
        self._enqueue(lambda: self._set_screenshot_ui(image_path))

    def set_search_results(self, query: str, summary: str, sources: List[Dict[str, str]]) -> None:
        self._enqueue(lambda: self._set_search_results_ui(query, summary, sources))

    def set_generated_image(self, image_path: str, prompt: str) -> None:
        self._enqueue(lambda: self._set_generated_image_ui(image_path, prompt))

    def _enqueue(self, operation: Callable[[], None]) -> None:
        if self._stop_requested.is_set():
            return
        self._ops.put(operation)

    def _run_ui(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            self._root = root

            root.title("VESPER HUD")
            root.geometry(f"{self._width}x{self._height}+{self._x}+{self._y}")
            root.configure(bg=self._background)
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.attributes("-alpha", self._alpha)

            # macOS best-effort non-activating floating window.
            try:
                root.focusmodel("passive")
                root.tk.call(
                    "::tk::unsupported::MacWindowStyle",
                    "style",
                    root._w,
                    "floating",
                    "noActivates",
                )
            except Exception:
                pass

            self._build_layout(tk)
            self._running.set()
            self._tick()
            root.mainloop()
        except Exception as exc:
            # D4: the legacy tkinter overlay is best-effort. On any failure — an
            # incompatible Tk/macOS build (the "requires macOS N" AppKit error),
            # no window server, a headless session — degrade with ONE clean line
            # instead of a traceback. The Tauri HUD (hud/) is the intended UI.
            logger.warning(f"Legacy HUD overlay unavailable, continuing without it: {exc}")
        finally:
            self._running.clear()
            self._root = None

    def _build_layout(self, tk: Any) -> None:
        if not self._root:
            return

        self._canvas = tk.Canvas(
            self._root,
            width=88,
            height=88,
            bg=self._background,
            highlightthickness=0,
            bd=0,
        )
        self._canvas.place(x=18, y=18)

        right_panel_x = self._width - 286
        text_left_x = 120
        text_width = max(220, right_panel_x - text_left_x - 20)

        self._command_label = tk.Label(
            self._root,
            text="Command: ",
            anchor="w",
            justify="left",
            fg="#ffffff",
            bg=self._background,
            font=("Menlo", 13),
            wraplength=text_width,
        )
        self._command_label.place(x=text_left_x, y=18, width=text_width)

        self._response_label = tk.Label(
            self._root,
            text="Response: ",
            anchor="w",
            justify="left",
            fg="#00e5ff",
            bg=self._background,
            font=("Menlo", 13),
            wraplength=text_width,
        )
        self._response_label.place(x=text_left_x, y=60, width=text_width)

        self._screenshot_label = tk.Label(
            self._root,
            text="",
            bg=self._background,
            bd=1,
            relief="solid",
        )
        self._screenshot_label.place(x=self._width - 130, y=18, width=110, height=76)

        search_header = tk.Label(
            self._root,
            text="Search Sources",
            anchor="w",
            justify="left",
            fg="#9eb2d0",
            bg=self._background,
            font=("Menlo", 11),
        )
        search_header.place(x=self._width - 280, y=104)

        self._search_listbox = tk.Listbox(
            self._root,
            fg="#d7dff0",
            bg="#111827",
            selectbackground="#1f2937",
            highlightthickness=0,
            borderwidth=1,
            relief="solid",
            font=("Menlo", 9),
        )
        self._search_listbox.place(x=self._width - 280, y=124, width=250, height=120)

        search_scrollbar = tk.Scrollbar(self._root, orient="vertical", command=self._search_listbox.yview)
        search_scrollbar.place(x=self._width - 30, y=124, width=10, height=120)
        self._search_listbox.config(yscrollcommand=search_scrollbar.set)

        self._search_summary_label = tk.Label(
            self._root,
            text="",
            anchor="w",
            justify="left",
            fg="#bfe3ff",
            bg=self._background,
            font=("Menlo", 9),
            wraplength=250,
        )
        self._search_summary_label.place(x=self._width - 280, y=248, width=250, height=66)

        self._generated_image_label = tk.Label(
            self._root,
            text="",
            bg=self._background,
            bd=1,
            relief="solid",
        )
        self._generated_image_label.place(
            x=self._width - 286,
            y=self._height - 286,
            width=256,
            height=256,
        )

        self._generated_image_caption = tk.Label(
            self._root,
            text="",
            anchor="w",
            justify="left",
            fg="#d6ecff",
            bg=self._background,
            font=("Menlo", 9),
            wraplength=256,
        )
        self._generated_image_caption.place(
            x=self._width - 286,
            y=self._height - 28,
            width=256,
            height=22,
        )

        header = tk.Label(
            self._root,
            text="Event Log",
            anchor="w",
            justify="left",
            fg="#9eb2d0",
            bg=self._background,
            font=("Menlo", 11),
        )
        header.place(x=18, y=128)

        self._event_labels = []
        for idx in range(5):
            line = tk.Label(
                self._root,
                text="",
                anchor="w",
                justify="left",
                fg="#d7dff0",
                bg=self._background,
                font=("Menlo", 10),
            )
            line.place(x=18, y=150 + idx * 24)
            self._event_labels.append(line)

        self._status_canvas = tk.Canvas(
            self._root,
            width=self._width - 36,
            height=30,
            bg=self._background,
            highlightthickness=0,
            bd=0,
        )
        self._status_canvas.place(x=18, y=self._height - 38)

        self._render_status_row()
        self._render_event_lines()
        self._draw_waveform()

    def _tick(self) -> None:
        if not self._root:
            return

        while True:
            try:
                op = self._ops.get_nowait()
            except queue.Empty:
                break
            try:
                op()
            except Exception as exc:
                logger.warning(f"HUD operation failed: {exc}")

        self._pulse_phase = (self._pulse_phase + 0.22) % (2 * math.pi)
        self._draw_waveform()

        if self._stop_requested.is_set():
            self._shutdown_ui()
            return

        self._root.after(33, self._tick)

    def _shutdown_ui(self) -> None:
        if self._root is None:
            return
        try:
            self._root.quit()
            self._root.destroy()
        except Exception:
            pass

    def _set_listening_ui(self, is_listening: bool) -> None:
        self._model.is_listening = is_listening

    def _set_last_command_ui(self, text: str) -> None:
        self._model.last_command = text.strip()
        if self._command_label is not None:
            self._command_label.config(text=f"Command: {self._model.last_command}")

    def _set_last_response_ui(self, text: str) -> None:
        self._model.last_response = text.strip()
        if self._response_label is not None:
            self._response_label.config(text=f"Response: {self._model.last_response}")

    def _add_event_line_ui(self, line: str) -> None:
        self._model.event_lines.append(line)
        self._render_event_lines()

    def _render_event_lines(self) -> None:
        padded = list(self._model.event_lines)
        padded = ["" for _ in range(max(0, 5 - len(padded)))] + padded
        for label, text in zip(self._event_labels, padded):
            label.config(text=text)

    def _set_agent_health_ui(self, agent_name: str, healthy: bool) -> None:
        self._model.agent_health[agent_name] = healthy
        self._render_status_row()

    def _set_screenshot_ui(self, image_path: str) -> None:
        if self._screenshot_label is None or not image_path or not os.path.exists(image_path):
            return

        try:
            from PIL import Image, ImageTk

            image = Image.open(image_path).convert("RGB")
            image.thumbnail((110, 76))
            self._screenshot_image = ImageTk.PhotoImage(image)
            self._screenshot_label.config(image=self._screenshot_image)
        except Exception as exc:
            logger.warning(f"Failed to render HUD screenshot: {exc}")

    def _set_search_results_ui(self, query: str, summary: str, sources: List[Dict[str, str]]) -> None:
        self._model.search_query = query.strip()
        self._model.search_summary = summary.strip()
        self._model.search_sources = list(sources or [])

        if self._search_listbox is not None:
            self._search_listbox.delete(0, "end")
            if self._model.search_query:
                self._search_listbox.insert("end", f"Query: {self._model.search_query}")
                self._search_listbox.insert("end", "")

            for idx, source in enumerate(self._model.search_sources, start=1):
                title = str(source.get("title", "Untitled"))
                url = str(source.get("url", ""))
                self._search_listbox.insert("end", f"{idx}. {title}")
                self._search_listbox.insert("end", f"   {url}")
                self._search_listbox.insert("end", "")

        if self._search_summary_label is not None:
            self._search_summary_label.config(text=self._model.search_summary)

    def _set_generated_image_ui(self, image_path: str, prompt: str) -> None:
        if (
            self._generated_image_label is None
            or self._generated_image_caption is None
            or not image_path
            or not os.path.exists(image_path)
        ):
            return

        self._model.generated_image_path = image_path
        self._model.generated_prompt = prompt.strip()

        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB").resize((256, 256))
            self._animate_generated_image_ui(image, 0)
            self._generated_image_caption.config(text=self._model.generated_prompt)
        except Exception as exc:
            logger.warning(f"Failed to render generated HUD image: {exc}")

    def _animate_generated_image_ui(self, image: Any, frame: int) -> None:
        if self._generated_image_label is None or self._root is None:
            return

        try:
            from PIL import ImageEnhance, ImageTk
        except Exception:
            return

        total_frames = 10
        alpha = max(0.0, min(1.0, frame / total_frames))
        faded = ImageEnhance.Brightness(image).enhance(alpha)
        self._generated_image_photo = ImageTk.PhotoImage(faded)
        self._generated_image_label.config(image=self._generated_image_photo)

        if frame < total_frames:
            self._root.after(50, lambda: self._animate_generated_image_ui(image, frame + 1))

    def _render_status_row(self) -> None:
        if self._status_canvas is None:
            return

        canvas = self._status_canvas
        canvas.delete("all")
        canvas.create_text(
            2,
            15,
            anchor="w",
            text="Agents:",
            fill="#9eb2d0",
            font=("Menlo", 11),
        )

        x = 70
        for name in TARGET_AGENTS:
            healthy = self._model.agent_health.get(name, False)
            color = "#00ff85" if healthy else "#ff4b4b"
            canvas.create_oval(x, 10, x + 10, 20, fill=color, outline=color)
            canvas.create_text(
                x + 16,
                15,
                anchor="w",
                text=name,
                fill="#d7dff0",
                font=("Menlo", 10),
            )
            x += 112

    def _draw_waveform(self) -> None:
        if self._canvas is None:
            return

        c = self._canvas
        c.delete("wave")

        cx, cy = 44, 44
        base = 15
        pulse = 10 if self._model.is_listening else 2
        outer = base + pulse * (0.5 + 0.5 * math.sin(self._pulse_phase))
        inner = max(8, outer - 8)

        ring_color = "#00e5ff" if self._model.is_listening else "#3a5278"
        core_color = "#1a2a45" if self._model.is_listening else "#152036"

        c.create_oval(
            cx - outer,
            cy - outer,
            cx + outer,
            cy + outer,
            outline=ring_color,
            width=3,
            tags="wave",
        )
        c.create_oval(
            cx - inner,
            cy - inner,
            cx + inner,
            cy + inner,
            fill=core_color,
            outline=ring_color,
            width=2,
            tags="wave",
        )


class HUDOverlayController:
    """EventBus bridge that feeds the HUD without blocking orchestrator flow."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._event_bus = event_bus or get_event_bus()
        self._config = config or {}
        self._tokens: List[SubscriptionToken] = []
        self._hud: Optional[VesperHUDOverlay] = None
        self._hud_process: Optional[mp.Process] = None
        self._hud_process_queue: Optional[mp.Queue] = None
        self._use_process_mode = platform.system() == "Darwin"

    async def start(self) -> None:
        ui_cfg = (self._config.get("ui") or {}).get("hud") or {}
        if ui_cfg.get("enabled", True) is False:
            logger.info("HUD disabled via config")
            return

        hud_kwargs = {
            "width": int(ui_cfg.get("width", 680)),
            "height": int(ui_cfg.get("height", 380)),
            "collapsed_size": int(ui_cfg.get("collapsed_size", 112)),
            "x": int(ui_cfg.get("x", 24)),
            "y": int(ui_cfg.get("y", 24)),
            "alpha": float(ui_cfg.get("alpha", 0.88)),
            "background": str(ui_cfg.get("background", "#0a0e1a")),
        }

        if self._use_process_mode:
            self._hud_process_queue = mp.Queue()
            self._hud_process = mp.Process(
                target=_hud_process_main,
                args=(hud_kwargs, self._hud_process_queue),
                name="VesperHUDProcess",
                daemon=True,
            )
            self._hud_process.start()
        else:
            self._hud = VesperHUDOverlay(**hud_kwargs)
            self._hud.start()

        self._tokens.append(self._event_bus.subscribe(BaseEvent, self._on_any_event))
        self._tokens.append(self._event_bus.subscribe(ListeningStateChangedEvent, self._on_listening_state))
        self._tokens.append(self._event_bus.subscribe(VoiceInputEvent, self._on_voice_input))
        self._tokens.append(self._event_bus.subscribe(VoiceOutputEvent, self._on_voice_output))
        self._tokens.append(self._event_bus.subscribe(ResponseGeneratedEvent, self._on_response_generated))
        self._tokens.append(self._event_bus.subscribe(AgentStartedEvent, self._on_agent_started))
        self._tokens.append(self._event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped))
        self._tokens.append(self._event_bus.subscribe(AgentErrorEvent, self._on_agent_error))
        self._tokens.append(self._event_bus.subscribe(AgentHealthCheckEvent, self._on_agent_health))
        self._tokens.append(self._event_bus.subscribe(HUDUpdateEvent, self._on_hud_update))
        self._tokens.append(self._event_bus.subscribe(HUDGraphStateEvent, self._on_hud_graph_state))

        logger.info("HUD overlay controller started")

    async def stop(self) -> None:
        for token in self._tokens:
            token.unsubscribe()
        self._tokens.clear()

        if self._hud:
            self._hud.stop()
            self._hud = None

        if self._hud_process_queue:
            try:
                self._hud_process_queue.put_nowait({"kind": "shutdown"})
            except Exception:
                pass

        if self._hud_process and self._hud_process.is_alive():
            self._hud_process.join(timeout=1.0)
            if self._hud_process.is_alive():
                self._hud_process.terminate()

        self._hud_process = None
        self._hud_process_queue = None

        logger.info("HUD overlay controller stopped")

    def set_agent_health(self, agent_name: str, healthy: bool) -> None:
        if self._hud:
            self._hud.set_agent_health(agent_name, healthy)
        elif self._hud_process_queue:
            self._push_process_message(
                {
                    "kind": "agent_health",
                    "agent_name": agent_name,
                    "healthy": healthy,
                }
            )

    def _push_process_message(self, message: Dict[str, Any]) -> None:
        if not self._hud_process_queue:
            return
        try:
            self._hud_process_queue.put_nowait(message)
        except Exception:
            pass

    async def _on_any_event(self, event: BaseEvent) -> None:
        if self._hud:
            self._hud.add_event_log_line(event.timestamp, event.type, event.source)
        else:
            self._push_process_message(
                {
                    "kind": "event_log",
                    "value": f"{event.timestamp.strftime('%H:%M:%S')} {event.type} {event.source}",
                }
            )

    async def _on_listening_state(self, event: ListeningStateChangedEvent) -> None:
        if self._hud:
            self._hud.set_listening(event.is_listening)
        else:
            self._push_process_message({"kind": "listening", "value": event.is_listening})

    async def _on_voice_input(self, event: VoiceInputEvent) -> None:
        if self._hud:
            self._hud.set_last_command(event.text)
        else:
            self._push_process_message({"kind": "command", "value": event.text})

    async def _on_voice_output(self, event: VoiceOutputEvent) -> None:
        if self._hud:
            self._hud.set_last_response(event.text)
        else:
            self._push_process_message({"kind": "response", "value": event.text})

    async def _on_response_generated(self, event: ResponseGeneratedEvent) -> None:
        response = event.display_text or event.text
        if self._hud:
            self._hud.set_last_response(response)
        else:
            self._push_process_message({"kind": "response", "value": response})

    async def _on_agent_started(self, event: AgentStartedEvent) -> None:
        if self._hud:
            self._hud.set_agent_health(event.agent_name, True)
        else:
            self._push_process_message(
                {
                    "kind": "agent_health",
                    "agent_name": event.agent_name,
                    "healthy": True,
                }
            )

    async def _on_agent_stopped(self, event: AgentStoppedEvent) -> None:
        if self._hud:
            self._hud.set_agent_health(event.agent_name, False)
        else:
            self._push_process_message(
                {
                    "kind": "agent_health",
                    "agent_name": event.agent_name,
                    "healthy": False,
                }
            )

    async def _on_agent_error(self, event: AgentErrorEvent) -> None:
        if self._hud:
            self._hud.set_agent_health(event.agent_name, False)
        else:
            self._push_process_message(
                {
                    "kind": "agent_health",
                    "agent_name": event.agent_name,
                    "healthy": False,
                }
            )

    async def _on_agent_health(self, event: AgentHealthCheckEvent) -> None:
        if self._hud:
            self._hud.set_agent_health(event.agent_name, event.is_healthy)
        else:
            self._push_process_message(
                {
                    "kind": "agent_health",
                    "agent_name": event.agent_name,
                    "healthy": event.is_healthy,
                }
            )

    async def _on_hud_update(self, event: HUDUpdateEvent) -> None:
        if not event.image_path:
            return

        if self._hud:
            self._hud.set_screenshot(event.image_path)
        else:
            self._push_process_message(
                {
                    "kind": "hud_image",
                    "image_path": event.image_path,
                }
            )

    async def _on_hud_search_results(self, event: HUDSearchResultsEvent) -> None:
        if self._hud:
            self._hud.set_search_results(event.query, event.summary, event.sources)
        else:
            self._push_process_message(
                {
                    "kind": "hud_search_results",
                    "query": event.query,
                    "summary": event.summary,
                    "sources": event.sources,
                }
            )

    async def _on_hud_graph_state(self, event: HUDGraphStateEvent) -> None:
        plan_preview = " | ".join(event.plan_steps[:4]) if event.plan_steps else "No plan yet"
        latest_result = event.tool_results[-1] if event.tool_results else "No tool output yet"

        command_text = f"Graph[{event.current_node}] {plan_preview}"
        response_text = latest_result

        if self._hud:
            self._hud.set_last_command(command_text)
            self._hud.set_last_response(response_text)
        else:
            self._push_process_message({"kind": "command", "value": command_text})
            self._push_process_message({"kind": "response", "value": response_text})
