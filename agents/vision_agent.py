"""VisionAgent - screenshot-driven vision workflows via Gemini."""

from __future__ import annotations

import asyncio
import base64
import difflib
import json
import ssl
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import parse, request

from PIL import Image, ImageGrab

try:
    import pyautogui

    PYAUTOGUI_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent import
    pyautogui = None
    PYAUTOGUI_AVAILABLE = False

try:
    import google.generativeai as genai

    GEMINI_SDK_AVAILABLE = True
except Exception:  # pragma: no cover - dependency/environment dependent
    genai = None
    GEMINI_SDK_AVAILABLE = False

try:
    from google import genai as google_genai
    from google.genai import types as google_genai_types

    GEMINI_GENAI_AVAILABLE = True
except Exception:  # pragma: no cover - dependency/environment dependent
    google_genai = None  # type: ignore
    google_genai_types = None  # type: ignore
    GEMINI_GENAI_AVAILABLE = False

try:
    import certifi

    CERTIFI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    certifi = None  # type: ignore
    CERTIFI_AVAILABLE = False

try:
    import pytesseract

    OCR_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore
    OCR_AVAILABLE = False

from agents.base_agent import AgentCapability, BaseAgent
from schemas.events import (
    HUDUpdateEvent,
    IntentRecognizedEvent,
    ScreenshotEvent,
    VoiceOutputEvent,
)
from utils.api_keys import get_gemini_api_key


SCREENSHOT_PATH = Path("/tmp/vesper_screen.png")
DEFAULT_GEMINI_MODEL_NAME = "gemini-2.0-flash"
MIN_CLICK_CONFIDENCE = 0.7
MAX_OCR_RESPONSE_CHARS = 1800


@dataclass
class CoordinateResult:
    x: Optional[int]
    y: Optional[int]
    confidence: float
    raw: str = ""

    @property
    def has_coordinates(self) -> bool:
        return self.x is not None and self.y is not None


@dataclass
class OCRWord:
    text: str
    x: int
    y: int
    confidence: float


class VisionAgent(BaseAgent):
    """Agent that captures screenshots and delegates visual reasoning to Gemini."""

    def __init__(
        self,
        name: Optional[str] = None,
        event_bus=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name or "VisionAgent", event_bus=event_bus, config=config)
        self._model = None
        self._genai_client = None
        self._gemini_model_name = DEFAULT_GEMINI_MODEL_NAME
        self._use_gemini = False
        self._local_ocr_enabled = True
        self._gemini_enabled = False
        self._gemini_status_reason = "not initialized"
        self._gemini_api_key: Optional[str] = None

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="screen_understanding",
                description="Captures screen, answers visual questions, and performs safe click actions",
                input_events=["IntentRecognizedEvent", "ScreenshotEvent"],
                output_events=["VoiceOutputEvent", "HUDUpdateEvent"],
            )
        ]

    async def _setup(self) -> None:
        self._configure_pyautogui()
        self._use_gemini = self._is_truthy(self._get_config("vision.use_gemini", False))
        self._local_ocr_enabled = self._is_truthy(self._get_config("vision.local_ocr_enabled", True))
        if self._use_gemini:
            self._initialize_gemini()
        else:
            self._gemini_enabled = False
            self._gemini_status_reason = "disabled (local OCR mode)"
        self._subscribe(IntentRecognizedEvent, self._handle_intent)
        self._subscribe(ScreenshotEvent, self._handle_screenshot_event)

    async def _teardown(self) -> None:
        self._model = None
        self._genai_client = None
        self._gemini_enabled = False
        self._gemini_api_key = None
        self._gemini_status_reason = "stopped"

    def _configure_pyautogui(self) -> None:
        if PYAUTOGUI_AVAILABLE and pyautogui is not None:
            pyautogui.FAILSAFE = True

    def _initialize_gemini(self) -> bool:
        self._gemini_enabled = False
        self._model = None
        self._genai_client = None
        self._gemini_api_key = None
        self._gemini_status_reason = "initializing"
        self._gemini_model_name = str(
            self._get_config("vision.gemini.model")
            or self._get_config("intent.gemini.model")
            or DEFAULT_GEMINI_MODEL_NAME
        ).strip()

        # Reload .env keys if user updated them while the app is running.
        try:
            from dotenv import load_dotenv

            project_root = Path(__file__).resolve().parents[1]
            load_dotenv(project_root / ".env", override=True)
        except Exception:
            pass

        api_key = get_gemini_api_key(self._get_config)
        if not api_key:
            self._logger.warning("Gemini API key not configured for VisionAgent")
            self._gemini_status_reason = (
                "missing API key (set GEMINI_API_KEY or gemini_api_key, "
                "or configure intent.gemini.api_key)"
            )
            return False

        self._gemini_api_key = api_key

        try:
            if GEMINI_SDK_AVAILABLE and genai is not None:
                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel(model_name=self._gemini_model_name)
                self._gemini_enabled = True
                self._gemini_status_reason = "ready (google-generativeai)"
                self._logger.info(
                    f"Vision Gemini initialized with model={self._gemini_model_name}"
                )
                return True

            if GEMINI_GENAI_AVAILABLE and google_genai is not None:
                self._genai_client = google_genai.Client(api_key=api_key)
                self._gemini_enabled = True
                self._gemini_status_reason = "ready (google-genai)"
                self._logger.info(
                    f"Vision Gemini initialized with google-genai model={self._gemini_model_name}"
                )
                return True

            # SDK unavailable: continue with REST fallback.
            self._gemini_enabled = True
            self._gemini_status_reason = "ready (REST fallback)"
            self._logger.warning(
                "google-generativeai SDK unavailable; using Gemini REST fallback in VisionAgent."
            )
            return True
        except Exception as exc:  # pragma: no cover - remote SDK init
            # Initialization failed with SDK path; keep REST fallback enabled if key exists.
            self._model = None
            self._gemini_enabled = True
            self._gemini_status_reason = f"ready (REST fallback after SDK init failure: {exc})"
            self._logger.warning(
                f"Failed to initialize Gemini SDK client ({exc}); using REST fallback."
            )
            return True

    async def _handle_screenshot_event(self, event: ScreenshotEvent) -> None:
        try:
            if event.bbox:
                await self.capture_region(*event.bbox)
            else:
                await self.capture_full_screen()
            await self._emit(
                HUDUpdateEvent(
                    image_path=str(SCREENSHOT_PATH),
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )
        except Exception as exc:
            await self._emit(
                VoiceOutputEvent(
                    text=f"I couldn't capture the screen: {exc}",
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )

    async def _handle_intent(self, event: IntentRecognizedEvent) -> None:
        intent_text = (event.intent or "").strip().lower()
        raw_text = (event.raw_text or "").strip().lower()
        candidates = [intent_text, raw_text]

        if self._matches_any(
            candidates,
            [
                "what's on my screen",
                "whats on my screen",
                "what is on my screen",
                "read my screen",
                "read screen",
                "describe screen",
                "describe my screen",
            ],
        ):
            await self._answer_visual_prompt(
                event=event,
                prompt="Describe everything visible on this screen in detail",
                mode="describe",
            )
            return

        if self._matches_any(candidates, ["read that", "what does it say"]):
            await self._answer_visual_prompt(
                event=event,
                prompt="Read all text visible on this screen",
                mode="read",
            )
            return

        find_query = self._extract_query(candidates, [r"find\s+(.+?)\s+on\s+screen", r"where is\s+(.+)"])
        if find_query:
            await self._answer_visual_prompt(
                event=event,
                prompt=f"Where is {find_query} on this screen? Give pixel coordinates if possible",
                mode="find",
                query=find_query,
            )
            return

        click_query = self._extract_query(candidates, [r"click\s+(.+)"])
        if click_query:
            await self._click_query_target(click_query, event)
            return

    @staticmethod
    def _matches_any(candidates: List[str], phrases: List[str]) -> bool:
        return any(any(phrase in candidate for phrase in phrases) for candidate in candidates if candidate)

    @staticmethod
    def _extract_query(candidates: List[str], patterns: List[str]) -> Optional[str]:
        for candidate in candidates:
            if not candidate:
                continue
            for pattern in patterns:
                match = re.search(pattern, candidate)
                if match:
                    query = match.group(1).strip().strip("?.!,")
                    if query:
                        return query
        return None

    async def _answer_visual_prompt(
        self,
        event: IntentRecognizedEvent,
        prompt: str,
        mode: str = "describe",
        query: Optional[str] = None,
    ) -> None:
        try:
            await self.capture_full_screen()
            await self._emit(
                HUDUpdateEvent(
                    image_path=str(SCREENSHOT_PATH),
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )
            if self._use_gemini:
                response_text = await self._ask_gemini(prompt)
            else:
                response_text = await self._answer_with_local_ocr(
                    mode=mode,
                    query=query,
                    gemini_exc=RuntimeError("Cloud vision disabled in config"),
                )
        except Exception as gemini_exc:
            response_text = await self._answer_with_local_ocr(
                mode=mode,
                query=query,
                gemini_exc=gemini_exc,
            )

        await self._emit(
            VoiceOutputEvent(
                text=response_text,
                source=self._name,
                correlation_id=event.correlation_id or event.event_id,
            )
        )

    async def _click_query_target(self, query: str, event: IntentRecognizedEvent) -> None:
        if not PYAUTOGUI_AVAILABLE or pyautogui is None:
            await self._emit(
                VoiceOutputEvent(
                    text="Click actions are unavailable because pyautogui is not installed.",
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )
            return

        try:
            await self.capture_full_screen()
            await self._emit(
                HUDUpdateEvent(
                    image_path=str(SCREENSHOT_PATH),
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )
            prompt = (
                f"Find '{query}' on this screen and provide pixel coordinates. "
                "Return strict JSON with keys x, y, confidence. "
                "Confidence must be a number between 0 and 1."
            )
            if self._use_gemini:
                response_text = await self._ask_gemini(prompt)
                parsed = self._parse_coordinates(response_text)
            else:
                raise RuntimeError("Cloud vision disabled in config")
        except Exception as gemini_exc:
            try:
                parsed = await self._find_query_coordinates_via_ocr(query)
                if not parsed.has_coordinates:
                    message = (
                        "I couldn't complete that click with cloud vision, and OCR fallback could "
                        f"not find '{query}'. Error: {gemini_exc}"
                    )
                elif parsed.confidence < MIN_CLICK_CONFIDENCE:
                    message = (
                        f"OCR found possible coordinates ({parsed.x}, {parsed.y}) for {query}, but "
                        f"confidence {parsed.confidence:.2f} is below 0.70, so I did not click."
                    )
                else:
                    clicked = await self._safe_click(parsed.x, parsed.y)
                    if clicked:
                        message = (
                            f"Cloud vision was unavailable, so I used local OCR and clicked {query} at "
                            f"({parsed.x}, {parsed.y}) with confidence {parsed.confidence:.2f}."
                        )
                    else:
                        message = "I found the target with OCR but the click action failed safely."
            except Exception as ocr_exc:
                message = f"I couldn't complete that click action: {ocr_exc}"
        else:
            if not parsed.has_coordinates:
                message = (
                    "I could not find clickable coordinates for that target. "
                    f"Model response: {response_text}"
                )
            elif parsed.confidence < MIN_CLICK_CONFIDENCE:
                message = (
                    f"I found possible coordinates ({parsed.x}, {parsed.y}) but confidence "
                    f"{parsed.confidence:.2f} is below 0.70, so I did not click."
                )
            else:
                clicked = await self._safe_click(parsed.x, parsed.y)
                if clicked:
                    message = (
                        f"Clicked {query} at ({parsed.x}, {parsed.y}) with "
                        f"confidence {parsed.confidence:.2f}."
                    )
                else:
                    message = "I found the target but the click action failed safely."

        await self._emit(
            VoiceOutputEvent(
                text=message,
                source=self._name,
                correlation_id=event.correlation_id or event.event_id,
            )
        )

    async def capture_full_screen(self) -> str:
        """Capture full screen using PIL.ImageGrab.grab()."""
        return await self._capture_screen()

    async def capture_region(self, x: int, y: int, w: int, h: int) -> str:
        """Capture a region using grab(bbox=(x, y, w, h))."""
        return await self._capture_screen((x, y, w, h))

    async def _capture_screen(
        self,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> str:
        def _do_capture() -> str:
            if bbox is None:
                image = ImageGrab.grab()
            else:
                x, y, w, h = bbox
                # Supports both (x, y, right, bottom) and (x, y, width, height).
                pil_bbox = (x, y, w, h)
                if w <= x or h <= y:
                    pil_bbox = (x, y, x + w, y + h)
                image = ImageGrab.grab(bbox=pil_bbox)
            image.save(SCREENSHOT_PATH)
            return str(SCREENSHOT_PATH)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do_capture)

    async def _ask_gemini(self, prompt: str) -> str:
        if not self._gemini_enabled:
            self._initialize_gemini()
        # REST mode does not require SDK model object.
        if not self._gemini_enabled or (
            self._model is None and self._genai_client is None and not self._gemini_api_key
        ):
            raise RuntimeError(f"Gemini Vision is not configured ({self._gemini_status_reason})")

        def _run_request() -> str:
            image_bytes = SCREENSHOT_PATH.read_bytes()

            if self._model is not None and GEMINI_SDK_AVAILABLE:
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                contents = [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64,
                        }
                    }
                ]
                try:
                    response = self._model.generate_content(contents)
                    text = getattr(response, "text", "")
                    return (
                        (text or "").strip()
                        or "I analyzed the screen but received no text response."
                    )
                except Exception as exc:
                    self._raise_gemini_error_if_known(exc)
                    raise

            if self._genai_client is not None and GEMINI_GENAI_AVAILABLE and google_genai_types is not None:
                try:
                    response = self._genai_client.models.generate_content(
                        model=self._gemini_model_name,
                        contents=[
                            prompt,
                            google_genai_types.Part.from_bytes(
                                data=image_bytes,
                                mime_type="image/png",
                            ),
                        ],
                    )
                    text = getattr(response, "text", "")
                    return (
                        (text or "").strip()
                        or "I analyzed the screen but received no text response."
                    )
                except Exception as exc:
                    self._raise_gemini_error_if_known(exc)
                    raise

            # SDK unavailable or failed: use REST fallback.
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            return self._run_gemini_rest_request(prompt=prompt, image_b64=image_b64)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_request)

    def _run_gemini_rest_request(self, prompt: str, image_b64: str) -> str:
        if not self._gemini_api_key:
            raise RuntimeError("Gemini REST fallback unavailable: missing API key.")

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._gemini_model_name}:generateContent?key={parse.quote(self._gemini_api_key)}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_b64,
                            }
                        },
                    ]
                }
            ]
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        ssl_context = self._build_ssl_context()

        try:
            with request.urlopen(req, timeout=45, context=ssl_context) as resp:
                raw = resp.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            try:
                details = exc.read().decode("utf-8")
            except Exception:
                details = str(exc)
            self._raise_gemini_error_if_known(RuntimeError(details))
            raise RuntimeError(f"Gemini REST request failed: {details}") from exc
        except urlerror.URLError as exc:
            if self._is_ssl_certificate_error(exc):
                raise RuntimeError(
                    "Gemini connection failed due to TLS certificate verification. "
                    "Install/update CA certificates (certifi) or set "
                    "vision.gemini.allow_insecure_ssl=true as a temporary workaround."
                ) from exc
            self._raise_gemini_error_if_known(exc)
            raise RuntimeError(f"Gemini REST network error: {exc}") from exc
        except Exception as exc:
            self._raise_gemini_error_if_known(exc)
            raise

        try:
            data = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"Gemini REST returned invalid JSON: {exc}") from exc

        parts: List[str] = []
        for candidate in data.get("candidates", []) or []:
            content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
            for part in content.get("parts", []) or []:
                if isinstance(part, dict):
                    text = str(part.get("text", "")).strip()
                    if text:
                        parts.append(text)

        if parts:
            return "\n".join(parts).strip()

        err = data.get("error", {}) if isinstance(data, dict) else {}
        if err:
            message = str(err.get("message", "Unknown Gemini REST error"))
            self._raise_gemini_error_if_known(RuntimeError(message))
            raise RuntimeError(f"Gemini REST error: {message}")

        return "I analyzed the screen but received no text response."

    def _build_ssl_context(self) -> ssl.SSLContext:
        # Explicit CA configuration avoids macOS Python trust-store mismatches.
        if self._is_truthy(self._get_config("vision.gemini.allow_insecure_ssl")):
            self._logger.warning(
                "VisionAgent is using insecure SSL for Gemini REST requests "
                "(vision.gemini.allow_insecure_ssl=true)."
            )
            insecure_context = ssl.create_default_context()
            insecure_context.check_hostname = False
            insecure_context.verify_mode = ssl.CERT_NONE
            return insecure_context

        if CERTIFI_AVAILABLE and certifi is not None:
            try:
                return ssl.create_default_context(cafile=certifi.where())
            except Exception as exc:
                self._logger.warning(f"Could not load certifi CA bundle: {exc}")

        return ssl.create_default_context()

    @staticmethod
    def _is_ssl_certificate_error(exc: urlerror.URLError) -> bool:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            return True
        if isinstance(reason, ssl.SSLError):
            lowered = str(reason).lower()
            return "certificate_verify_failed" in lowered or "unable to get local issuer" in lowered
        lowered = str(exc).lower()
        return "certificate_verify_failed" in lowered or "unable to get local issuer" in lowered

    @staticmethod
    def _is_truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _raise_gemini_error_if_known(self, exc: Exception) -> None:
        msg = str(exc)
        lowered = msg.lower()
        if any(
            token in lowered
            for token in (
                "resource_exhausted",
                "quota exceeded",
                "too many requests",
                "\"code\": 429",
                "http error 429",
            )
        ):
            if "limit: 0" in lowered:
                raise RuntimeError(
                    "Gemini quota is currently 0 for this project/key. Set up billing for the "
                    "project in Google AI Studio, or switch to a project/key with available quota."
                ) from exc
            raise RuntimeError(
                "Gemini rate/quota limit reached. Please wait a moment and retry, or increase quota/billing."
            ) from exc
        if any(
            token in lowered
            for token in (
                "model not found",
                "is not found",
                "unknown model",
                "unsupported model",
                "not available for your account",
                "not available in your location",
            )
        ):
            raise RuntimeError(
                f"Gemini model '{self._gemini_model_name}' is unavailable for this key/project. "
                "Set `vision.gemini.model` to a supported model (for example `gemini-2.0-flash`) "
                "and retry."
            ) from exc
        if any(
            token in lowered
            for token in (
                "api has not been used",
                "service disabled",
                "api is not enabled",
                "enable it by visiting",
            )
        ):
            raise RuntimeError(
                "Gemini API is not enabled for this project. Enable the Generative Language API "
                "for the key's project, then retry."
            ) from exc
        if (
            "invalid key" in lowered
            or "api key not valid" in lowered
            or "expired api key" in lowered
        ):
            raise RuntimeError(
                "Gemini API key is invalid. Set a valid AI Studio Gemini API key in GEMINI_API_KEY."
            ) from exc
        if (
            "permission" in lowered
            or "403" in lowered
            or "unauth" in lowered
            or "denied" in lowered
            or "forbidden" in lowered
        ):
            raise RuntimeError(
                "Gemini rejected this request. Check key restrictions and confirm the key has "
                "Gemini API access for model generation."
            ) from exc

    async def _safe_click(self, x: Optional[int], y: Optional[int]) -> bool:
        if x is None or y is None or not PYAUTOGUI_AVAILABLE or pyautogui is None:
            return False

        def _do_click() -> bool:
            try:
                pyautogui.click(x=x, y=y)
                return True
            except Exception as exc:
                self._logger.warning(f"Safe click failed: {exc}")
                return False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do_click)

    async def _answer_with_local_ocr(
        self,
        mode: str,
        query: Optional[str],
        gemini_exc: Exception,
    ) -> str:
        try:
            ocr_text, words = await self._run_local_ocr()
        except Exception as ocr_exc:
            return (
                f"I couldn't analyze the screen with cloud vision ({gemini_exc}), and local OCR fallback "
                f"failed: {ocr_exc}"
            )

        if mode == "find" and query:
            match = self._find_best_ocr_match(query, words)
            if match is None:
                return (
                    f"Cloud vision is unavailable ({gemini_exc}). I used local OCR but couldn't find "
                    f"'{query}' on screen."
                )
            return (
                f"Cloud vision is unavailable ({gemini_exc}). Local OCR found '{query}' near "
                f"pixel ({match.x}, {match.y}) with confidence {match.confidence:.2f}."
            )

        cleaned = self._clean_ocr_text(ocr_text)
        if not cleaned:
            return (
                f"Cloud vision is unavailable ({gemini_exc}). Local OCR ran, but no readable text "
                "was detected on screen."
            )

        if mode == "read":
            return (
                "Cloud vision is unavailable, so I used local OCR. "
                f"Here is the text I can read:\n{cleaned}"
            )

        # For describe/general, provide OCR text as best-effort local fallback.
        return (
            "Cloud vision is unavailable, so I used local OCR. "
            f"Visible text on screen:\n{cleaned}"
        )

    async def _find_query_coordinates_via_ocr(self, query: str) -> CoordinateResult:
        try:
            _, words = await self._run_local_ocr()
        except Exception as exc:
            self._logger.warning(f"OCR click fallback failed: {exc}")
            return CoordinateResult(x=None, y=None, confidence=0.0, raw=str(exc))

        match = self._find_best_ocr_match(query, words)
        if match is None:
            return CoordinateResult(x=None, y=None, confidence=0.0, raw=f"No OCR match for {query}")

        return CoordinateResult(
            x=match.x,
            y=match.y,
            confidence=match.confidence,
            raw=f"OCR match '{match.text}'",
        )

    async def _run_local_ocr(self) -> Tuple[str, List[OCRWord]]:
        if not self._local_ocr_enabled:
            raise RuntimeError("Local OCR is disabled in config.")
        if not OCR_AVAILABLE or pytesseract is None:
            raise RuntimeError(
                "pytesseract is not installed. Install with: pip install pytesseract"
            )
        if not SCREENSHOT_PATH.exists():
            raise RuntimeError("No screenshot available for OCR.")

        def _scan() -> Tuple[str, List[OCRWord]]:
            image = Image.open(SCREENSHOT_PATH)

            text = (pytesseract.image_to_string(image) or "").strip()
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            words: List[OCRWord] = []
            tokens = data.get("text", []) or []
            count = len(tokens)
            for idx in range(count):
                token = str(tokens[idx] or "").strip()
                if not token:
                    continue
                conf_raw = str((data.get("conf", []) or ["-1"] * count)[idx]).strip()
                try:
                    conf_num = float(conf_raw)
                except Exception:
                    conf_num = -1.0
                if conf_num < 0:
                    continue

                left = int((data.get("left", []) or [0] * count)[idx])
                top = int((data.get("top", []) or [0] * count)[idx])
                width = int((data.get("width", []) or [0] * count)[idx])
                height = int((data.get("height", []) or [0] * count)[idx])

                words.append(
                    OCRWord(
                        text=token,
                        x=left + max(0, width // 2),
                        y=top + max(0, height // 2),
                        confidence=max(0.0, min(1.0, conf_num / 100.0)),
                    )
                )
            return text, words

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _scan)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

    def _find_best_ocr_match(self, query: str, words: List[OCRWord]) -> Optional[OCRWord]:
        normalized_query = self._normalize_text(query)
        if not normalized_query:
            return None

        best: Optional[OCRWord] = None
        best_score = 0.0
        for word in words:
            normalized_word = self._normalize_text(word.text)
            if not normalized_word:
                continue

            if normalized_query in normalized_word or normalized_word in normalized_query:
                text_score = 1.0
            else:
                text_score = difflib.SequenceMatcher(
                    None, normalized_query, normalized_word
                ).ratio()

            score = (0.45 * text_score) + (0.55 * word.confidence)
            if score > best_score:
                best_score = score
                best = OCRWord(
                    text=word.text,
                    x=word.x,
                    y=word.y,
                    confidence=max(0.0, min(1.0, score)),
                )

        if best is None or best.confidence < 0.45:
            return None
        return best

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        cleaned = re.sub(r"[ \t]+\n", "\n", (text or ""))
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if len(cleaned) > MAX_OCR_RESPONSE_CHARS:
            return f"{cleaned[:MAX_OCR_RESPONSE_CHARS].rstrip()}..."
        return cleaned

    @staticmethod
    def _parse_coordinates(response_text: str) -> CoordinateResult:
        text = (response_text or "").strip()
        if not text:
            return CoordinateResult(x=None, y=None, confidence=0.0, raw=response_text)

        # Try direct JSON first.
        json_candidate = text
        if "{" in text and "}" in text:
            json_candidate = text[text.find("{") : text.rfind("}") + 1]

        try:
            payload = json.loads(json_candidate)
            x = int(payload.get("x")) if payload.get("x") is not None else None
            y = int(payload.get("y")) if payload.get("y") is not None else None
            confidence = float(payload.get("confidence", 0.0))
            return CoordinateResult(x=x, y=y, confidence=confidence, raw=text)
        except Exception:
            pass

        coord_match = re.search(r"\(?\s*(\d{1,5})\s*,\s*(\d{1,5})\s*\)?", text)
        x = int(coord_match.group(1)) if coord_match else None
        y = int(coord_match.group(2)) if coord_match else None

        conf_match = re.search(r"confidence[^0-9]*(1(?:\.0+)?|0?\.\d+|\d{1,3}%?)", text, flags=re.IGNORECASE)
        confidence = 0.0
        if conf_match:
            token = conf_match.group(1)
            if token.endswith("%"):
                confidence = float(token.rstrip("%")) / 100.0
            else:
                confidence = float(token)
                if confidence > 1.0:
                    confidence = confidence / 100.0

        return CoordinateResult(x=x, y=y, confidence=confidence, raw=text)
