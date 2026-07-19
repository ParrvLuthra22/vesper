"""
Voice Agent Module - Speech recognition and synthesis for JARVIS.

This agent handles all voice-related functionality:
    - Wake word detection using Vosk (FREE, offline)
    - Speech-to-text using whisper.cpp (FREE, offline, high accuracy)
    - Text-to-speech using Kokoro ONNX (offline neural voice)
    - Non-blocking audio processing in background threads

Architecture:
    - VoiceAgent: Main agent coordinating all voice components
    - MicrophoneStream: Captures audio from microphone
    - VoskWakeWordDetector: Listens for "Hey Jarvis" 
    - WhisperTranscriber: Converts speech to text
    - KokoroTTS: Speaks responses with neural voice synthesis

Event Flow:
    1. Listen for wake word continuously
    2. On wake word detection, emit WakeWordDetectedEvent
    3. Record user speech until silence
    4. Transcribe with whisper.cpp
    5. Emit VoiceInputEvent with text + confidence
    6. Subscribe to VoiceOutputEvent to speak responses

All communication happens via EventBus - no direct agent calls.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import AgentCapability, BaseAgent
from bus.event_bus import EventBus
from schemas.events import (
    ListeningStateChangedEvent,
    ShutdownRequestedEvent,
    VoiceInputEvent,
    VoiceOutputEvent,
    WakeWordDetectedEvent,
)
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from kokoro_onnx import Kokoro
    import sounddevice as sd

    KOKORO_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Kokoro = None  # type: ignore[assignment]
    sd = None  # type: ignore[assignment]
    KOKORO_AVAILABLE = False


# =============================================================================
# Voice Agent States
# =============================================================================

class VoiceAgentState(Enum):
    """State machine for voice agent."""
    IDLE = auto()                # Not listening
    LISTENING_WAKE_WORD = auto() # Waiting for "Hey Jarvis"
    LISTENING_COMMAND = auto()   # Recording user speech
    TRANSCRIBING = auto()        # Converting speech to text
    SPEAKING = auto()            # Playing TTS response
    ERROR = auto()               # Error state


@dataclass
class VoiceConfig:
    """Configuration for VoiceAgent loaded from settings.yaml."""
    # Wake word settings
    wake_word: str = "vesper"
    wake_word_sensitivity: float = 0.5
    
    # Vosk settings
    vosk_model_path: str = "models/vosk-model-small-en-us-0.15"
    
    # Whisper settings
    whisper_binary_path: str = "/usr/local/bin/whisper-cpp"
    whisper_model_path: str = "models/ggml-base.en.bin"
    whisper_language: str = "en"
    whisper_threads: int = 4
    
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    
    # Recording settings
    silence_threshold: int = 500
    silence_duration_ms: int = 1500
    max_recording_seconds: float = 30.0
    min_recording_seconds: float = 0.5
    buffer_seconds: float = 0.5
    
    # TTS settings
    tts_provider: str = "kokoro"
    tts_voice_id: str = "af_heart"
    tts_rate: int = 180
    tts_volume: float = 1.0
    tts_fallback_provider: str = "system"
    tts_fallback_voice: str = "Samantha"
    address_user_as_sir: bool = True


class KokoroTTS:
    """
    Kokoro ONNX text-to-speech implementation.

    Uses local model files:
      - kokoro-v0_19.onnx
      - voices.bin
    """

    def __init__(self, voice_id: str = "af_heart"):
        if not KOKORO_AVAILABLE or Kokoro is None or sd is None:
            raise RuntimeError("kokoro-onnx/sounddevice are not available")
        self.model = Kokoro("kokoro-v0_19.onnx", "voices.bin")
        self.voice = voice_id or "af_heart"

    def speak(self, text: str):
        if not text.strip():
            return
        samples, sample_rate = self.model.create(
            text,
            voice=self.voice,
            speed=1.0,
            lang="en-us",
        )
        sd.play(samples, sample_rate)
        sd.wait()


class AsyncKokoroTTS:
    """Async adapter so VoiceAgent can use Kokoro with the existing async API."""

    def __init__(self, voice_id: str = "af_heart"):
        self._engine = KokoroTTS(voice_id=voice_id)
        self._is_speaking = False
        self._lock = threading.Lock()
        self._validate_output_device()

    @staticmethod
    def _validate_output_device() -> None:
        """Fail fast if there is no usable output device for sounddevice."""
        if not KOKORO_AVAILABLE or sd is None:
            raise RuntimeError("sounddevice is unavailable")
        try:
            sd.query_devices(kind="output")
        except Exception as exc:
            raise RuntimeError(f"No output audio device available: {exc}") from exc

    async def speak(self, text: str, voice: str = "", rate: float = 1.0) -> None:
        del rate  # Kokoro speed is fixed in KokoroTTS for now.
        if voice and voice.lower() != "default":
            self._engine.voice = voice

        with self._lock:
            self._is_speaking = True

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._engine.speak, text)
        finally:
            with self._lock:
                self._is_speaking = False

    async def stop(self) -> None:
        if KOKORO_AVAILABLE and sd is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, sd.stop)
        with self._lock:
            self._is_speaking = False

    async def shutdown(self) -> None:
        await self.stop()

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking


# =============================================================================
# Voice Agent
# =============================================================================

class VoiceAgent(BaseAgent):
    """
    Voice Agent for JARVIS - handles all speech I/O.
    
    This agent:
    - Continuously listens for the wake word in a background thread
    - Records user speech after wake word detection
    - Transcribes speech using whisper.cpp
    - Emits USER_SPOKE events with transcription + confidence
    - Subscribes to SPEAK events to vocalize responses
    - Handles interruption when new wake word is detected
    
    All operations are non-blocking and use the EventBus.
    
    Usage:
        agent = VoiceAgent(event_bus)
        await agent.start()
        # Agent now listening for wake word...
        # When user says "Hey Jarvis, what time is it?"
        # -> Emits VoiceInputEvent(text="what time is it", confidence=0.95)
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name="VoiceAgent",
            event_bus=event_bus,
            config=config,
        )
        
        # Load configuration (also stored separately for voice-specific parsing)
        self._voice_config = self._load_config(config)
        
        # State
        self._voice_state = VoiceAgentState.IDLE
        self._state_lock = threading.Lock()
        
        # Components (initialized in start())
        self._mic_stream = None
        self._wake_word_detector = None
        self._transcriber = None
        self._tts = None
        
        # Audio buffer and recorder
        self._audio_buffer = None
        self._recorder = None
        
        # Background tasks
        self._listen_task: Optional[asyncio.Task] = None
        self._process_task: Optional[asyncio.Task] = None
        
        # Queues for thread-safe communication
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._command_audio: Optional[bytes] = None
        
        # Flags
        self._wake_word_detected = asyncio.Event()
        self._recording_complete = asyncio.Event()
        self._shutdown_event = threading.Event()
        self._keyboard_fallback_logged = False
        self._last_mic_retry = 0.0
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> VoiceConfig:
        """Load configuration from dict (from settings.yaml)."""
        if not config:
            return VoiceConfig()
        
        voice_config = config.get("voice", {})
        vosk_config = voice_config.get("vosk", {})
        whisper_config = voice_config.get("whisper", {})
        recognition_config = voice_config.get("recognition", {})
        synthesis_config = voice_config.get("synthesis", {})
        tts_config = voice_config.get("tts", {})

        tts_provider = str(
            tts_config.get("engine", synthesis_config.get("provider", "kokoro"))
        )
        tts_voice_id = str(
            tts_config.get("voice_id", synthesis_config.get("voice", "af_heart"))
        )
        
        return VoiceConfig(
            # Wake word
            wake_word=voice_config.get("wake_word", "vesper"),
            wake_word_sensitivity=voice_config.get("wake_word_sensitivity", 0.5),
            
            # Vosk
            vosk_model_path=vosk_config.get("model_path", "models/vosk-model-small-en-us-0.15"),
            
            # Whisper
            whisper_binary_path=whisper_config.get("binary_path", "/usr/local/bin/whisper-cpp"),
            whisper_model_path=whisper_config.get("model_path", "models/ggml-base.en.bin"),
            whisper_language=whisper_config.get("language", "en"),
            whisper_threads=whisper_config.get("threads", 4),
            
            # Audio
            sample_rate=recognition_config.get("sample_rate", 16000),
            channels=recognition_config.get("channels", 1),
            
            # Recording
            silence_threshold=recognition_config.get("silence_threshold", 500),
            silence_duration_ms=recognition_config.get("silence_duration_ms", 1500),
            max_recording_seconds=recognition_config.get("max_recording_seconds", 30.0),
            min_recording_seconds=recognition_config.get("min_recording_seconds", 0.5),
            buffer_seconds=recognition_config.get("buffer_seconds", 0.5),
            
            # TTS
            tts_provider=tts_provider,
            tts_voice_id=tts_voice_id,
            tts_rate=synthesis_config.get("rate", 180),
            tts_volume=synthesis_config.get("volume", 1.0),
            tts_fallback_provider=synthesis_config.get("fallback_provider", "system"),
            tts_fallback_voice=synthesis_config.get("fallback_voice", "Samantha"),
            address_user_as_sir=bool(voice_config.get("address_user_as_sir", True)),
        )
    
    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """Return the capabilities this agent provides."""
        return [
            AgentCapability(
                name="voice_input",
                description="Captures voice input via microphone and transcribes to text",
                input_events=[],
                output_events=["VoiceInputEvent", "WakeWordDetectedEvent"],
            ),
            AgentCapability(
                name="voice_output",
                description="Speaks text responses using text-to-speech",
                input_events=["VoiceOutputEvent"],
                output_events=["ListeningStateChangedEvent"],
            ),
        ]
    
    # =========================================================================
    # Agent Lifecycle
    # =========================================================================
    
    async def _setup(self) -> None:
        """Initialize voice components and start listening."""
        self._logger.info(f"Voice agent initializing with wake word: '{self._voice_config.wake_word}'")
        
        # Subscribe via BaseAgent bookkeeping to avoid duplicate handlers on restart.
        self._subscribe(VoiceOutputEvent, self._handle_voice_output)
        
        # Initialize components
        await self._initialize_components()
        
        # Start background listening
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._process_task = asyncio.create_task(self._process_loop())
        
        self._set_voice_state(VoiceAgentState.LISTENING_WAKE_WORD)
        self._logger.info("Voice agent started - listening for wake word")
    
    async def _teardown(self) -> None:
        """Stop all voice components."""
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._listen_task, self._process_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop components
        await self._shutdown_components()
        
        self._logger.info("Voice agent shutdown complete")
    
    # =========================================================================
    # Component Initialization
    # =========================================================================
    
    async def _initialize_components(self) -> None:
        """Initialize all voice components."""
        # Import here to handle missing dependencies gracefully
        from utils.audio import AudioBuffer, AudioConfig, AudioRecorder, MicrophoneStream
        
        # Initialize audio buffer
        audio_config = AudioConfig(
            sample_rate=self._voice_config.sample_rate,
            channels=self._voice_config.channels,
            silence_threshold=self._voice_config.silence_threshold,
            silence_duration_ms=self._voice_config.silence_duration_ms,
            max_recording_seconds=self._voice_config.max_recording_seconds,
            min_recording_seconds=self._voice_config.min_recording_seconds,
        )
        
        self._audio_buffer = AudioBuffer(
            max_seconds=self._voice_config.buffer_seconds,
            sample_rate=self._voice_config.sample_rate,
        )
        
        self._recorder = AudioRecorder(config=audio_config)
        
        # Initialize wake word detector (Vosk)
        await self._initialize_wake_word_detector()
        
        # Initialize transcriber (whisper.cpp)
        await self._initialize_transcriber()
        
        # Initialize TTS
        await self._initialize_tts()
        
        # Initialize microphone
        await self._initialize_microphone()
    
    async def _initialize_wake_word_detector(self) -> None:
        """Initialize Vosk wake word detector."""
        try:
            from utils.stt import VoskWakeWordDetector
            
            self._wake_word_detector = VoskWakeWordDetector(
                model_path=self._voice_config.vosk_model_path,
                wake_word=self._voice_config.wake_word,
                sample_rate=self._voice_config.sample_rate,
                sensitivity=self._voice_config.wake_word_sensitivity,
                on_wake_word=self._on_wake_word_callback,
            )
            
            if not self._wake_word_detector.initialize():
                self._logger.warning("Vosk wake word detector not available - using fallback mode")
                self._wake_word_detector = None
        except ImportError as e:
            self._logger.warning(f"Vosk not available: {e}")
            self._wake_word_detector = None
    
    async def _initialize_transcriber(self) -> None:
        """Initialize whisper.cpp transcriber."""
        try:
            from utils.stt import WhisperTranscriber, SpeechRecognitionTranscriber
            
            # Try whisper.cpp first
            self._transcriber = WhisperTranscriber(
                binary_path=self._voice_config.whisper_binary_path,
                model_path=self._voice_config.whisper_model_path,
                language=self._voice_config.whisper_language,
                threads=self._voice_config.whisper_threads,
            )
            
            if not self._transcriber.initialize():
                self._logger.warning("whisper.cpp not available - trying SpeechRecognition fallback")
                
                # Try SpeechRecognition as fallback
                fallback = SpeechRecognitionTranscriber()
                if fallback.initialize():
                    self._transcriber = fallback
                else:
                    self._logger.error("No speech recognition available!")
                    self._transcriber = None
        except ImportError as e:
            self._logger.error(f"Speech recognition imports failed: {e}")
            self._transcriber = None
    
    async def _initialize_tts(self) -> None:
        """Initialize text-to-speech."""
        provider = (self._voice_config.tts_provider or "kokoro").lower()

        if provider == "kokoro":
            try:
                self._tts = AsyncKokoroTTS(voice_id=self._voice_config.tts_voice_id)
                self._logger.info(
                    f"TTS initialized with provider: kokoro ({self._voice_config.tts_voice_id})"
                )
                return
            except Exception as e:
                self._logger.warning(
                    f"Kokoro TTS failed to initialize ({e}); falling back to pyttsx3"
                )
                provider = "pyttsx3"

        try:
            from utils.tts import create_tts_provider
            
            self._tts = await create_tts_provider(
                provider=provider,
                voice=self._voice_config.tts_voice_id,
                rate=self._voice_config.tts_rate,
                fallback=True,
            )
            self._logger.info(f"TTS initialized with provider: {provider}")
        except Exception as e:
            self._logger.error(f"Failed to initialize TTS: {e}")
            
            # Use basic macOS TTS as last resort
            try:
                from utils.tts import MacOSSystemTTS
                self._tts = MacOSSystemTTS(
                    voice=self._voice_config.tts_fallback_voice,
                    rate=self._voice_config.tts_rate,
                )
                await self._tts.initialize()
                self._logger.info("Using macOS system TTS fallback")
            except Exception as e2:
                self._logger.error(f"TTS completely unavailable: {e2}")
                self._tts = None

    async def _switch_to_fallback_tts(self, reason: str) -> bool:
        """
        Switch to fallback TTS provider and keep assistant responsive.

        Returns True when fallback provider is ready.
        """
        self._logger.warning(f"Switching TTS to fallback provider: {reason}")
        try:
            from utils.tts import create_tts_provider

            if self._tts:
                try:
                    await self._tts.shutdown()
                except Exception:
                    pass

            # Prefer pyttsx3 for compatibility with user-requested fallback.
            self._tts = await create_tts_provider(
                provider="pyttsx3",
                voice=self._voice_config.tts_fallback_voice,
                rate=self._voice_config.tts_rate,
                fallback=True,
            )
            self._logger.info("Fallback TTS initialized (pyttsx3/system chain)")
            return True
        except Exception as exc:
            self._logger.error(f"Fallback TTS initialization failed: {exc}")
            self._tts = None
            return False
    
    async def _initialize_microphone(self) -> None:
        """Initialize microphone capture."""
        try:
            from utils.audio import MicrophoneStream
            
            self._mic_stream = MicrophoneStream(
                callback=self._on_audio_chunk,
                sample_rate=self._voice_config.sample_rate,
                channels=self._voice_config.channels,
                chunk_size=self._voice_config.chunk_size,
            )
            
            if not self._mic_stream.start():
                self._logger.error(f"Microphone error: {self._mic_stream.error}")
                try:
                    devices = self._mic_stream.list_input_devices()
                    if devices:
                        self._logger.warning(
                            "Available input devices: "
                            + ", ".join(
                                f"{d['index']}:{d['name']}"
                                for d in devices
                            )
                        )
                    else:
                        self._logger.warning("No input devices detected by PyAudio")
                except Exception as e:
                    self._logger.warning(f"Failed to enumerate input devices: {e}")
                self._mic_stream = None
            else:
                self._logger.info("Microphone initialized")
        except Exception as e:
            self._logger.error(f"Failed to initialize microphone: {e}")
            self._mic_stream = None
    
    async def _shutdown_components(self) -> None:
        """Shutdown all voice components."""
        if self._mic_stream:
            self._mic_stream.stop()
        
        if self._wake_word_detector:
            self._wake_word_detector.shutdown()
        
        if self._tts:
            await self._tts.shutdown()
    
    # =========================================================================
    # Audio Processing Callbacks (called from audio thread)
    # =========================================================================
    
    def _on_audio_chunk(self, audio_data: bytes) -> None:
        """
        Handle audio chunk from microphone.
        
        Called from PyAudio thread - must be fast and non-blocking.
        """
        if self._shutdown_event.is_set():
            return
        
        # Add to rolling buffer
        if self._audio_buffer:
            self._audio_buffer.add_chunk(audio_data)
        
        # Process based on current state
        with self._state_lock:
            state = self._voice_state
        
        if state == VoiceAgentState.LISTENING_WAKE_WORD:
            # Check for wake word
            if self._wake_word_detector and self._wake_word_detector.is_listening:
                self._wake_word_detector.process_audio(audio_data)
        
        elif state == VoiceAgentState.LISTENING_COMMAND:
            # Record user speech
            if self._recorder:
                self._recorder.add_chunk(audio_data)
                
                # Check if recording complete (silence detected)
                if self._recorder.is_complete:
                    self._command_audio = self._recorder.stop_recording()
                    self._recording_complete.set()
    
    def _on_wake_word_callback(self) -> None:
        """
        Callback when wake word is detected.
        
        Called from audio thread via Vosk.
        """
        self._logger.info(f"Wake word '{self._voice_config.wake_word}' detected!")
        
        # Signal the async processing loop
        try:
            # Use thread-safe method to signal async code
            asyncio.get_event_loop().call_soon_threadsafe(
                self._wake_word_detected.set
            )
        except RuntimeError:
            # No running event loop - set directly
            self._wake_word_detected.set()
    
    # =========================================================================
    # Main Processing Loops
    # =========================================================================
    
    async def _listen_loop(self) -> None:
        """
        Main listening loop - coordinates wake word and recording.
        
        This runs continuously while the agent is active.
        """
        self._logger.info("Starting listen loop...")
        
        # Track last emitted state to avoid log spam
        last_emitted_mode: Optional[str] = None
        
        while self.is_running and not self._shutdown_event.is_set():
            try:
                # If microphone unavailable, use keyboard fallback
                if not self._mic_stream:
                    await self._recover_microphone()
                    continue

                # If no wake word detector, use speech-activated mode
                if not self._wake_word_detector:
                    await self._listen_without_wake_word()
                    continue
                
                # Wait for wake word detection
                self._wake_word_detected.clear()
                self._set_voice_state(VoiceAgentState.LISTENING_WAKE_WORD)
                
                # Emit listening state only on state change (avoid log spam)
                if last_emitted_mode != "wake_word":
                    await self._event_bus.emit(ListeningStateChangedEvent(
                        is_listening=True,
                        listening_mode="wake_word",
                        source="VoiceAgent",
                    ))
                    last_emitted_mode = "wake_word"
                
                # Wait for wake word (with timeout for shutdown check)
                try:
                    await asyncio.wait_for(
                        self._wake_word_detected.wait(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue  # Check shutdown flag and retry
                
                # Wake word detected - reset state tracking for next cycle
                last_emitted_mode = None
                
                # Wake word detected!
                await self._handle_wake_word_detected()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in listen loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Back off on error

    async def _recover_microphone(self) -> None:
        """
        Keep trying microphone recovery so voice capture survives tab/app switches.

        Falls back to keyboard input only when stdin is interactive.
        """
        now = time.monotonic()
        if now - self._last_mic_retry >= 3.0:
            self._last_mic_retry = now
            self._logger.warning("Microphone unavailable, retrying audio input backend")
            await self._initialize_microphone()
            if self._mic_stream:
                self._logger.info("Microphone recovered")
                self._keyboard_fallback_logged = False
                return
        await self._keyboard_input_fallback()

    async def _listen_without_wake_word(self) -> None:
        """Listen continuously using VAD when wake word detector is unavailable."""
        if not self._recorder:
            await self._keyboard_input_fallback()
            return

        # Emit listening state only once per cycle
        await self._event_bus.emit(ListeningStateChangedEvent(
            is_listening=True,
            listening_mode="continuous",
            source="VoiceAgent",
        ))

        self._set_voice_state(VoiceAgentState.LISTENING_COMMAND)
        self._recording_complete.clear()

        # Start recording (uses VAD internally)
        self._recorder.start_recording()
        if self._audio_buffer:
            pre_audio = self._audio_buffer.get_last_seconds(self._voice_config.buffer_seconds)
            if pre_audio:
                self._recorder.add_chunk(pre_audio)

        self._logger.info("Listening for speech (no wake word detector)")

        try:
            await asyncio.wait_for(
                self._recording_complete.wait(),
                timeout=self._voice_config.max_recording_seconds + 1.0,
            )
        except asyncio.TimeoutError:
            if self._recorder:
                self._command_audio = self._recorder.stop_recording()

        if self._command_audio and len(self._command_audio) > 0 and self._recorder.has_speech:
            await self._process_command_audio(self._command_audio)
        else:
            self._logger.warning("No speech detected in continuous mode")

        self._command_audio = None
    
    async def _handle_wake_word_detected(self) -> None:
        """Handle wake word detection - start recording user speech."""
        # Emit wake word event
        await self._event_bus.emit(WakeWordDetectedEvent(
            wake_word=self._voice_config.wake_word,
            source="VoiceAgent",
        ))
        
        # Stop any current speech (interruption)
        if self._tts and self._tts.is_speaking:
            await self._tts.stop()
        
        # Play acknowledgment sound (optional)
        # await self._play_ding()
        
        # Disable wake word detection during command listening
        if self._wake_word_detector:
            self._wake_word_detector.disable()
        
        # Start recording
        self._set_voice_state(VoiceAgentState.LISTENING_COMMAND)
        self._recording_complete.clear()
        
        if self._recorder:
            # Include pre-wake-word audio buffer
            self._recorder.start_recording()
            if self._audio_buffer:
                pre_audio = self._audio_buffer.get_last_seconds(self._voice_config.buffer_seconds)
                if pre_audio:
                    self._recorder.add_chunk(pre_audio)
        
        # Emit state change
        await self._event_bus.emit(ListeningStateChangedEvent(
            is_listening=True,
            listening_mode="command",
            source="VoiceAgent",
        ))
        
        self._logger.info("Recording user command...")
        
        # Wait for recording to complete
        try:
            await asyncio.wait_for(
                self._recording_complete.wait(),
                timeout=self._voice_config.max_recording_seconds + 1.0,
            )
        except asyncio.TimeoutError:
            self._logger.warning("Recording timed out")
            if self._recorder:
                self._command_audio = self._recorder.stop_recording()
        
        # Process the recorded audio
        if self._command_audio and len(self._command_audio) > 0:
            await self._process_command_audio(self._command_audio)
        else:
            self._logger.warning("No audio recorded")
        
        # Re-enable wake word detection
        if self._wake_word_detector:
            self._wake_word_detector.enable()
        
        # Clear for next command
        self._command_audio = None
    
    async def _process_command_audio(self, audio_data: bytes) -> None:
        """Process recorded audio - transcribe and emit event."""
        self._set_voice_state(VoiceAgentState.TRANSCRIBING)
        
        # Check minimum duration
        bytes_per_second = self._voice_config.sample_rate * 2 * self._voice_config.channels
        duration = len(audio_data) / bytes_per_second
        
        if duration < self._voice_config.min_recording_seconds:
            self._logger.warning(f"Recording too short ({duration:.2f}s), ignoring")
            return
        
        self._logger.info(f"Transcribing {duration:.2f}s of audio...")
        
        # Transcribe
        if not self._transcriber:
            self._logger.error("No transcriber available")
            return
        
        result = await self._transcriber.transcribe(
            audio_data,
            sample_rate=self._voice_config.sample_rate,
        )
        
        if not result.is_valid:
            self._logger.warning(f"Transcription failed: {result.error}")
            return
        
        text = result.text.strip()
        confidence = result.confidence
        
        if not text:
            self._logger.warning("Empty transcription")
            return
        
        self._logger.info(f"Transcribed: '{text}' (confidence: {confidence:.2f})")
        
        # Emit VoiceInputEvent (USER_SPOKE)
        await self._event_bus.emit(VoiceInputEvent(
            text=text,
            confidence=confidence,
            language=self._voice_config.whisper_language,
            is_wake_word=False,
            audio_duration=duration,
            source="VoiceAgent",
        ))
    
    async def _process_loop(self) -> None:
        """
        Secondary processing loop for async operations.
        
        Handles operations that need to run in the async context.
        """
        while self.is_running and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
    
    async def _keyboard_input_fallback(self) -> None:
        """
        Fallback mode when microphone/Vosk not available.
        
        Allows testing via stdin keyboard input.
        """
        if not self._keyboard_fallback_logged:
            self._logger.warning("Running in keyboard input mode (no microphone)")
            self._keyboard_fallback_logged = True
        
        import sys
        import select

        if not sys.stdin or not sys.stdin.isatty():
            await asyncio.sleep(0.5)
            return
        
        # Check if stdin has input (non-blocking)
        if sys.stdin in select.select([sys.stdin], [], [], 0.5)[0]:
            line = sys.stdin.readline().strip()
            if line:
                self._logger.info(f"Keyboard input: '{line}'")
                
                # Emit as voice input
                await self._event_bus.emit(VoiceInputEvent(
                    text=line,
                    confidence=1.0,
                    language="en",
                    is_wake_word=False,
                    audio_duration=0.0,
                    source="VoiceAgent:keyboard",
                ))
        else:
            await asyncio.sleep(0.5)
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    async def _handle_voice_output(self, event: VoiceOutputEvent) -> None:
        """Handle request to speak text."""
        self._logger.info(f"Received VoiceOutputEvent: '{event.text[:50]}...' from {event.source}")
        
        if not event.text:
            return

        spoken_text = self._format_for_owner(event.text)
        
        # Set speaking state
        self._set_voice_state(VoiceAgentState.SPEAKING)
        
        # Disable wake word during speech (prevent self-triggering)
        if self._wake_word_detector:
            self._wake_word_detector.disable()
        
        try:
            if self._tts:
                try:
                    await self._tts.speak(
                        text=spoken_text,
                        voice=event.voice_id if event.voice_id != "default" else "",
                        rate=event.speed,
                    )
                except Exception as exc:
                    self._logger.warning(f"Primary TTS playback failed: {exc}")
                    # Runtime fallback for output-device errors and other playback faults.
                    switched = await self._switch_to_fallback_tts(str(exc))
                    if switched and self._tts:
                        try:
                            await self._tts.speak(
                                text=spoken_text,
                                voice=event.voice_id if event.voice_id != "default" else "",
                                rate=event.speed,
                            )
                        except Exception as retry_exc:
                            self._logger.error(f"Fallback TTS playback failed: {retry_exc}")
            else:
                self._logger.warning("TTS not available")
        finally:
            # Re-enable wake word detection
            if self._wake_word_detector:
                self._wake_word_detector.enable()
            
            self._set_voice_state(VoiceAgentState.LISTENING_WAKE_WORD)

    def _format_for_owner(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned or not self._voice_config.address_user_as_sir:
            return cleaned
        lowered = cleaned.lower()
        if (
            "verification successful" in lowered
            or lowered.startswith("hello sir")
            or lowered.endswith(" sir")
            or lowered.endswith(" sir.")
        ):
            return cleaned
        if lowered.startswith(("sir,", "sir ", "good morning sir", "good evening sir")):
            return cleaned
        return f"Sir, {cleaned}"
    
    async def _handle_shutdown(self, event: ShutdownRequestedEvent) -> None:
        """Handle shutdown request."""
        self._logger.info(f"Received shutdown request: {event.reason}")
        await self.stop(event.reason)
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def _set_voice_state(self, new_state: VoiceAgentState) -> None:
        """Thread-safe state change."""
        with self._state_lock:
            old_state = self._voice_state
            self._voice_state = new_state
        
        if old_state != new_state:
            self._logger.debug(f"Voice state: {old_state.name} -> {new_state.name}")
    
    @property
    def voice_state(self) -> VoiceAgentState:
        """Get current voice state."""
        with self._state_lock:
            return self._voice_state
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    async def speak(self, text: str) -> None:
        """
        Speak text using TTS.
        
        Convenience method - prefer using VoiceOutputEvent.
        """
        if self._tts:
            await self._tts.speak(text)
    
    async def stop_speaking(self) -> None:
        """Stop current speech immediately."""
        if self._tts:
            await self._tts.stop()
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._tts.is_speaking if self._tts else False
    
    @property
    def is_listening(self) -> bool:
        """Check if listening for wake word."""
        with self._state_lock:
            return self._voice_state in (
                VoiceAgentState.LISTENING_WAKE_WORD,
                VoiceAgentState.LISTENING_COMMAND,
            )
