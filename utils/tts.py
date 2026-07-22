"""
Text-to-Speech Providers for VESPER Virtual Assistant.

This module provides:
- Pyttsx3TTS: Offline TTS using pyttsx3 (macOS native voices)
- MacOSSystemTTS: Fallback using macOS 'say' command

Both are FREE and work offline.
"""

import asyncio
import logging
import os
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from queue import Queue, Empty
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# TTS State
# =============================================================================

class TTSState(Enum):
    """State of the TTS engine."""
    IDLE = auto()
    SPEAKING = auto()
    INTERRUPTED = auto()
    ERROR = auto()


@dataclass
class TTSRequest:
    """Request to speak text."""
    text: str
    voice: str = ""
    rate: int = 180
    volume: float = 1.0
    priority: int = 0  # Higher = more important
    callback: Optional[Callable[[], None]] = None


# =============================================================================
# Base TTS Provider
# =============================================================================

class TTSProvider(ABC):
    """Abstract base class for TTS providers."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider. Returns True if successful."""
        pass
    
    @abstractmethod
    async def speak(self, text: str, voice: str = "", rate: float = 1.0) -> None:
        """Speak the given text."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop current speech."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        pass
    
    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        pass


# =============================================================================
# Pyttsx3 TTS Provider
# =============================================================================

class Pyttsx3TTS(TTSProvider):
    """
    Offline Text-to-Speech using pyttsx3.
    
    pyttsx3 uses native system TTS engines:
    - macOS: NSSpeechSynthesizer
    - Windows: SAPI5
    - Linux: espeak
    
    Features:
    - Completely offline
    - Native macOS voices (Samantha, Alex, etc.)
    - Interruptible speech
    - Non-blocking via background thread
    
    Requirements:
        pip install pyttsx3
        (On macOS, pyobjc may be needed)
    
    Usage:
        tts = Pyttsx3TTS()
        await tts.initialize()
        await tts.speak("Hello, I am VESPER")
    """
    
    def __init__(
        self,
        voice_id: str = "",
        rate: int = 180,
        volume: float = 1.0,
    ):
        """
        Initialize pyttsx3 TTS.
        
        Args:
            voice_id: Voice identifier (empty = system default)
            rate: Speech rate in words per minute
            volume: Volume level 0.0-1.0
        """
        self._default_voice = voice_id
        self._default_rate = rate
        self._default_volume = volume
        
        self._engine = None
        self._state = TTSState.IDLE
        self._initialized = False
        
        # Background thread for speech
        self._speech_thread: Optional[threading.Thread] = None
        self._speech_queue: Queue[Optional[TTSRequest]] = Queue()
        self._stop_event = threading.Event()
        self._interrupt_event = threading.Event()
        
        # Lock for state changes
        self._lock = threading.Lock()
    
    async def initialize(self) -> bool:
        """Initialize pyttsx3 engine."""
        try:
            import pyttsx3
            
            # Create engine (must be done in thread that will use it)
            # We'll initialize in the speech thread
            self._initialized = True
            
            # Start background speech thread
            self._stop_event.clear()
            self._speech_thread = threading.Thread(
                target=self._speech_loop,
                name="pyttsx3-speech",
                daemon=True,
            )
            self._speech_thread.start()
            
            logger.info("pyttsx3 TTS initialized")
            return True
            
        except ImportError:
            logger.warning("pyttsx3 not installed. Run: pip install pyttsx3")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            return False
    
    def _speech_loop(self) -> None:
        """Background thread for processing speech requests."""
        import pyttsx3
        
        try:
            # Create engine in this thread
            self._engine = pyttsx3.init()
            
            # Configure default voice
            if self._default_voice:
                try:
                    self._engine.setProperty('voice', self._default_voice)
                except Exception as e:
                    logger.warning(f"Failed to set voice '{self._default_voice}': {e}")
                    # List available voices
                    voices = self._engine.getProperty('voices')
                    logger.info(f"Available voices: {[v.id for v in voices]}")
            
            self._engine.setProperty('rate', self._default_rate)
            self._engine.setProperty('volume', self._default_volume)
            
            logger.debug("pyttsx3 engine created in speech thread")
            
        except Exception as e:
            logger.error(f"Failed to create pyttsx3 engine: {e}")
            self._state = TTSState.ERROR
            return
        
        while not self._stop_event.is_set():
            try:
                # Wait for speech request
                request = self._speech_queue.get(timeout=0.5)
                
                if request is None:
                    # Poison pill - shutdown
                    break
                
                self._speak_request(request)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Speech loop error: {e}")
        
        # Cleanup
        try:
            self._engine.stop()
        except Exception:
            pass
        
        logger.debug("Speech thread exiting")
    
    def _speak_request(self, request: TTSRequest) -> None:
        """Process a single speech request."""
        with self._lock:
            self._state = TTSState.SPEAKING
            self._interrupt_event.clear()
        
        try:
            # Apply request-specific settings
            if request.voice:
                try:
                    self._engine.setProperty('voice', request.voice)
                except Exception:
                    pass
            
            if request.rate > 0:
                self._engine.setProperty('rate', request.rate)
            
            if request.volume >= 0:
                self._engine.setProperty('volume', request.volume)
            
            logger.debug(f"Speaking: {request.text[:50]}...")
            
            # Speak (blocks until complete or interrupted)
            self._engine.say(request.text)
            self._engine.runAndWait()
            
            # Call completion callback
            if request.callback and not self._interrupt_event.is_set():
                try:
                    request.callback()
                except Exception as e:
                    logger.error(f"Speech callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Speech error: {e}")
        finally:
            with self._lock:
                if self._state == TTSState.SPEAKING:
                    self._state = TTSState.IDLE
    
    async def speak(
        self,
        text: str,
        voice: str = "",
        rate: float = 1.0,
        volume: float = -1.0,
        callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Speak text asynchronously.
        
        Args:
            text: Text to speak
            voice: Voice ID (empty = use default)
            rate: Rate multiplier (1.0 = normal)
            volume: Volume 0.0-1.0 (-1 = use default)
            callback: Called when speech completes
        """
        if not self._initialized:
            logger.warning("pyttsx3 not initialized")
            return
        
        if not text.strip():
            return
        
        # Calculate actual rate
        actual_rate = int(self._default_rate * rate) if rate > 0 else self._default_rate
        actual_volume = volume if volume >= 0 else self._default_volume
        
        request = TTSRequest(
            text=text,
            voice=voice or self._default_voice,
            rate=actual_rate,
            volume=actual_volume,
            callback=callback,
        )
        
        self._speech_queue.put(request)
        
        # Wait for speech to complete (non-blocking wait)
        while self.is_speaking:
            await asyncio.sleep(0.1)
    
    async def stop(self) -> None:
        """Stop current speech immediately."""
        with self._lock:
            if self._state == TTSState.SPEAKING:
                self._interrupt_event.set()
                self._state = TTSState.INTERRUPTED
        
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
        
        # Clear pending requests
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except Empty:
                break
        
        logger.debug("Speech stopped")
    
    async def shutdown(self) -> None:
        """Shutdown TTS engine."""
        self._stop_event.set()
        
        # Send poison pill
        self._speech_queue.put(None)
        
        # Wait for thread to finish
        if self._speech_thread and self._speech_thread.is_alive():
            self._speech_thread.join(timeout=2.0)
        
        self._initialized = False
        self._state = TTSState.IDLE
        logger.info("pyttsx3 TTS shutdown")
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        with self._lock:
            return self._state == TTSState.SPEAKING
    
    def list_voices(self) -> list:
        """List available voices."""
        if self._engine:
            return self._engine.getProperty('voices')
        return []


# =============================================================================
# macOS System TTS (Fallback)
# =============================================================================

class MacOSSystemTTS(TTSProvider):
    """
    Text-to-speech using macOS 'say' command via afplay.
    
    This is a fallback when pyttsx3 doesn't work.
    Uses file-based playback for reliable audio completion.
    """
    
    def __init__(
        self,
        voice: str = "Samantha",
        rate: int = 180,
    ):
        self._default_voice = voice
        self._default_rate = rate
        self._current_process: Optional[asyncio.subprocess.Process] = None
        self._state = TTSState.IDLE
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> bool:
        """Verify 'say' command is available."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "which", "say",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            
            if proc.returncode != 0:
                logger.error("macOS 'say' command not found")
                return False
            
            logger.info("macOS System TTS initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize macOS TTS: {e}")
            return False
    
    async def speak(
        self,
        text: str,
        voice: str = "",
        rate: float = 1.0,
    ) -> None:
        """Speak text using macOS say command with file-based playback."""
        if not text.strip():
            return
        
        voice = voice if voice and voice.lower() != "default" else self._default_voice
        actual_rate = int(self._default_rate * rate)
        
        await self.stop()
        
        async with self._lock:
            self._state = TTSState.SPEAKING
        
        temp_path = None
        try:
            # Create temp file for audio
            temp_file = tempfile.NamedTemporaryFile(suffix='.aiff', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Generate speech to file
            gen_proc = await asyncio.create_subprocess_exec(
                "say",
                "-v", voice,
                "-r", str(actual_rate),
                "-o", temp_path,
                text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await gen_proc.wait()
            
            # Play audio file
            self._current_process = await asyncio.create_subprocess_exec(
                "afplay",
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await self._current_process.wait()
            
        except Exception as e:
            logger.error(f"macOS TTS error: {e}")
        finally:
            self._current_process = None
            async with self._lock:
                self._state = TTSState.IDLE
            
            # Cleanup temp file
            if temp_path and Path(temp_path).exists():
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
    
    async def stop(self) -> None:
        """Stop current speech."""
        if self._current_process:
            try:
                self._current_process.terminate()
                await self._current_process.wait()
            except ProcessLookupError:
                pass
            self._current_process = None
        
        async with self._lock:
            self._state = TTSState.IDLE
    
    async def shutdown(self) -> None:
        """Cleanup resources."""
        await self.stop()
        logger.info("macOS System TTS shutdown")
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._state == TTSState.SPEAKING


# =============================================================================
# TTS Factory
# =============================================================================

async def create_tts_provider(
    provider: str = "pyttsx3",
    voice: str = "",
    rate: int = 180,
    fallback: bool = True,
) -> TTSProvider:
    """
    Create a TTS provider based on configuration.
    
    Args:
        provider: Provider name ('pyttsx3', 'system')
        voice: Default voice ID
        rate: Speech rate
        fallback: If True, fall back to system TTS if primary fails
    
    Returns:
        Initialized TTS provider
    """
    if provider == "pyttsx3":
        tts = Pyttsx3TTS(voice_id=voice, rate=rate)
        if await tts.initialize():
            return tts
        
        if fallback:
            logger.warning("Falling back to macOS system TTS")
            tts = MacOSSystemTTS(voice="Samantha", rate=rate)
            await tts.initialize()
            return tts
    
    # Default to system TTS
    tts = MacOSSystemTTS(voice=voice or "Samantha", rate=rate)
    await tts.initialize()
    return tts
