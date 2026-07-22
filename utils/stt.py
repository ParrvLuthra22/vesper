"""
Speech Recognition Providers for VESPER Virtual Assistant.

This module provides:
- VoskWakeWordDetector: Lightweight wake word detection using Vosk
- WhisperTranscriber: High-accuracy transcription using whisper.cpp

Both are designed to be FREE and run locally without paid APIs.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.audio import (
    AudioBuffer,
    AudioConfig,
    AudioRecorder,
    create_temp_wav_file,
    DEFAULT_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Transcription Result
# =============================================================================

@dataclass
class TranscriptionResult:
    """Result from speech transcription."""
    text: str
    confidence: float = 1.0  # 0.0 to 1.0
    language: str = "en"
    duration_seconds: float = 0.0
    words: List[Dict[str, Any]] = field(default_factory=list)
    is_partial: bool = False
    error: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if transcription is valid (has text and no error)."""
        return bool(self.text.strip()) and self.error is None
    
    def __str__(self) -> str:
        return self.text


# =============================================================================
# Wake Word Detection State
# =============================================================================

class WakeWordState(Enum):
    """State of wake word detection."""
    IDLE = auto()
    LISTENING = auto()
    DETECTED = auto()
    ERROR = auto()


# =============================================================================
# Vosk Wake Word Detector
# =============================================================================

class VoskWakeWordDetector:
    """
    Wake word detection using Vosk speech recognition.
    
    Vosk is lightweight and runs entirely locally. It continuously
    listens for the wake word and triggers a callback when detected.
    
    Requirements:
        pip install vosk
        Download model from https://alphacephei.com/vosk/models
        Recommended: vosk-model-small-en-us-0.15 (40MB)
    
    Usage:
        detector = VoskWakeWordDetector(
            model_path="models/vosk-model-small-en-us",
            wake_word="vesper",
            on_wake_word=lambda: print("Wake word detected!"),
        )
        detector.start()
        # Feed audio chunks...
        detector.process_audio(chunk)
    """
    
    def __init__(
        self,
        model_path: str,
        wake_word: str = "vesper",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        sensitivity: float = 0.5,
        on_wake_word: Optional[Callable[[], None]] = None,
        on_partial: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize Vosk wake word detector.
        
        Args:
            model_path: Path to Vosk model directory
            wake_word: Wake word to listen for (case-insensitive)
            sample_rate: Audio sample rate in Hz
            sensitivity: Detection sensitivity (0.0-1.0, lower = more strict)
            on_wake_word: Callback when wake word is detected
            on_partial: Callback for partial transcription results
        """
        self._model_path = model_path
        self._wake_word = wake_word.lower()
        self._sample_rate = sample_rate
        self._sensitivity = sensitivity
        self._on_wake_word = on_wake_word
        self._on_partial = on_partial
        
        self._model = None
        self._recognizer = None
        self._state = WakeWordState.IDLE
        self._enabled = True
        self._lock = threading.Lock()
        
        # Cooldown to prevent multiple triggers
        self._last_detection: float = 0
        self._cooldown_seconds: float = 2.0
        
        # Alternative wake words
        self._wake_word_variants = self._generate_variants(wake_word)
    
    def _generate_variants(self, wake_word: str) -> List[str]:
        """Generate common misheard variants of wake word."""
        word = wake_word.lower()
        variants = [word]
        
        # Common mishearings for "vesper"
        if word == "vesper":
            variants.extend([
                "vesper",
                "travis",
                "jervis",
                "service",
                "jarvus",
                "charvis",
                "hey vesper",
                "hey travis",
                "ok vesper",
            ])
        
        return variants
    
    def initialize(self) -> bool:
        """
        Initialize Vosk model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from vosk import Model, KaldiRecognizer
            
            # Check model path exists
            model_path = Path(self._model_path)
            if not model_path.exists():
                logger.warning(f"Vosk model not found at: {model_path}")
                logger.info("Download from: https://alphacephei.com/vosk/models")
                self._state = WakeWordState.ERROR
                return False
            
            logger.info(f"Loading Vosk model from: {model_path}")
            self._model = Model(str(model_path))
            self._recognizer = KaldiRecognizer(self._model, self._sample_rate)
            self._recognizer.SetWords(True)
            
            self._state = WakeWordState.LISTENING
            logger.info("Vosk wake word detector initialized")
            return True
            
        except ImportError:
            logger.warning("Vosk not installed. Run: pip install vosk")
            self._state = WakeWordState.ERROR
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Vosk: {e}")
            self._state = WakeWordState.ERROR
            return False
    
    def process_audio(self, audio_data: bytes) -> bool:
        """
        Process audio chunk for wake word detection.
        
        Args:
            audio_data: Raw 16-bit PCM audio bytes
        
        Returns:
            True if wake word was detected
        """
        if not self._enabled or self._recognizer is None:
            return False
        
        if self._state != WakeWordState.LISTENING:
            return False
        
        try:
            # Check cooldown
            now = time.time()
            if now - self._last_detection < self._cooldown_seconds:
                return False
            
            # Feed audio to recognizer
            if self._recognizer.AcceptWaveform(audio_data):
                # Final result
                result = json.loads(self._recognizer.Result())
                text = result.get("text", "").lower()
                
                if self._check_wake_word(text):
                    return self._trigger_wake_word()
            else:
                # Partial result
                partial = json.loads(self._recognizer.PartialResult())
                partial_text = partial.get("partial", "").lower()
                
                if self._on_partial:
                    self._on_partial(partial_text)
                
                if self._check_wake_word(partial_text):
                    # Reset recognizer to clear the wake word
                    self._recognizer.Reset()
                    return self._trigger_wake_word()
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return False
    
    def _check_wake_word(self, text: str) -> bool:
        """Check if text contains wake word or variant."""
        if not text:
            return False
        
        text_lower = text.lower()
        for variant in self._wake_word_variants:
            if variant in text_lower:
                return True
        
        return False
    
    def _trigger_wake_word(self) -> bool:
        """Handle wake word detection."""
        with self._lock:
            self._last_detection = time.time()
            self._state = WakeWordState.DETECTED
            
            logger.info(f"Wake word '{self._wake_word}' detected!")
            
            if self._on_wake_word:
                try:
                    self._on_wake_word()
                except Exception as e:
                    logger.error(f"Wake word callback error: {e}")
            
            # Return to listening after short delay
            self._state = WakeWordState.LISTENING
            return True
    
    def enable(self) -> None:
        """Enable wake word detection."""
        self._enabled = True
        self._state = WakeWordState.LISTENING
        logger.debug("Wake word detection enabled")
    
    def disable(self) -> None:
        """Disable wake word detection temporarily."""
        self._enabled = False
        self._state = WakeWordState.IDLE
        logger.debug("Wake word detection disabled")
    
    def reset(self) -> None:
        """Reset the recognizer state."""
        if self._recognizer:
            self._recognizer.Reset()
        self._state = WakeWordState.LISTENING
    
    def shutdown(self) -> None:
        """Clean up resources."""
        self._enabled = False
        self._state = WakeWordState.IDLE
        self._recognizer = None
        self._model = None
        logger.info("Vosk wake word detector shutdown")
    
    @property
    def state(self) -> WakeWordState:
        """Get current state."""
        return self._state
    
    @property
    def is_listening(self) -> bool:
        """Check if actively listening for wake word."""
        return self._state == WakeWordState.LISTENING and self._enabled


# =============================================================================
# Whisper.cpp Transcriber
# =============================================================================

class WhisperTranscriber:
    """
    Speech transcription using whisper.cpp (local binary).
    
    whisper.cpp is a C++ port of OpenAI's Whisper that runs efficiently
    on CPU without requiring Python ML frameworks.
    
    Requirements:
        - Build whisper.cpp from https://github.com/ggerganov/whisper.cpp
        - Download a model (ggml-base.en.bin recommended)
    
    Usage:
        transcriber = WhisperTranscriber(
            binary_path="/usr/local/bin/whisper-cpp",
            model_path="models/ggml-base.en.bin",
        )
        result = await transcriber.transcribe(audio_bytes)
        print(result.text)
    """
    
    def __init__(
        self,
        binary_path: str,
        model_path: str,
        language: str = "en",
        threads: int = 4,
        output_format: str = "json",
    ):
        """
        Initialize whisper.cpp transcriber.
        
        Args:
            binary_path: Path to whisper.cpp binary (main or whisper-cpp)
            model_path: Path to .bin model file
            language: Language code (e.g., 'en', 'es', 'auto')
            threads: Number of CPU threads to use
            output_format: Output format (json, txt, srt, vtt)
        """
        self._binary_path = binary_path
        self._model_path = model_path
        self._language = language
        self._threads = threads
        self._output_format = output_format
        
        self._initialized = False
        self._lock = asyncio.Lock()
    
    def initialize(self) -> bool:
        """
        Verify whisper.cpp binary and model exist.
        
        Returns:
            True if ready, False otherwise
        """
        # Check binary
        binary = Path(self._binary_path)
        if not binary.exists():
            logger.warning(f"whisper.cpp binary not found at: {binary}")
            logger.info("Build from: https://github.com/ggerganov/whisper.cpp")
            return False
        
        # Check model
        model = Path(self._model_path)
        if not model.exists():
            logger.warning(f"Whisper model not found at: {model}")
            logger.info("Download from: https://huggingface.co/ggerganov/whisper.cpp")
            return False
        
        self._initialized = True
        logger.info(f"Whisper transcriber initialized (model: {model.name})")
        return True
    
    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> TranscriptionResult:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw PCM audio bytes (16-bit, mono)
            sample_rate: Audio sample rate in Hz
        
        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        if not self._initialized:
            return TranscriptionResult(
                text="",
                error="Transcriber not initialized",
            )
        
        async with self._lock:
            return await self._transcribe_impl(audio_data, sample_rate)
    
    async def _transcribe_impl(
        self,
        audio_data: bytes,
        sample_rate: int,
    ) -> TranscriptionResult:
        """Internal transcription implementation."""
        temp_wav = None
        temp_output = None
        
        try:
            start_time = time.time()
            
            # Create temp WAV file
            temp_wav = create_temp_wav_file(audio_data, sample_rate)
            if not temp_wav:
                return TranscriptionResult(
                    text="",
                    error="Failed to create temp audio file",
                )
            
            # Output file for JSON
            temp_output = tempfile.NamedTemporaryFile(
                suffix='.json',
                prefix='whisper_out_',
                delete=False,
            ).name
            
            # Build command with optimized settings
            cmd = [
                self._binary_path,
                "-m", self._model_path,
                "-f", temp_wav,
                "-l", self._language,
                "-t", str(self._threads),
                "-oj",  # Output JSON
                "-of", temp_output.replace('.json', ''),  # Output file prefix
                "--no-prints",  # Suppress progress
                # Improve accuracy with initial prompt
                "--prompt", "Commands like: open Safari, what time is it, get system stats, open Finder, close app, play music, search the web.",
            ]
            
            logger.debug(f"Running whisper.cpp: {' '.join(cmd)}")
            
            # Run whisper.cpp
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                logger.error(f"whisper.cpp failed: {error_msg}")
                return TranscriptionResult(
                    text="",
                    error=f"Transcription failed: {error_msg}",
                )
            
            # Parse JSON output
            json_file = temp_output
            if Path(json_file).exists():
                result = self._parse_whisper_json(json_file)
            else:
                # Fallback: try to parse stdout
                result = self._parse_whisper_output(stdout.decode())
            
            duration = time.time() - start_time
            result.duration_seconds = duration
            
            logger.info(f"Transcription complete in {duration:.2f}s: '{result.text[:50]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return TranscriptionResult(
                text="",
                error=str(e),
            )
        finally:
            # Cleanup temp files
            for temp_file in [temp_wav, temp_output]:
                if temp_file and Path(temp_file).exists():
                    try:
                        os.unlink(temp_file)
                    except Exception:
                        pass
    
    def _parse_whisper_json(self, json_path: str) -> TranscriptionResult:
        """Parse whisper.cpp JSON output."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract transcription
            transcription = data.get("transcription", [])
            
            texts = []
            words = []
            total_confidence = 0.0
            
            for segment in transcription:
                text = segment.get("text", "").strip()
                if text:
                    texts.append(text)
                
                # Extract word-level info if available
                segment_words = segment.get("words", [])
                words.extend(segment_words)
                
                # Estimate confidence from probability
                prob = segment.get("avg_logprob", -1.0)
                if prob > -1.0:
                    # Convert log probability to confidence
                    import math
                    conf = math.exp(prob)
                    total_confidence += conf
            
            full_text = " ".join(texts).strip()
            avg_confidence = total_confidence / len(transcription) if transcription else 0.5
            
            return TranscriptionResult(
                text=full_text,
                confidence=min(avg_confidence, 1.0),
                words=words,
            )
            
        except Exception as e:
            logger.error(f"Failed to parse whisper JSON: {e}")
            return TranscriptionResult(text="", error=str(e))
    
    def _parse_whisper_output(self, output: str) -> TranscriptionResult:
        """Parse whisper.cpp text output as fallback."""
        # Remove timestamps like [00:00:00.000 --> 00:00:05.000]
        import re
        text = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]', '', output)
        text = text.strip()
        
        return TranscriptionResult(
            text=text,
            confidence=0.5,  # Unknown confidence for text output
        )
    
    def transcribe_sync(
        self,
        audio_data: bytes,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> TranscriptionResult:
        """
        Synchronous transcription (for non-async contexts).
        
        Args:
            audio_data: Raw PCM audio bytes
            sample_rate: Audio sample rate
        
        Returns:
            TranscriptionResult
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.transcribe(audio_data, sample_rate)
            )
        finally:
            loop.close()


# =============================================================================
# Fallback: SpeechRecognition Library Transcriber
# =============================================================================

class SpeechRecognitionTranscriber:
    """
    Fallback transcriber using the SpeechRecognition library.
    
    Uses Google's free web API (requires internet) or other backends.
    This is a fallback when whisper.cpp is not available.
    
    Note: Google's free API has usage limits.
    """
    
    def __init__(self, language: str = "en-US"):
        self._language = language
        self._recognizer = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the recognizer."""
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._initialized = True
            logger.info("SpeechRecognition fallback initialized")
            return True
        except ImportError:
            logger.warning("SpeechRecognition not installed. Run: pip install SpeechRecognition")
            return False
    
    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> TranscriptionResult:
        """Transcribe using Google's free API."""
        if not self._initialized:
            return TranscriptionResult(text="", error="Not initialized")
        
        try:
            import speech_recognition as sr
            
            # Create AudioData from raw bytes
            audio = sr.AudioData(audio_data, sample_rate, 2)
            
            # Run in thread to not block
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                lambda: self._recognizer.recognize_google(audio, language=self._language)
            )
            
            return TranscriptionResult(
                text=text,
                confidence=0.8,  # Google doesn't return confidence
            )
            
        except Exception as e:
            logger.error(f"SpeechRecognition error: {e}")
            return TranscriptionResult(text="", error=str(e))
