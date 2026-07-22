"""Audio utilities for the VESPER Virtual Assistant.

This module provides helper functions for:
- Microphone access and audio recording
- Audio format conversion
- Silence/speech detection
- Voice Activity Detection (VAD)
- Audio buffer management

All audio operations are designed to be non-blocking where possible.

Note on Python 3.13+
====================
The built-in :mod:`audioop` module was removed from the Python standard
library in Python 3.13. This module previously relied on :mod:`audioop`
for a few basic operations (RMS, volume scaling, and resampling).

To keep VESPER working on Python 3.13+, we provide small, pure-Python
fallback implementations for the subset of functionality we use. When
``audioop`` is available (e.g. on Python 3.12), it will be used; when it
is not available, the fallbacks are used instead. The fallbacks are
designed for correctness and simplicity rather than absolute speed, but
they are perfectly adequate for the relatively small audio chunks used
here.
"""

import asyncio
import io
import logging
import struct
import tempfile
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # Python 3.12 and earlier
    import audioop as _audioop  # type: ignore[attr-defined]
except ModuleNotFoundError:  # Python 3.13+ (audioop removed)
    _audioop = None
    logger.warning(
        "audioop module not available (likely Python 3.13+); "
        "falling back to pure-Python audio processing."
    )


# =============================================================================
# Small pure-Python fallbacks for audioop functionality
# =============================================================================

def _unpack_samples(audio_data: bytes, sample_width: int) -> Tuple[List[int], int]:
    """Unpack raw PCM bytes into a list of integer samples.

    Supports 8-bit, 16-bit, and 32-bit signed samples (the only formats
    we use in this project). Returns the list of samples and the maximum
    absolute value representable for the given width (used for clipping).
    """

    if not audio_data:
        return [], 0

    if sample_width == 1:
        fmt_char = "b"   # signed 8-bit
        max_abs = 127
    elif sample_width == 2:
        fmt_char = "h"   # signed 16-bit
        max_abs = 32767
    elif sample_width == 4:
        fmt_char = "i"   # signed 32-bit
        max_abs = 2147483647
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    count = len(audio_data) // sample_width
    if count == 0:
        return [], max_abs

    fmt = f"<{count}{fmt_char}"
    samples = list(struct.unpack(fmt, audio_data))
    return samples, max_abs


def _pack_samples(samples: List[int], sample_width: int, max_abs: int) -> bytes:
    """Pack integer samples back into PCM bytes, with clipping."""

    if not samples:
        return b""

    if sample_width == 1:
        fmt_char = "b"
    elif sample_width == 2:
        fmt_char = "h"
    elif sample_width == 4:
        fmt_char = "i"
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Clip to valid range
    clipped = [
        max(-max_abs - 1, min(max_abs, int(s)))
        for s in samples
    ]

    fmt = f"<{len(clipped)}{fmt_char}"
    return struct.pack(fmt, *clipped)


def _rms_fallback(audio_data: bytes, sample_width: int) -> int:
    """Pure-Python RMS calculation used when audioop is unavailable."""

    samples, _ = _unpack_samples(audio_data, sample_width)
    if not samples:
        return 0
    # Standard RMS: sqrt(mean(x^2))
    acc = 0.0
    for s in samples:
        acc += float(s) * float(s)
    mean_sq = acc / float(len(samples))
    return int(mean_sq ** 0.5)


def _mul_fallback(audio_data: bytes, sample_width: int, factor: float) -> bytes:
    """Pure-Python volume scaling used when audioop is unavailable."""

    samples, max_abs = _unpack_samples(audio_data, sample_width)
    if not samples:
        return audio_data

    scaled = [s * factor for s in samples]
    return _pack_samples(scaled, sample_width, max_abs)


def _ratecv_fallback(
    audio_data: bytes,
    sample_width: int,
    channels: int,
    from_rate: int,
    to_rate: int,
) -> Tuple[bytes, None]:
    """Very simple linear-resampling fallback for audioop.ratecv.

    This operates per-frame, preserving the number of channels. It is
    not as sophisticated as the C implementation but is sufficient for
    microphone audio used by the assistant.
    """

    if from_rate == to_rate or not audio_data:
        return audio_data, None

    frame_width = sample_width * channels
    if frame_width <= 0:
        return audio_data, None

    total_frames = len(audio_data) // frame_width
    if total_frames == 0:
        return b"", None

    # Unpack all samples, then view them as a list of frames.
    flat_samples, max_abs = _unpack_samples(audio_data, sample_width)
    if not flat_samples:
        return b"", None

    frames: List[Tuple[int, ...]] = [
        tuple(flat_samples[i * channels:(i + 1) * channels])
        for i in range(total_frames)
    ]

    # Compute number of output frames
    new_total_frames = int(total_frames * to_rate / from_rate)
    if new_total_frames <= 0:
        return b"", None

    if new_total_frames == 1:
        new_frames = [frames[0]]
    else:
        new_frames: List[Tuple[int, ...]] = []
        for i in range(new_total_frames):
            # Position in input space
            src_pos = (i * (total_frames - 1)) / float(new_total_frames - 1)
            left_idx = int(src_pos)
            right_idx = min(left_idx + 1, total_frames - 1)
            alpha = src_pos - left_idx

            left = frames[left_idx]
            right = frames[right_idx]

            if left_idx == right_idx or alpha <= 0.0:
                new_frames.append(left)
            elif alpha >= 1.0:
                new_frames.append(right)
            else:
                # Linear interpolation per channel
                interp = tuple(
                    int((1.0 - alpha) * float(l) + alpha * float(r))
                    for l, r in zip(left, right)
                )
                new_frames.append(interp)

    # Flatten frames back to a single list of samples
    out_samples: List[int] = []
    for frame in new_frames:
        out_samples.extend(frame)

    return _pack_samples(out_samples, sample_width, max_abs), None


# =============================================================================
# Constants
# =============================================================================

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit audio
DEFAULT_CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION_MS = 1500


# =============================================================================
# Audio State
# =============================================================================

class AudioState(Enum):
    """State of the audio capture system."""
    IDLE = auto()
    LISTENING = auto()
    RECORDING = auto()
    PROCESSING = auto()
    ERROR = auto()


@dataclass
class AudioConfig:
    """Configuration for audio capture."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    sample_width: int = DEFAULT_SAMPLE_WIDTH
    chunk_size: int = DEFAULT_CHUNK_SIZE
    silence_threshold: int = SILENCE_THRESHOLD
    silence_duration_ms: int = SILENCE_DURATION_MS
    max_recording_seconds: float = 30.0
    min_recording_seconds: float = 0.5
    buffer_seconds: float = 0.5
    vad_aggressiveness: int = 2  # 0-3


@dataclass
class AudioChunk:
    """A chunk of audio data with metadata."""
    data: bytes
    timestamp: float = field(default_factory=time.time)
    rms: int = 0
    is_speech: bool = False
    
    def __post_init__(self):
        if self.rms == 0 and self.data:
            self.rms = calculate_rms(self.data)


# =============================================================================
# Audio Helper Functions
# =============================================================================

def calculate_rms(audio_data: bytes, sample_width: int = 2) -> int:
    """
    Calculate the Root Mean Square (RMS) of audio data.
    
    RMS is a measure of audio loudness/energy.
    
    Args:
        audio_data: Raw audio bytes
        sample_width: Bytes per sample (2 for 16-bit)
    
    Returns:
        RMS value (0-32768 for 16-bit audio)
    """
    if not audio_data:
        return 0

    # Prefer C implementation when available, fall back to pure Python.
    if _audioop is not None:
        try:
            return _audioop.rms(audio_data, sample_width)
        except Exception:
            # Fall back if audioop errors for any reason
            pass

    return _rms_fallback(audio_data, sample_width)


def is_silence(audio_data: bytes, threshold: int = SILENCE_THRESHOLD) -> bool:
    """
    Check if audio chunk is silence based on RMS threshold.
    
    Args:
        audio_data: Raw audio bytes
        threshold: RMS threshold below which is considered silence
    
    Returns:
        True if audio is below threshold (silence)
    """
    return calculate_rms(audio_data) < threshold


def normalize_audio(audio_data: bytes, sample_width: int = 2, target_rms: int = 8000) -> bytes:
    """
    Normalize audio volume to target RMS level.
    
    Args:
        audio_data: Raw audio bytes
        sample_width: Bytes per sample
        target_rms: Target RMS level
    
    Returns:
        Normalized audio bytes
    """
    if not audio_data:
        return audio_data
    
    current_rms = calculate_rms(audio_data, sample_width)
    if current_rms == 0:
        return audio_data
    
    factor = target_rms / current_rms
    # Clamp factor to prevent distortion
    factor = min(factor, 4.0)

    # Prefer C implementation when available, fall back otherwise.
    if _audioop is not None:
        try:
            return _audioop.mul(audio_data, sample_width, factor)
        except Exception:
            pass
    
    return _mul_fallback(audio_data, sample_width, factor)


def convert_sample_rate(
    audio_data: bytes,
    from_rate: int,
    to_rate: int,
    sample_width: int = 2,
    channels: int = 1,
) -> bytes:
    """
    Convert audio sample rate.
    
    Args:
        audio_data: Raw audio bytes
        from_rate: Original sample rate
        to_rate: Target sample rate
        sample_width: Bytes per sample
        channels: Number of channels
    
    Returns:
        Resampled audio bytes
    """
    if from_rate == to_rate:
        return audio_data
    
    # Prefer C implementation when available.
    if _audioop is not None:
        try:
            converted, _ = _audioop.ratecv(
                audio_data,
                sample_width,
                channels,
                from_rate,
                to_rate,
                None,
            )
            return converted
        except Exception as e:
            logger.error(f"Failed to convert sample rate with audioop: {e}")
    
    # Fallback to pure-Python resampling
    try:
        converted, _ = _ratecv_fallback(
            audio_data,
            sample_width,
            channels,
            from_rate,
            to_rate,
        )
        return converted
    except Exception as e:
        logger.error(f"Failed to convert sample rate with fallback: {e}")
        return audio_data


def audio_to_wav_bytes(
    audio_data: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> bytes:
    """
    Convert raw audio bytes to WAV format in memory.
    
    Args:
        audio_data: Raw PCM audio bytes
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        sample_width: Bytes per sample
    
    Returns:
        WAV file bytes
    """
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    return buffer.getvalue()


def save_audio_to_file(
    audio_data: bytes,
    file_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> bool:
    """
    Save raw audio data to a WAV file.
    
    Args:
        audio_data: Raw PCM audio bytes
        file_path: Path to save the WAV file
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        sample_width: Bytes per sample
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        return True
    except Exception as e:
        logger.error(f"Failed to save audio to {file_path}: {e}")
        return False


def create_temp_wav_file(
    audio_data: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
    prefix: str = "vesper_audio_",
) -> Optional[str]:
    """
    Create a temporary WAV file from audio data.
    
    The caller is responsible for deleting the file.
    
    Args:
        audio_data: Raw PCM audio bytes
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        sample_width: Bytes per sample
        prefix: Prefix for temp file name
    
    Returns:
        Path to temp file, or None if failed
    """
    try:
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav',
            prefix=prefix,
            delete=False,
        )
        temp_path = temp_file.name
        temp_file.close()
        
        if save_audio_to_file(audio_data, temp_path, sample_rate, channels, sample_width):
            return temp_path
        return None
    except Exception as e:
        logger.error(f"Failed to create temp WAV file: {e}")
        return None


# =============================================================================
# Audio Buffer
# =============================================================================

class AudioBuffer:
    """
    Thread-safe circular audio buffer.
    
    Maintains a rolling buffer of audio chunks for:
    - Pre-wake-word audio capture
    - Continuous recording with backlog
    
    Usage:
        buffer = AudioBuffer(max_seconds=5.0)
        buffer.add_chunk(audio_data)
        all_audio = buffer.get_all()
    """
    
    def __init__(
        self,
        max_seconds: float = 5.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
        channels: int = DEFAULT_CHANNELS,
    ):
        self._max_seconds = max_seconds
        self._sample_rate = sample_rate
        self._sample_width = sample_width
        self._channels = channels
        
        # Calculate max bytes
        bytes_per_second = sample_rate * sample_width * channels
        self._max_bytes = int(max_seconds * bytes_per_second)
        
        self._buffer: Deque[bytes] = deque()
        self._current_bytes = 0
        self._lock = threading.Lock()
    
    def add_chunk(self, audio_data: bytes) -> None:
        """Add audio chunk to buffer, removing old data if needed."""
        with self._lock:
            self._buffer.append(audio_data)
            self._current_bytes += len(audio_data)
            
            # Remove old chunks if over limit
            while self._current_bytes > self._max_bytes and self._buffer:
                removed = self._buffer.popleft()
                self._current_bytes -= len(removed)
    
    def get_all(self) -> bytes:
        """Get all audio data in buffer."""
        with self._lock:
            return b''.join(self._buffer)
    
    def get_last_seconds(self, seconds: float) -> bytes:
        """Get the last N seconds of audio."""
        bytes_per_second = self._sample_rate * self._sample_width * self._channels
        target_bytes = int(seconds * bytes_per_second)
        
        with self._lock:
            all_data = b''.join(self._buffer)
            if len(all_data) <= target_bytes:
                return all_data
            return all_data[-target_bytes:]
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._current_bytes = 0
    
    @property
    def duration_seconds(self) -> float:
        """Get current buffer duration in seconds."""
        bytes_per_second = self._sample_rate * self._sample_width * self._channels
        return self._current_bytes / bytes_per_second if bytes_per_second > 0 else 0


# =============================================================================
# Voice Activity Detection (VAD)
# =============================================================================

class SimpleVAD:
    """
    Simple Voice Activity Detection based on energy/RMS thresholds.
    
    This is a lightweight alternative to webrtcvad for basic speech detection.
    
    Usage:
        vad = SimpleVAD(threshold=500)
        for chunk in audio_stream:
            if vad.is_speech(chunk):
                process_speech(chunk)
    """
    
    def __init__(
        self,
        threshold: int = SILENCE_THRESHOLD,
        min_speech_frames: int = 3,
        min_silence_frames: int = 15,
    ):
        """
        Initialize VAD.
        
        Args:
            threshold: RMS threshold for speech detection
            min_speech_frames: Minimum consecutive speech frames to trigger
            min_silence_frames: Minimum consecutive silence frames to end
        """
        self._threshold = threshold
        self._min_speech_frames = min_speech_frames
        self._min_silence_frames = min_silence_frames
        
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
    
    def is_speech(self, audio_data: bytes) -> bool:
        """
        Check if audio chunk contains speech.
        
        Uses hysteresis to avoid rapid state changes.
        
        Args:
            audio_data: Raw audio bytes
        
        Returns:
            True if speech is detected
        """
        rms = calculate_rms(audio_data)
        
        if rms >= self._threshold:
            self._speech_frames += 1
            self._silence_frames = 0
            
            if self._speech_frames >= self._min_speech_frames:
                self._in_speech = True
        else:
            self._silence_frames += 1
            self._speech_frames = 0
            
            if self._silence_frames >= self._min_silence_frames:
                self._in_speech = False
        
        return self._in_speech
    
    def reset(self) -> None:
        """Reset VAD state."""
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
    
    @property
    def in_speech(self) -> bool:
        """Check if currently in speech segment."""
        return self._in_speech


# =============================================================================
# Microphone Capture
# =============================================================================

class MicrophoneStream:
    """
    Non-blocking microphone audio stream using PyAudio.
    
    Captures audio in a background thread and provides chunks via callback.
    
    Usage:
        def on_audio(chunk: bytes):
            process_chunk(chunk)
        
        mic = MicrophoneStream(callback=on_audio)
        mic.start()
        # ... later ...
        mic.stop()
    """
    
    def __init__(
        self,
        callback: Callable[[bytes], None],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        device_index: Optional[int] = None,
    ):
        """
        Initialize microphone stream.
        
        Args:
            callback: Function to call with each audio chunk
            sample_rate: Sample rate in Hz
            channels: Number of channels (1 = mono)
            chunk_size: Frames per buffer
            device_index: Specific input device index (None = default)
        """
        self._callback = callback
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._device_index = device_index
        
        self._pyaudio = None
        self._stream = None
        self._sd = None
        self._sd_stream = None
        self._backend = "none"
        self._running = False
        self._error: Optional[str] = None
    
    def start(self) -> bool:
        """
        Start capturing audio.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            import pyaudio
            
            self._pyaudio = pyaudio.PyAudio()

            # Resolve device index
            device_index = self._resolve_input_device_index()

            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self._chunk_size,
                stream_callback=self._audio_callback,
            )
            
            self._running = True
            self._backend = "pyaudio"
            self._stream.start_stream()
            logger.info("Microphone stream started (backend=pyaudio)")
            return True
            
        except ImportError:
            logger.warning("PyAudio not installed; attempting sounddevice input backend")
            return self._start_with_sounddevice()
        except OSError as e:
            logger.warning(f"PyAudio microphone open failed ({e}); attempting sounddevice backend")
            return self._start_with_sounddevice()
        except Exception as e:
            logger.warning(f"PyAudio backend failed ({e}); attempting sounddevice backend")
            return self._start_with_sounddevice()

    def _start_with_sounddevice(self) -> bool:
        """Fallback microphone backend using sounddevice when PyAudio is unavailable."""
        try:
            import sounddevice as sd
            import numpy as np

            del np  # imported to ensure ndarray support is available at runtime
            self._sd = sd
            device_index = self._resolve_sounddevice_input_device(sd)
            self._sd_stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                blocksize=self._chunk_size,
                dtype="int16",
                device=device_index,
                callback=self._sounddevice_callback,
            )
            self._sd_stream.start()
            self._running = True
            self._backend = "sounddevice"
            logger.info("Microphone stream started (backend=sounddevice)")
            return True
        except Exception as e:
            self._error = (
                f"Microphone access error: {e}. "
                "On macOS, ensure Terminal (or your IDE) has Microphone permission "
                "in System Settings > Privacy & Security > Microphone."
            )
            logger.warning(self._error)
            return False

    def _resolve_sounddevice_input_device(self, sd) -> Optional[int]:
        """Resolve sounddevice input device with safe fallback ordering."""
        if self._device_index is not None:
            try:
                info = sd.query_devices(self._device_index)
                if int(info.get("max_input_channels", 0)) > 0:
                    logger.debug(
                        f"Using configured sounddevice input device: {info.get('name', 'unknown')}"
                    )
                    return int(self._device_index)
            except Exception as e:
                logger.warning(f"Configured sounddevice index {self._device_index} unavailable: {e}")

        try:
            default_input, _ = sd.default.device
            if default_input is not None and int(default_input) >= 0:
                info = sd.query_devices(int(default_input))
                if int(info.get("max_input_channels", 0)) > 0:
                    logger.debug(f"Using default sounddevice input: {info.get('name', 'unknown')}")
                    return int(default_input)
        except Exception as e:
            logger.warning(f"Default sounddevice input unavailable: {e}")

        devices = self.list_input_devices()
        if devices:
            first = devices[0]
            logger.debug(f"Using first available sounddevice input: {first['name']}")
            return int(first["index"])

        raise OSError("No input devices found")

    def _resolve_input_device_index(self) -> Optional[int]:
        """Resolve the input device index with fallback to first available input device."""
        if not self._pyaudio:
            return self._device_index

        if self._device_index is not None:
            try:
                info = self._pyaudio.get_device_info_by_index(self._device_index)
                if info.get("maxInputChannels", 0) > 0:
                    logger.debug(f"Using configured input device: {info.get('name', 'unknown')}")
                    return self._device_index
                logger.warning(
                    f"Configured device index {self._device_index} has no input channels; "
                    "falling back to default input device."
                )
            except Exception as e:
                logger.warning(
                    f"Configured device index {self._device_index} not usable: {e}. "
                    "Falling back to default input device."
                )

        try:
            info = self._pyaudio.get_default_input_device_info()
            logger.debug(f"Using default input device: {info.get('name', 'unknown')}")
            return info.get("index")
        except Exception as e:
            logger.warning(f"Default input device not available: {e}")

        # Fallback: first available input device
        input_devices = self.list_input_devices()
        if input_devices:
            first = input_devices[0]
            logger.debug(f"Using first available input device: {first['name']}")
            return first["index"]

        raise OSError("No input devices found")

    def list_input_devices(self) -> List[dict]:
        """List available input devices for debugging."""
        devices: List[dict] = []
        created_local = False

        if not self._pyaudio:
            try:
                import pyaudio

                self._pyaudio = pyaudio.PyAudio()
                created_local = True
            except Exception as e:
                logger.warning(f"PyAudio not available for device listing: {e}")
                self._pyaudio = None
                return self._list_input_devices_sounddevice()

        try:
            count = self._pyaudio.get_device_count()
            for idx in range(count):
                info = self._pyaudio.get_device_info_by_index(idx)
                if info.get("maxInputChannels", 0) > 0:
                    devices.append({
                        "index": idx,
                        "name": info.get("name", "unknown"),
                        "channels": info.get("maxInputChannels", 0),
                        "defaultSampleRate": info.get("defaultSampleRate"),
                    })
        except Exception as e:
            logger.warning(f"Failed to list input devices: {e}")
        finally:
            if created_local and self._pyaudio:
                try:
                    self._pyaudio.terminate()
                except Exception:
                    pass
                self._pyaudio = None

        return devices

    def _list_input_devices_sounddevice(self) -> List[dict]:
        """List input devices through sounddevice when PyAudio is unavailable."""
        devices: List[dict] = []
        try:
            import sounddevice as sd

            queried = sd.query_devices()
            for idx, info in enumerate(queried):
                max_input = int(info.get("max_input_channels", 0))
                if max_input > 0:
                    devices.append(
                        {
                            "index": idx,
                            "name": info.get("name", "unknown"),
                            "channels": max_input,
                            "defaultSampleRate": info.get("default_samplerate"),
                        }
                    )
        except Exception as e:
            logger.warning(f"sounddevice not available for device listing: {e}")
        return devices
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - called from audio thread."""
        import pyaudio
        
        if self._running and in_data:
            try:
                self._callback(in_data)
            except Exception as e:
                logger.error(f"Audio callback error: {e}")
        
        return (None, pyaudio.paContinue if self._running else pyaudio.paComplete)

    def _sounddevice_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice callback - convert frames to raw bytes for pipeline compatibility."""
        del frames, time_info
        if status:
            logger.debug(f"sounddevice status: {status}")
        if self._running and indata is not None:
            try:
                self._callback(indata.tobytes())
            except Exception as e:
                logger.error(f"Audio callback error (sounddevice): {e}")
    
    def stop(self) -> None:
        """Stop capturing audio."""
        self._running = False
        
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.debug(f"Error closing stream: {e}")
            self._stream = None
        
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception as e:
                logger.debug(f"Error terminating PyAudio: {e}")
            self._pyaudio = None

        if self._sd_stream:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception as e:
                logger.debug(f"Error closing sounddevice stream: {e}")
            self._sd_stream = None

        self._sd = None
        self._backend = "none"
        logger.info("Microphone stream stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if stream is running."""
        return self._running
    
    @property
    def error(self) -> Optional[str]:
        """Get last error message."""
        return self._error


# =============================================================================
# Audio Recorder
# =============================================================================

class AudioRecorder:
    """
    Records audio with automatic silence detection.
    
    Starts recording on speech detection and stops after silence timeout.
    
    Usage:
        recorder = AudioRecorder()
        recorder.start_recording()
        # ... audio chunks are added via add_chunk() ...
        if recorder.is_complete:
            audio = recorder.get_recording()
    """
    
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
    ):
        self._config = config or AudioConfig()
        self._vad = SimpleVAD(
            threshold=self._config.silence_threshold,
        )
        
        self._recording: List[bytes] = []
        self._is_recording = False
        self._speech_started = False
        self._silence_start: Optional[float] = None
        self._recording_start: Optional[float] = None
        self._lock = threading.Lock()
    
    def start_recording(self) -> None:
        """Start a new recording session."""
        with self._lock:
            self._recording.clear()
            self._is_recording = True
            self._speech_started = False
            self._silence_start = None
            self._recording_start = time.time()
            self._vad.reset()
        logger.debug("Recording started")
    
    def add_chunk(self, audio_data: bytes) -> None:
        """Add audio chunk to recording."""
        if not self._is_recording:
            return
        
        with self._lock:
            self._recording.append(audio_data)
            
            # Check for speech
            is_speech = self._vad.is_speech(audio_data)
            
            if is_speech:
                self._speech_started = True
                self._silence_start = None
            elif self._speech_started:
                # Speech ended, start silence timer
                if self._silence_start is None:
                    self._silence_start = time.time()
    
    def stop_recording(self) -> bytes:
        """Stop recording and return audio data."""
        with self._lock:
            self._is_recording = False
            audio_data = b''.join(self._recording)
            self._recording.clear()
        logger.debug(f"Recording stopped, {len(audio_data)} bytes")
        return audio_data
    
    def get_recording(self) -> bytes:
        """Get current recording without stopping."""
        with self._lock:
            return b''.join(self._recording)
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
    
    @property
    def is_complete(self) -> bool:
        """
        Check if recording is complete.
        
        Complete when:
        - Speech was detected AND silence timeout reached, OR
        - Max recording duration reached
        """
        if not self._is_recording:
            return False
        
        now = time.time()
        
        # Check max duration
        if self._recording_start:
            duration = now - self._recording_start
            if duration >= self._config.max_recording_seconds:
                return True
        
        # Check silence timeout (only after speech started)
        if self._speech_started and self._silence_start:
            silence_duration = (now - self._silence_start) * 1000  # ms
            if silence_duration >= self._config.silence_duration_ms:
                return True
        
        return False
    
    @property
    def duration_seconds(self) -> float:
        """Get current recording duration."""
        if self._recording_start:
            return time.time() - self._recording_start
        return 0.0
    
    @property
    def has_speech(self) -> bool:
        """Check if speech was detected in recording."""
        return self._speech_started
