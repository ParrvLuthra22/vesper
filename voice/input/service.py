"""Run the voice-input pipeline as a standalone client of the gateway.

    python -m voice.input          # start listening (if voice.input.enabled)

Loads the app config (settings.yaml), builds the real stages, and drives the
mic. Off unless voice.input.enabled is true.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

from voice.input.audio import FileSource, MicSource
from voice.input.config import VoiceInputConfig, VoiceTransition
from voice.input.pipeline import VoiceInputPipeline
from voice.input.sink import GatewayRestSink
from voice.input.stages import Transcriber, VoiceInputUnavailable, WakeDetector

logger = logging.getLogger("vesper.voice")

#: Printed once the microphone is streaming and the wake model is loaded. The
#: launcher's readiness probe matches on this exact text.
READY_MARKER = "voice input ready"

#: Exit status for "voice input cannot work this session" (missing model/package,
#: mic unavailable). Distinct from 1 (an uncaught crash, which Python also exits
#: with) so the launcher knows NOT to restart-loop against it. (sysexits EX_UNAVAILABLE.)
EXIT_UNAVAILABLE = 69


def _log_emitter(t: VoiceTransition) -> None:
    # PV4 will forward these to the HUD (star flare on wake, etc.).
    logger.debug("transition: %s <- %s %s", t.state.value, t.previous.value, t.detail)


def build_pipeline(config: VoiceInputConfig, sink=None, emitter=None) -> VoiceInputPipeline:
    sink = sink if sink is not None else GatewayRestSink(config)
    on_wake = getattr(sink, "signal_wake", None)  # POST /wake -> cinematic reveal
    return VoiceInputPipeline(
        config=config,
        wake=WakeDetector(config),
        transcriber=Transcriber(config),
        sink=sink,
        emitter=emitter if emitter is not None else _log_emitter,
        on_wake=on_wake,
    )


class _AnnouncingMicSource(MicSource):
    """MicSource that logs READY_MARKER once the first frame arrives — i.e. the
    mic is open and streaming. The launcher waits for this line.

    A watchdog also covers the case where the first frame NEVER arrives: a blocked
    permission prompt or a device held by another app makes the stream's blocking
    read hang forever (it does not raise). After `first_frame_timeout` seconds the
    process exits EXIT_UNAVAILABLE with a clear message, so the launcher reports one
    precise failure instead of timing out and retrying."""

    def __init__(self, *args, first_frame_timeout: float = 0.0, on_timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_frame_timeout = first_frame_timeout
        self._on_timeout = on_timeout or _exit_no_audio

    def frames(self):
        announced = False
        timer = None
        if self._first_frame_timeout > 0:
            timer = threading.Timer(self._first_frame_timeout, self._on_timeout, args=(self._first_frame_timeout,))
            timer.daemon = True
            timer.start()
        try:
            for frame in super().frames():
                if not announced:
                    announced = True
                    if timer is not None:
                        timer.cancel()
                    logger.info(READY_MARKER)
                yield frame
        finally:
            if timer is not None:
                timer.cancel()


def _exit_no_audio(timeout: float) -> None:
    logger.error(
        "Voice input unavailable: the microphone delivered no audio within %.0fs. Grant Microphone "
        "permission to your terminal (System Settings > Privacy & Security > Microphone), close any app "
        "holding the mic, and run scripts/doctor.py.", timeout,
    )
    logging.shutdown()
    os._exit(EXIT_UNAVAILABLE)


def load_config() -> VoiceInputConfig:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass
    from config.settings import load_config_dict

    return VoiceInputConfig.from_app_config(load_config_dict())


def run_file(config: VoiceInputConfig, wav_path: str, sink=None, emitter=None) -> VoiceInputPipeline:
    """Drive the pipeline over a WAV file (for tests / no-mic verification)."""
    pipeline = build_pipeline(config, sink=sink, emitter=emitter)
    pipeline.run(FileSource(wav_path, config.frame_samples, config.sample_rate))
    return pipeline


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config = load_config()
    if not config.enabled:
        logger.info("voice.input.enabled is false — voice input is off. Enable it in settings.yaml.")
        return 0

    logger.info(
        "Voice input starting | wake_model=%s | whisper=%s (%s) | vad=%s | mic=%s",
        config.wake_model, config.whisper_model, config.whisper_compute_type,
        config.vad_backend, config.mic_device if config.mic_device is not None else "default",
    )
    pipeline = build_pipeline(config)
    source = _AnnouncingMicSource(
        config.sample_rate, config.frame_samples, config.mic_device,
        first_frame_timeout=config.first_frame_timeout_seconds,
    )
    try:
        # Load the wake model NOW (not lazily on the first frame) so a missing
        # package/model fails immediately and loudly, and "ready" below means
        # "will actually hear the wake word". Whisper stays lazy on purpose:
        # it is ~100 MB that only needs to exist while transcribing.
        pipeline.warm_up()
        pipeline.run(source)
    except KeyboardInterrupt:
        pipeline.stop()
        logger.info("voice input stopped")
        return 0
    except VoiceInputUnavailable as exc:
        # Raised before the frame loop starts (e.g. sounddevice missing, or
        # the mic is held by another process). One line, then exit — never a
        # retry loop against a device that is not coming back this session.
        logger.warning("Voice input disabled for this session — %s", exc)
        return EXIT_UNAVAILABLE
    except Exception as exc:
        logger.warning(
            "Voice input disabled for this session — microphone unavailable: %s. "
            "Run scripts/doctor.py to check the audio stack.",
            exc,
        )
        return EXIT_UNAVAILABLE

    # A permanent stage failure stops run() without raising; report it here so
    # the exit status reflects that voice never actually started.
    if pipeline.disabled_reason is not None:
        return EXIT_UNAVAILABLE
    return 0


if __name__ == "__main__":
    sys.exit(main())
