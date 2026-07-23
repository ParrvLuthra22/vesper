"""Run the voice-input pipeline as a standalone client of the gateway.

    python -m voice.input          # start listening (if voice.input.enabled)

Loads the app config (settings.yaml), builds the real stages, and drives the
mic. Off unless voice.input.enabled is true.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from voice.input.audio import FileSource, MicSource
from voice.input.config import VoiceInputConfig, VoiceTransition
from voice.input.pipeline import VoiceInputPipeline
from voice.input.sink import GatewayRestSink
from voice.input.stages import Transcriber, WakeDetector

logger = logging.getLogger("vesper.voice")


def _log_emitter(t: VoiceTransition) -> None:
    # PV4 will forward these to the HUD (star flare on wake, etc.).
    logger.debug("transition: %s <- %s %s", t.state.value, t.previous.value, t.detail)


def build_pipeline(config: VoiceInputConfig, sink=None, emitter=None) -> VoiceInputPipeline:
    return VoiceInputPipeline(
        config=config,
        wake=WakeDetector(config),
        transcriber=Transcriber(config),
        sink=sink if sink is not None else GatewayRestSink(config),
        emitter=emitter if emitter is not None else _log_emitter,
    )


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
    source = MicSource(config.sample_rate, config.frame_samples, config.mic_device)
    try:
        pipeline.run(source)
    except KeyboardInterrupt:
        pipeline.stop()
        logger.info("voice input stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
