"""Run voice output as a gateway client: subscribe, speak Vesper's serif lines.

    python -m voice.output       # speaks replies (if voice.output.enabled)
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from voice.output.config import VoiceOutputConfig
from voice.output.speaker import VoiceOutputService

logger = logging.getLogger("vesper.voice.out")


def load_config() -> VoiceOutputConfig:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass
    from config.settings import load_config_dict

    return VoiceOutputConfig.from_app_config(load_config_dict())


async def _run(config: VoiceOutputConfig) -> None:
    import websockets

    service = VoiceOutputService(config)
    token = config.gateway_token
    uri = f"ws://{config.gateway_host}:{config.gateway_port}/ws"
    if token:
        uri += f"?token={token}"

    backoff = 0.5
    while True:
        try:
            async with websockets.connect(uri) as ws:
                backoff = 0.5
                logger.info("voice output connected to gateway")
                async for raw in ws:
                    try:
                        service.handle(json.loads(raw))
                    except Exception:
                        logger.exception("voice output failed to handle a message")
        except Exception as exc:
            logger.info("gateway link down (%s); retrying in %.1fs", type(exc).__name__, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config = load_config()
    if not config.enabled:
        logger.info("voice.output.enabled is false — voice output is off.")
        return 0
    logger.info("Voice output starting | voice=%s | streaming=%s", config.tts_voice, config.streaming)
    try:
        asyncio.run(_run(config))
    except KeyboardInterrupt:
        logger.info("voice output stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
