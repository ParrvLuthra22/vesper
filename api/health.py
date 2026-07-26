"""Health check API for VESPER."""

from __future__ import annotations

import socket
from threading import Thread
from typing import Callable, Dict, Optional

from fastapi import FastAPI
import uvicorn

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)

AgentsProvider = Callable[[], Dict[str, BaseAgent]]


def create_app(agents_provider: AgentsProvider) -> FastAPI:
    """Create a FastAPI app that exposes a /health endpoint."""
    app = FastAPI(title="VESPER Health", version="1.0.0")

    @app.get("/health")
    async def health() -> Dict[str, Dict[str, str]]:
        agents = agents_provider() or {}
        statuses: Dict[str, str] = {}
        all_healthy = True

        for name, agent in agents.items():
            try:
                healthy = bool(agent.is_healthy())
            except Exception:
                healthy = False
            statuses[name] = "healthy" if healthy else "unhealthy"
            if not healthy:
                all_healthy = False

        return {
            "status": "ok" if all_healthy else "degraded",
            "agents": statuses,
        }

    return app


class HealthServer:
    """Run the health API in a background thread."""

    def __init__(
        self,
        agents_provider: AgentsProvider,
        host: str = "127.0.0.1",  # localhost only — health API is local telemetry
        port: int = 8080,
        log_level: str = "warning",
    ) -> None:
        self._agents_provider = agents_provider
        self._host = host
        self._port = port
        self._log_level = log_level
        self._thread: Optional[Thread] = None
        self._server: Optional[uvicorn.Server] = None

    @staticmethod
    def _port_in_use(host: str, port: int) -> bool:
        probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((probe_host, int(port)) ) == 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        # Skip cleanly if the port is already taken (usually a leftover VESPER
        # still running) instead of letting uvicorn crash the thread with a
        # scary "address already in use" traceback. The health API is optional.
        if self._port_in_use(self._host, self._port):
            logger.info(
                f"Health API not started — {self._host}:{self._port} already in use "
                "(another VESPER instance may be running)."
            )
            return

        app = create_app(self._agents_provider)
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            log_level=self._log_level,
            loop="asyncio",
        )
        self._server = uvicorn.Server(config)
        self._thread = Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self._server = None
