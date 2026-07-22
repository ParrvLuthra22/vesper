"""Health check API for VESPER."""

from __future__ import annotations

from threading import Thread
from typing import Callable, Dict, Optional

from fastapi import FastAPI
import uvicorn

from agents.base_agent import BaseAgent


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
        host: str = "0.0.0.0",
        port: int = 8080,
        log_level: str = "warning",
    ) -> None:
        self._agents_provider = agents_provider
        self._host = host
        self._port = port
        self._log_level = log_level
        self._thread: Optional[Thread] = None
        self._server: Optional[uvicorn.Server] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
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
