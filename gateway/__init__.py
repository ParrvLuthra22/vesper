"""Vesper API Gateway (PV0).

A FastAPI surface — WebSocket + REST — that wraps the existing event bus so
any interface (web, mobile, another process) can attach to the SAME running
Brain. The gateway runs in-process with the Brain and adds no assistant logic
of its own: it only bridges the bus. LOCALHOST ONLY in v2.
"""

from gateway.server import Gateway, create_app
from gateway.wire import FORWARDED_EVENT_TYPES, WIRE_SPEC, to_wire

__all__ = ["Gateway", "create_app", "to_wire", "WIRE_SPEC", "FORWARDED_EVENT_TYPES"]
