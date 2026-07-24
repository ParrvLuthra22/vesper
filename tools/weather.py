"""Weather tool (PC4) — current_weather(location) via Open-Meteo.

Open-Meteo is free and needs no API key: a geocoding call turns a place name into
coordinates, then a forecast call returns current conditions. `location` may be a
place name ("London"), a "lat,lon" pair, or empty (falls back to the configured
default). Safe tier — a read-only lookup. Registered on import (tools/__init__.py).
"""
from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

from tools.registry import ToolSpec, get_registry
from utils.logger import get_logger

logger = get_logger(__name__)
registry = get_registry()

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather-code → short description (the codes Open-Meteo returns).
_WEATHER_CODES = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle", 61: "light rain", 63: "rain",
    65: "heavy rain", 66: "freezing rain", 67: "freezing rain", 71: "light snow",
    73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
    82: "violent showers", 85: "snow showers", 86: "snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def _cfg(key: str, default: Any = None) -> Any:
    from config.settings import load_config_dict

    value: Any = load_config_dict()
    for part in key.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(part)
        if value is None:
            return default
    return value


def _http_get_json(url: str) -> Dict[str, Any]:
    """Blocking GET → parsed JSON. Split out as the single network seam so tests
    can monkeypatch it."""
    req = urllib.request.Request(url, headers={"User-Agent": "Vesper/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read(200_000).decode(errors="replace"))


async def _get_json(url: str) -> Dict[str, Any]:
    return await asyncio.get_event_loop().run_in_executor(None, _http_get_json, url)


def _parse_latlon(location: str) -> Optional[Tuple[float, float]]:
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", location or "")
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


async def _geocode(place: str) -> Optional[Tuple[float, float, str]]:
    url = f"{GEOCODE_URL}?{urllib.parse.urlencode({'name': place, 'count': 1})}"
    data = await _get_json(url)
    results = data.get("results") or []
    if not results:
        return None
    r = results[0]
    return float(r["latitude"]), float(r["longitude"]), r.get("name", place)


async def current_weather(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    location = str(arguments.get("location", "") or "").strip()
    if not location:
        location = str(_cfg("proactive.morning_routine.weather_location", "") or "").strip()
    if not location or location.lower() == "auto":
        return "No location set for weather, Sir — give me a city."

    label = location
    latlon = _parse_latlon(location)
    try:
        if latlon is None:
            geo = await _geocode(location)
            if geo is None:
                return f"I couldn't find '{location}', Sir."
            lat, lon, label = geo
        else:
            lat, lon = latlon

        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
        }
        data = await _get_json(f"{FORECAST_URL}?{urllib.parse.urlencode(params)}")
    except Exception as exc:
        logger.warning(f"[weather] lookup failed for {location!r}: {exc}")
        return f"Weather lookup failed, Sir: {exc}"

    current = data.get("current", {}) or {}
    daily = data.get("daily", {}) or {}
    temp = current.get("temperature_2m")
    code = int(current.get("weather_code", -1)) if current.get("weather_code") is not None else -1
    desc = _WEATHER_CODES.get(code, "unclear conditions")
    highs = (daily.get("temperature_2m_max") or [None])[0]
    lows = (daily.get("temperature_2m_min") or [None])[0]

    parts = [f"{label}: {desc}"]
    if temp is not None:
        parts.append(f"{round(temp)}°C now")
    if highs is not None and lows is not None:
        parts.append(f"{round(lows)}–{round(highs)}°C today")
    return ", ".join(parts) + "."


def _register() -> None:
    registry.register(ToolSpec(
        name="current_weather",
        description=(
            "Current weather for a location via Open-Meteo (no API key). `location` "
            "may be a place name, a 'lat,lon' pair, or omitted to use the configured "
            "default. Returns a one-line summary."
        ),
        parameters={"type": "object", "properties": {
            "location": {"type": "string", "description": "Place name or 'lat,lon'; optional."}}},
        tier="safe", handler=current_weather, category="general",
    ))


_register()
