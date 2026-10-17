"""Search/weather failures are explained in one honest sentence, not a generic 'service unavailable'. Mocked providers; no network."""
from __future__ import annotations

import asyncio
import urllib.error

import pytest

from launcher import doctor
from tools.weather import explain_weather_failure
from tools.web_search import Hit, explain_failure, make_handler


class Boom:
    def __init__(self, exc, name="duckduckgo"):
        self.exc, self.name = exc, name

    def search(self, q, n):
        raise self.exc


async def run(exc, **kw):
    return await make_handler(Boom(exc), None)({"query": "rust"}, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("exc,expect", [
    (ImportError("No module named 'ddgs'", name="ddgs"), "ddgs is missing"),
    (RuntimeError("202 Ratelimit"), "rate-limiting"),
    (RuntimeError("HTTP 429 Too Many Requests"), "rate-limiting"),
    (TimeoutError("timed out"), "did not answer in time"),
    (urllib.error.URLError("[Errno 8] nodename nor servname provided"), "cannot reach the internet"),
    (ConnectionError("Connection refused"), "cannot reach the internet"),
    (RuntimeError("HTTP Error 503"), "HTTP 503"),
    (ValueError("weird"), "ValueError"),
])
async def test_search_failure_gives_the_specific_reason(exc, expect):
    out = await run(exc)
    assert out.startswith("Web search could not run:") and expect in out
    assert "service unavailable" not in out.lower().replace("do not say 'service unavailable'", "")
    assert "one short sentence" in out


@pytest.mark.asyncio
async def test_a_working_fallback_hides_the_primary_failure():
    ok = type("Ok", (), {"name": "duckduckgo", "search": lambda self, q, n: [Hit("T", "S", "https://a.example/")]})()
    out = await make_handler(Boom(RuntimeError("429"), "tavily"), ok)({"query": "x y"}, {})
    assert "[a.example] T" in out and "could not run" not in out


def test_weather_failure_names_the_service_and_points_at_web_search():
    e = urllib.error.HTTPError("https://api.open-meteo.com/v1/forecast", 503, "Service Unavailable", {}, None)
    msg = explain_weather_failure(e)
    assert "Open-Meteo" in msg and "HTTP 503" in msg and "web_search" in msg
    assert explain_weather_failure(TimeoutError("timed out")).count("did not answer in time") == 1
    assert "cannot reach the internet" in explain_weather_failure(urllib.error.URLError("nodename nor servname"))
    assert "api.open-meteo.com" not in msg                                     # no URLs echoed to the model


def test_explain_failure_is_stable_for_unknown_errors():
    assert "KeyError" in explain_failure(KeyError("x"), "duckduckgo")


def test_doctor_fails_when_ddgs_cannot_be_imported_and_says_how_to_fix_it():
    def no(name):
        raise ModuleNotFoundError(name)

    [c] = doctor.check_web_search({}, importer=no)
    assert c.status == doctor.FAIL and "pip install ddgs" in c.fix and "ModuleNotFoundError" in c.detail
    ok = doctor.check_web_search({"TAVILY_API_KEY": "sekret-value"}, importer=lambda n: object())
    assert [x.status for x in ok] == [doctor.PASS, doctor.PASS] and "sekret-value" not in "".join(x.detail for x in ok)
