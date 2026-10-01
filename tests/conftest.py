import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import pytest

# Guardian audit isolation — runs at import, before any test can construct a
# Guardian. Tests must never write to the real data/audit.jsonl.
_TEST_AUDIT_DIR = tempfile.mkdtemp(prefix="vesper-test-audit-")
os.environ["VESPER_AUDIT_LOG"] = str(Path(_TEST_AUDIT_DIR) / "audit.jsonl")
_REAL_AUDIT_LOG = Path(__file__).resolve().parents[1] / "data" / "audit.jsonl"


def _audit_fingerprint():
    try:
        st = _REAL_AUDIT_LOG.stat()
        return (st.st_size, st.st_mtime_ns)
    except FileNotFoundError:
        return None


@pytest.fixture(scope="session", autouse=True)
def _real_audit_log_untouched():
    """Fail the run if anything in it modified (or created) the real audit log."""
    before = _audit_fingerprint()
    yield
    assert _audit_fingerprint() == before, (
        "the test run modified the real data/audit.jsonl — a test is not using "
        "an isolated audit path (see VESPER_AUDIT_LOG in tests/conftest.py)"
    )


class _FakeGeminiResponse:
    def __init__(self, payload: dict):
        self.text = json.dumps(payload)


class _FakeGeminiModels:
    def generate_content(self, *args: Any, **kwargs: Any) -> _FakeGeminiResponse:
        return _FakeGeminiResponse({"intents": [], "is_multi_command": False, "execution_mode": "sequential"})


class _FakeGeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = _FakeGeminiModels()


class _FakeGeminiConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


@pytest.fixture(autouse=True)
def _mock_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock Gemini client and ensure API key is present for tests."""
    os.environ.setdefault("GEMINI_API_KEY", "test-key")

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    types_module = types.ModuleType("google.genai.types")

    genai_module.Client = _FakeGeminiClient
    types_module.GenerateContentConfig = _FakeGeminiConfig
    genai_module.types = types_module
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)


@pytest.fixture(autouse=True)
def _mock_audio_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock audio hardware dependencies to keep CI tests deterministic."""
    import utils.audio as audio
    import utils.stt as stt
    import utils.tts as tts

    class DummyMicrophoneStream:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.error = "mocked"

        def start(self) -> bool:
            return False

        def stop(self) -> None:
            return None

        def list_input_devices(self):
            return []

    class DummyDetector:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def initialize(self) -> bool:
            return False

        def shutdown(self) -> None:
            return None

    class DummyTranscriber:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def initialize(self) -> bool:
            return False

        def shutdown(self) -> None:
            return None

    class DummyTTS:
        async def initialize(self) -> bool:
            return True

        async def speak(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def shutdown(self) -> None:
            return None

    async def dummy_create_tts_provider(*args: Any, **kwargs: Any) -> DummyTTS:
        return DummyTTS()

    monkeypatch.setattr(audio, "MicrophoneStream", DummyMicrophoneStream)
    monkeypatch.setattr(stt, "VoskWakeWordDetector", DummyDetector)
    monkeypatch.setattr(stt, "WhisperTranscriber", DummyTranscriber)
    monkeypatch.setattr(stt, "SpeechRecognitionTranscriber", DummyTranscriber)
    monkeypatch.setattr(tts, "create_tts_provider", dummy_create_tts_provider)
