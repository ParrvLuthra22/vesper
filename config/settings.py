"""Typed application settings using pydantic-settings with YAML support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class GeneralSettings(BaseModel):
    assistant_name: str = "VESPER"
    version: str = "0.1.0"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    log_file: str = "logs/vesper.log"
    log_rotation: str = "daily"
    log_max_size_mb: int = 10
    log_backup_count: int = 7
    debug_mode: bool = False
    json_logs: bool = False
    event_tracing: bool = True


class VoiceVoskSettings(BaseModel):
    model_path: str = "models/vosk-model-small-en-us-0.15"
    sample_rate: int = 16000


class VoiceWhisperSettings(BaseModel):
    binary_path: str = "/usr/local/bin/whisper-cpp"
    model_path: str = "models/ggml-base.en.bin"
    language: str = "en"
    threads: int = 4
    output_format: str = "json"


class VoiceRecognitionSettings(BaseModel):
    provider: str = "vosk_whisper"
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 100
    buffer_seconds: float = 0.5
    silence_threshold: int = 500
    silence_duration_ms: int = 1500
    max_recording_seconds: int = 30
    min_recording_seconds: float = 0.5
    vad_aggressiveness: int = 2


class VoiceSynthesisSettings(BaseModel):
    provider: str = "system"
    voice: str = "Samantha"
    rate: int = 180
    volume: float = 1.0
    fallback_provider: str = "system"
    fallback_voice: str = "Samantha"


class VoiceTTSSettings(BaseModel):
    engine: str = "kokoro"
    voice_id: str = "af_heart"


class VoiceSettings(BaseModel):
    wake_word: str = "vesper"
    wake_word_sensitivity: float = 0.5
    vosk: VoiceVoskSettings = Field(default_factory=VoiceVoskSettings)
    whisper: VoiceWhisperSettings = Field(default_factory=VoiceWhisperSettings)
    recognition: VoiceRecognitionSettings = Field(default_factory=VoiceRecognitionSettings)
    tts: VoiceTTSSettings = Field(default_factory=VoiceTTSSettings)
    synthesis: VoiceSynthesisSettings = Field(default_factory=VoiceSynthesisSettings)
    address_user_as_sir: bool = True


class IntentProviderSettings(BaseModel):
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 500
    api_key: Optional[str] = None


class IntentOllamaSettings(BaseModel):
    model: str = "qwen2.5:7b-instruct"
    endpoint: str = "http://127.0.0.1:11434/api/generate"
    temperature: float = 0.2


class IntentSettings(BaseModel):
    provider: str = "pattern"
    gemini: IntentProviderSettings = Field(default_factory=lambda: IntentProviderSettings(model="gemini-2.0-flash"))
    openai: IntentProviderSettings = Field(default_factory=lambda: IntentProviderSettings(model="gpt-4"))
    ollama: IntentOllamaSettings = Field(default_factory=IntentOllamaSettings)
    confidence_threshold: float = 0.5
    ambiguity_threshold: float = 0.3
    intents: List[Dict[str, Any]] = Field(default_factory=list)


class SystemMacOSSettings(BaseModel):
    use_applescript: bool = True
    use_shortcuts: bool = True
    allowed_apps: List[str] = Field(default_factory=list)
    blocked_apps: List[str] = Field(default_factory=list)
    confirm_actions: List[str] = Field(default_factory=list)


class SystemWeatherAPISettings(BaseModel):
    provider: str = "openweathermap"
    default_location: str = "auto"


class SystemAPISettings(BaseModel):
    weather: SystemWeatherAPISettings = Field(default_factory=SystemWeatherAPISettings)


class SystemSettings(BaseModel):
    macos: SystemMacOSSettings = Field(default_factory=SystemMacOSSettings)
    apis: SystemAPISettings = Field(default_factory=SystemAPISettings)


class MemorySQLiteSettings(BaseModel):
    database_path: str = "data/memory.db"


class MemoryBoundedSettings(BaseModel):
    max_items: int = 100
    ttl_seconds: Optional[int] = None


class MemoryConversationSettings(BaseModel):
    max_turns: int = 20
    summarize_after: int = 10


class MemoryVectorStoreSettings(BaseModel):
    enabled: bool = False
    provider: str = "chroma"
    persist_directory: str = "data/chroma_memory"
    collection_name: str = "vesper_memory"
    embedding_model: Optional[str] = None
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 64


class MemorySettings(BaseModel):
    backend: str = "sqlite"
    sqlite: MemorySQLiteSettings = Field(default_factory=MemorySQLiteSettings)
    short_term: MemoryBoundedSettings = Field(default_factory=lambda: MemoryBoundedSettings(max_items=100, ttl_seconds=3600))
    long_term: MemoryBoundedSettings = Field(default_factory=lambda: MemoryBoundedSettings(max_items=10000, ttl_seconds=None))
    conversation: MemoryConversationSettings = Field(default_factory=MemoryConversationSettings)
    vector_store: MemoryVectorStoreSettings = Field(default_factory=MemoryVectorStoreSettings)


class OrchestratorSettings(BaseModel):
    startup_order: List[str] = Field(default_factory=lambda: ["MemoryAgent", "SystemAgent", "VoiceAgent", "IntentAgent", "VisionAgent"])
    health_check_interval: int = 30
    default_task_timeout: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 1
    shutdown_timeout_seconds: int = 10


class EventBusSettings(BaseModel):
    max_queue_size: int = 1000
    persist_events: bool = False
    event_log_path: str = "data/events.log"
    enable_metrics: bool = True


class SecurityRateLimitSettings(BaseModel):
    enabled: bool = True
    max_requests_per_minute: int = 60


class SecuritySettings(BaseModel):
    sandbox_commands: bool = True
    max_command_length: int = 500
    rate_limit: SecurityRateLimitSettings = Field(default_factory=SecurityRateLimitSettings)


class GatewaySettings(BaseModel):
    """API Gateway (PV0). LOCALHOST ONLY in v2 — see settings.yaml notes.

    The token is read at runtime from env VESPER_GATEWAY_TOKEN (preferred),
    falling back to `token` here. An empty token fails closed (rejects all).
    """

    host: str = "127.0.0.1"
    port: int = 8760
    token: str = ""


class HUDSettings(BaseModel):
    enabled: bool = True
    width: int = 620
    height: int = 320
    x: int = 24
    y: int = 24
    alpha: float = 0.88
    background: str = "#0a0e1a"


class UISettings(BaseModel):
    enabled: bool = True
    hud: HUDSettings = Field(default_factory=HUDSettings)


class VisionCameraSettings(BaseModel):
    device_id: int = 0
    resolution: List[int] = Field(default_factory=lambda: [1280, 720])
    fps: int = 30
    capture_interval_ms: int = 100
    auto_start: bool = False
    show_preview: bool = True


class VisionGesturesSettings(BaseModel):
    enabled: bool = True
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    max_num_hands: int = 2
    cooldown_seconds: float = 1.0
    gesture_actions: Dict[str, str] = Field(default_factory=dict)


class VisionFaceSettings(BaseModel):
    enabled: bool = True
    use_mediapipe: bool = True
    enable_recognition: bool = True
    min_detection_confidence: float = 0.5
    recognition_tolerance: float = 0.5
    enrolled_faces_dir: str = "data/faces"
    detection_cooldown_seconds: float = 5.0
    recognition_cooldown_seconds: float = 30.0


class VisionPerformanceSettings(BaseModel):
    processing_scale: float = 0.5
    use_gpu: bool = False
    num_threads: int = 2


class VisionPrivacySettings(BaseModel):
    no_image_storage: bool = True
    blur_faces_in_storage: bool = True
    require_explicit_start: bool = True


class VisionSettings(BaseModel):
    enabled: bool = False
    use_gemini: bool = False
    local_ocr_enabled: bool = True
    camera: VisionCameraSettings = Field(default_factory=VisionCameraSettings)
    gestures: VisionGesturesSettings = Field(default_factory=VisionGesturesSettings)
    face: VisionFaceSettings = Field(default_factory=VisionFaceSettings)
    performance: VisionPerformanceSettings = Field(default_factory=VisionPerformanceSettings)
    privacy: VisionPrivacySettings = Field(default_factory=VisionPrivacySettings)


class WebSearchGeminiSettings(BaseModel):
    api_key: Optional[str] = None
    model: str = "gemini-1.5-flash"


class WebSearchOpenRouterSettings(BaseModel):
    api_key: Optional[str] = None
    model: str = "x-ai/grok-3-mini-beta"
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"


class WebSearchSettings(BaseModel):
    tavily_api_key: Optional[str] = None
    llm_provider: str = "auto"
    gemini: WebSearchGeminiSettings = Field(default_factory=WebSearchGeminiSettings)
    openrouter: WebSearchOpenRouterSettings = Field(default_factory=WebSearchOpenRouterSettings)


class LLMTierSettings(BaseModel):
    """A (provider, model) pair — used for both the primary/fallback tiers
    and per-purpose overrides."""

    provider: str = ""
    model: str = ""


class LLMGroqSettings(BaseModel):
    api_key: Optional[str] = None
    timeout_seconds: float = 30.0


class LLMOllamaSettings(BaseModel):
    endpoint: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 60.0


class LLMSettings(BaseModel):
    primary: LLMTierSettings = Field(
        default_factory=lambda: LLMTierSettings(provider="groq", model="openai/gpt-oss-120b")
    )
    fallback: LLMTierSettings = Field(
        default_factory=lambda: LLMTierSettings(provider="ollama", model="qwen3.5:latest")
    )
    # purpose -> {"primary": {...}, "fallback": {...}}; only overridden fields
    # need to be present, see llm/router.py ModelRouter._resolve_tier.
    purposes: Dict[str, Any] = Field(default_factory=dict)
    groq: LLMGroqSettings = Field(default_factory=LLMGroqSettings)
    ollama: LLMOllamaSettings = Field(default_factory=LLMOllamaSettings)


class SensorFocusSettings(BaseModel):
    enabled: bool = False
    poll_interval_seconds: float = 5.0


class SensorCalendarSettings(BaseModel):
    enabled: bool = False
    poll_interval_seconds: float = 300.0
    lookahead_minutes: int = 120


class SensorInboxSettings(BaseModel):
    """Polls unread count via the Gmail MCP bridge's list_unread tool —
    requires mcp.servers.gmail.enabled, not a direct Gmail connection."""

    enabled: bool = False
    poll_interval_seconds: float = 600.0
    surge_threshold: int = 5
    surge_cooldown_min: int = 60


class SensorsSettings(BaseModel):
    focus: SensorFocusSettings = Field(default_factory=SensorFocusSettings)
    calendar: SensorCalendarSettings = Field(default_factory=SensorCalendarSettings)
    inbox: SensorInboxSettings = Field(default_factory=SensorInboxSettings)


class ProactiveContextSwitchRuleSettings(BaseModel):
    enabled: bool = True
    threshold: int = 3
    window_min: int = 30
    cooldown_min: int = 90
    work_apps: List[str] = Field(
        default_factory=lambda: [
            "Visual Studio Code", "Terminal", "Xcode", "Slack", "Safari", "Google Chrome", "Mail",
        ]
    )


class ProactiveMeetingReminderRuleSettings(BaseModel):
    enabled: bool = True
    cooldown_min: int = 10


class ProactiveRulesSettings(BaseModel):
    context_switch: ProactiveContextSwitchRuleSettings = Field(
        default_factory=ProactiveContextSwitchRuleSettings
    )
    meeting_reminder: ProactiveMeetingReminderRuleSettings = Field(
        default_factory=ProactiveMeetingReminderRuleSettings
    )


class ProactiveMorningBriefingScheduleSettings(BaseModel):
    enabled: bool = True
    time: str = "08:30"
    # P09: whether the briefing also gathers a day-plan section (calendar +
    # reminders + Notion tasks + email pressure) for the same planner.run()
    # call to render alongside inbox/calendar triage.
    include_day_plan: bool = True


class ProactiveMidnightReflectionScheduleSettings(BaseModel):
    """Daily memory-reflection pass — see proactive/reflection.py."""

    enabled: bool = True
    time: str = "00:00"


class ProactiveScheduleSettings(BaseModel):
    morning_briefing: ProactiveMorningBriefingScheduleSettings = Field(
        default_factory=ProactiveMorningBriefingScheduleSettings
    )
    midnight_reflection: ProactiveMidnightReflectionScheduleSettings = Field(
        default_factory=ProactiveMidnightReflectionScheduleSettings
    )


class ProactiveSettings(BaseModel):
    rules: ProactiveRulesSettings = Field(default_factory=ProactiveRulesSettings)
    schedule: ProactiveScheduleSettings = Field(default_factory=ProactiveScheduleSettings)


class TracingSettings(BaseModel):
    """LangSmith instrumentation, with an always-on local JSONL fallback
    (see tracing/tracer.py). LANGSMITH_API_KEY comes from the environment,
    not this config — if unset, tracing degrades to local-only."""

    enabled: bool = True
    project_name: str = "vesper"
    local_dir: str = "data/traces"


class MCPGmailServerSettings(BaseModel):
    """See mcp_servers/gmail/ — runs under its own Python 3.10+ venv."""

    enabled: bool = False
    command: str = "mcp_servers/.venv/bin/python3"
    args: List[str] = Field(default_factory=lambda: ["mcp_servers/gmail/server.py"])
    tiers: Dict[str, str] = Field(
        default_factory=lambda: {
            "list_unread": "safe",
            "get_message": "safe",
            "search": "safe",
            "summarize_thread": "safe",
            "unread_count": "safe",
            "draft_reply": "confirm",
            "archive": "confirm",
            "mark_read": "confirm",
        }
    )
    slow_tools: List[str] = Field(default_factory=lambda: ["summarize_thread"])


class MCPAppleServerSettings(BaseModel):
    """See mcp_servers/apple_pim/ — Calendar + Reminders via EventKit."""

    enabled: bool = False
    command: str = "mcp_servers/.venv/bin/python3"
    args: List[str] = Field(default_factory=lambda: ["mcp_servers/apple_pim/server.py"])
    tiers: Dict[str, str] = Field(
        default_factory=lambda: {
            "today_events": "safe",
            "upcoming": "safe",
            "reminders_due": "safe",
            "add_reminder": "confirm",
            "create_event": "confirm",
        }
    )
    slow_tools: List[str] = Field(default_factory=list)


class MCPNotionServerSettings(BaseModel):
    """See mcp_servers/notion/ — read-only, requires NOTION_API_KEY in the environment."""

    enabled: bool = False
    command: str = "mcp_servers/.venv/bin/python3"
    args: List[str] = Field(default_factory=lambda: ["mcp_servers/notion/server.py"])
    tiers: Dict[str, str] = Field(
        default_factory=lambda: {
            "notion_search": "safe",
            "notion_get_page": "safe",
            "notion_get_database_rows": "safe",
        }
    )
    slow_tools: List[str] = Field(default_factory=list)


class MCPServersSettings(BaseModel):
    gmail: MCPGmailServerSettings = Field(default_factory=MCPGmailServerSettings)
    apple_pim: MCPAppleServerSettings = Field(default_factory=MCPAppleServerSettings)
    notion: MCPNotionServerSettings = Field(default_factory=MCPNotionServerSettings)


class MCPSettings(BaseModel):
    servers: MCPServersSettings = Field(default_factory=MCPServersSettings)


class AppSettings(BaseSettings):
    """Application settings loaded from YAML + environment."""

    model_config = SettingsConfigDict(
        env_prefix="VESPER_",
        env_nested_delimiter="__",
        extra="allow",
    )

    settings_file: str = str(DEFAULT_SETTINGS_PATH)

    general: GeneralSettings = Field(default_factory=GeneralSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    intent: IntentSettings = Field(default_factory=IntentSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    event_bus: EventBusSettings = Field(default_factory=EventBusSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)
    ui: UISettings = Field(default_factory=UISettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    sensors: SensorsSettings = Field(default_factory=SensorsSettings)
    proactive: ProactiveSettings = Field(default_factory=ProactiveSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = os.getenv("VESPER_CONFIG", str(DEFAULT_SETTINGS_PATH))
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=Path(yaml_file)),
            file_secret_settings,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Compatibility helper for existing dict-based config usage."""
        data = self.model_dump(mode="python")
        data.pop("settings_file", None)
        return data


def load_settings(config_path: Optional[str] = None) -> AppSettings:
    """Load settings, honoring explicit path, env override, and defaults."""
    path = config_path or os.getenv("VESPER_CONFIG") or str(DEFAULT_SETTINGS_PATH)
    return AppSettings(settings_file=path)


def load_config_dict(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Backward-compatible dict settings for existing orchestration code."""
    settings = load_settings(config_path=config_path)
    return settings.as_dict()
