"""Strict Pydantic API request and response schemas for LocalVox FastAPI routes."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.history import ConversationSessionDetail, ConversationSessionSummary
from backend.memory.models import MemoryCategory, MemoryRecord
from backend.wakeword.enrollment import EnrollmentSampleResult, VoiceProfileSummary
from backend.wakeword.events import WakeWordDetectionEvent, WakeWordEngineInfo


class HealthResponse(BaseModel):
    """Health probe payload."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    pipeline_state: str
    mic_active: bool


class LiveMessageItem(BaseModel):
    """Single message item in the active companion transcript."""

    model_config = ConfigDict(extra="forbid")

    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    language: str | None = None
    created_at: str
    metrics: dict[str, Any] | None = None


class StateSnapshotResponse(BaseModel):
    """Full live companion state payload for initial UI hydration."""

    model_config = ConfigDict(extra="forbid")

    pipeline_state: str
    mic_active: bool
    session_id: str
    streaming_partial: str
    messages: list[LiveMessageItem]
    last_metrics: dict[str, Any] | None = None
    wakeword_enabled: bool = False
    wakeword_active: bool = False
    wake_phrase: str = "Hey Jarvis"
    last_wake_event: WakeWordDetectionEvent | None = None
    wakeword_engine: WakeWordEngineInfo | None = None
    voice_profile: VoiceProfileSummary | None = None


class ChatRequest(BaseModel):
    """Desktop text chat request routed through VoicePipeline.process_text."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)
    speak: bool = True
    language: str | None = Field(default=None, max_length=16)


class ChatResponse(BaseModel):
    """Structured result of handling a desktop chat turn."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    response_text: str
    metrics: dict[str, Any] | None = None
    pipeline_state: str


class VoiceControlResponse(BaseModel):
    """Response from start/stop microphone and stop-speaking endpoints."""

    model_config = ConfigDict(extra="forbid")

    status: str
    pipeline_state: str
    mic_active: bool
    message: str


class MemoryCreateRequest(BaseModel):
    """Explicit memory insertion payload from the Memory Dashboard."""

    model_config = ConfigDict(extra="forbid")

    category: MemoryCategory
    subject: str = Field(default="", max_length=120)
    content: str = Field(min_length=1, max_length=2000)
    keywords: list[str] = Field(default_factory=list, max_length=20)
    importance: float = Field(default=0.8, ge=0.0, le=1.0)


class MemoryUpdateRequest(BaseModel):
    """Partial memory update payload from the Memory Dashboard."""

    model_config = ConfigDict(extra="forbid")

    category: MemoryCategory | None = None
    subject: str | None = Field(default=None, max_length=120)
    content: str | None = Field(default=None, min_length=1, max_length=2000)
    keywords: list[str] | None = Field(default=None, max_length=20)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)


class MemoryBulkDeleteRequest(BaseModel):
    """Confirmation guard required before wiping all persisted memories."""

    model_config = ConfigDict(extra="forbid")

    confirm_phrase: Literal["CLEAR_ALL_MEMORIES"]


class MemoryListResponse(BaseModel):
    """Memory dashboard list & category summary response."""

    model_config = ConfigDict(extra="forbid")

    total_count: int
    category_counts: dict[str, int]
    items: list[MemoryRecord]
    match_scores: dict[str, float] = Field(default_factory=dict)


class MemoryMutationResponse(BaseModel):
    """Response confirming create/update/delete database state."""

    model_config = ConfigDict(extra="forbid")

    status: str
    action: str
    record: MemoryRecord | None = None
    deleted_count: int = 0


class ConversationListResponse(BaseModel):
    """List of saved conversation sessions."""

    model_config = ConfigDict(extra="forbid")

    items: list[ConversationSessionSummary]
    active_session_id: str


class ConversationContinueResponse(BaseModel):
    """Response after loading a prior conversation into active ConversationManager."""

    model_config = ConfigDict(extra="forbid")

    detail: ConversationSessionDetail
    loaded_turns: int
    active_session_id: str


class EngineStatus(BaseModel):
    """Real status metadata for an individual local AI or storage component."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: Literal["ready", "loaded", "offline", "missing", "error"]
    detail: str
    model_name: str | None = None
    loaded: bool = False
    disk_size_mb: float | None = None
    memory_mb: float | None = None


class SystemMonitorResponse(BaseModel):
    """Real local hardware, AI engine, SQLite, extraction worker, and latency metrics."""

    model_config = ConfigDict(extra="forbid")

    hardware: str
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    process_rss_mb: float
    ollama_connected: bool
    qwen_available: bool
    available_ollama_models: list[str]
    engines: list[EngineStatus]
    sqlite_integrity: dict[str, Any]
    sqlite_size_mb: float | None
    extraction_worker: dict[str, Any]
    recent_latencies: list[dict[str, Any]]
    pipeline_state: str
    mic_active: bool


class SettingsPayload(BaseModel):
    """Validated user-configurable LocalVox desktop settings (paths untouched)."""

    model_config = ConfigDict(extra="forbid")

    kokoro_voice: str = Field(min_length=1, max_length=64)
    kokoro_speed: float = Field(ge=0.5, le=2.0)
    piper_length_scale: float = Field(ge=0.5, le=2.0)
    stt_default_language: str | None = Field(default=None, max_length=16)
    tts_routing: Literal["auto", "kokoro", "piper"]
    streaming_tts_enabled: bool
    barge_in_enabled: bool
    barge_in_sensitivity: Literal["low", "medium", "high"]
    llm_model_name: str = Field(min_length=1, max_length=128)
    llm_temperature: float = Field(ge=0.0, le=2.0)
    max_history_turns: int = Field(gt=0, le=50)
    memory_enabled: bool
    memory_auto_extract: bool
    memory_privacy_filter_enabled: bool
    wakeword_enabled: bool = True
    wake_phrase: str = Field(default="Hey Jarvis", min_length=1, max_length=64)
    wakeword_sensitivity: Literal["low", "medium", "high"] = "medium"
    wakeword_threshold: float = Field(default=0.5, ge=0.1, le=0.95)
    wakeword_conversation_timeout_s: float = Field(default=15.0, ge=2.0, le=180.0)
    wakeword_ack_enabled: bool = True
    wakeword_activation_grace_ms: int = Field(default=350, ge=0, le=2000)
    wakeword_personal_enabled: bool = True
    wakeword_personal_only: bool = False
    wakeword_personal_threshold: float = Field(default=0.72, ge=0.45, le=0.95)


class SettingsResponse(BaseModel):
    """Current desktop settings plus available local voices, models, and wakeword status."""

    model_config = ConfigDict(extra="forbid")

    settings: SettingsPayload
    available_kokoro_voices: list[str]
    available_ollama_models: list[str]
    wakeword_engine: WakeWordEngineInfo | None = None
    voice_profile: VoiceProfileSummary | None = None
    mic_active: bool = False


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ConversationContinueResponse",
    "ConversationListResponse",
    "EngineStatus",
    "EnrollmentSampleResult",
    "HealthResponse",
    "LiveMessageItem",
    "MemoryBulkDeleteRequest",
    "MemoryCreateRequest",
    "MemoryListResponse",
    "MemoryMutationResponse",
    "MemoryUpdateRequest",
    "SettingsPayload",
    "SettingsResponse",
    "StateSnapshotResponse",
    "SystemMonitorResponse",
    "VoiceControlResponse",
    "VoiceProfileSummary",
]
