"""Wake-word event models, metadata schemas, and exceptions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.memory.models import utc_now_iso


class WakeWordError(RuntimeError):
    """Base error for offline wake-word engine operations."""


class WakeWordModelMissingError(WakeWordError):
    """Raised when required local openWakeWord ONNX artifacts are missing."""


class WakeWordDetectionEvent(BaseModel):
    """Validated wake-word trigger payload emitted when confidence crosses threshold."""

    model_config = ConfigDict(extra="forbid")

    wake_phrase: str
    model_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    inference_ms: float = Field(ge=0.0)
    audio_rms: float = Field(ge=0.0)
    matched_by: Literal["openwakeword_onnx", "personalized_voice", "hybrid"] = "openwakeword_onnx"
    generic_confidence: float | None = None
    personal_confidence: float | None = None
    timestamp: str = Field(default_factory=utc_now_iso)
    detected_at_monotonic: float = 0.0


class WakeWordEngineInfo(BaseModel):
    """Real runtime and disk status for the local ONNX wake-word engine."""

    model_config = ConfigDict(extra="forbid")

    engine: str = "openwakeword-onnx"
    wake_phrase: str
    model_name: str
    model_path: str
    loaded: bool = False
    models_present: bool = False
    disk_size_mb: float | None = None
    execution_provider: str = "CPUExecutionProvider"
    last_inference_ms: float | None = None
    last_score: float = 0.0
    last_personal_score: float = 0.0
    total_detections: int = 0
    last_detection_at: str | None = None
    personalized_enrolled: bool = False
    personalized_samples: int = 0
    personal_only: bool = False
