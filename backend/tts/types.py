"""Shared TTS result models and exceptions."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class UnsupportedLanguageError(ValueError):
    """Raised when a TTS engine does not support the requested language."""


class TtsResult(BaseModel):
    """Output metrics and artifact path from TTS synthesis."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    model_name: str
    voice: str
    language: str
    output_path: Path
    sample_rate: int
    audio_duration_s: float
    load_time_ms: float
    processing_time_ms: float
    rtf: float | None
