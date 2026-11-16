"""Configuration models and YAML loader for LocalVox."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_FILENAME = "config.yaml"
_CONFIG_ENV_VAR = "LOCALVOX_CONFIG"


class ModelsConfig(BaseModel):
    """Local model identifiers and endpoints."""

    model_config = ConfigDict(extra="forbid")

    vad: str
    stt: str
    llm: str
    ollama_base_url: str
    tts_en: str
    tts_ta: str
    embedding: str
    qdrant_collection: str


class AudioConfig(BaseModel):
    """Audio capture and playback parameters."""

    model_config = ConfigDict(extra="forbid")

    sample_rate: int
    channels: int
    chunk_size: int
    tts_sample_rate: int


class ThresholdsConfig(BaseModel):
    """Pipeline thresholds and latency budgets."""

    model_config = ConfigDict(extra="forbid")

    vad_speech_prob: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    rag_score_threshold: float
    rag_top_k: int
    max_stage_latency_ms: float


class PathsConfig(BaseModel):
    """Filesystem paths for local storage and logs."""

    model_config = ConfigDict(extra="forbid")

    data_dir: Path
    qdrant_dir: Path
    models_dir: Path
    logs_dir: Path
    latency_log_file: Path
    app_log_file: Path


class AppConfig(BaseModel):
    """Root application configuration."""

    model_config = ConfigDict(extra="forbid")

    models: ModelsConfig
    audio: AudioConfig
    thresholds: ThresholdsConfig
    paths: PathsConfig


def _resolve_config_path(config_path: Path | str | None) -> Path:
    """Resolve explicit, env-configured, or repository-root config path."""
    if config_path is not None:
        return Path(config_path)
    env_path = os.environ.get(_CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path)
    return _PROJECT_ROOT / _DEFAULT_CONFIG_FILENAME


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load and validate configuration from a YAML file."""
    resolved = _resolve_config_path(config_path)
    with resolved.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return cached application configuration loaded from config.yaml."""
    return load_config()
