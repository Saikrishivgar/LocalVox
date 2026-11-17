"""Configuration models and YAML loader for LocalVox."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

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

    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    tts_sample_rate: int = Field(gt=0)
    piper_sample_rate: int = Field(gt=0)


class ThresholdsConfig(BaseModel):
    """Pipeline thresholds and latency budgets."""

    model_config = ConfigDict(extra="forbid")

    vad_speech_prob: float = Field(ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(ge=0)
    min_silence_duration_ms: int = Field(ge=0)
    rag_score_threshold: float = Field(ge=0.0, le=1.0)
    rag_top_k: int = Field(gt=0)
    max_stage_latency_ms: float = Field(gt=0.0)


class PathsConfig(BaseModel):
    """Filesystem paths for local storage, models, benchmarks, and logs."""

    model_config = ConfigDict(extra="forbid")

    data_dir: Path
    qdrant_dir: Path
    models_dir: Path
    vad_models_dir: Path
    stt_models_dir: Path
    tts_models_dir: Path
    logs_dir: Path
    latency_log_file: Path
    app_log_file: Path
    benchmarks_dir: Path
    test_audio_dir: Path
    tts_output_dir: Path


class VadConfig(BaseModel):
    """Voice Activity Detection configuration."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    backend: Literal["onnx", "torch"]
    sample_rate: Literal[8000, 16000]
    threshold: float = Field(ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(ge=0)
    min_silence_duration_ms: int = Field(ge=0)
    speech_pad_ms: int = Field(ge=0)
    window_size_samples: int = Field(gt=0)


class SttConfig(BaseModel):
    """Speech-to-Text configuration."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    local_model_path: Path
    candidate_models: list[Path]
    backend: Literal["mlx"]
    device: str = Field(min_length=1)
    default_language: str | None
    supported_languages: list[str]
    sample_rate: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=1.0)
    condition_on_previous_text: bool


class LlmConfig(BaseModel):
    """Local LLM inference configuration."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    backend: Literal["ollama"]
    base_url: str = Field(min_length=1)
    timeout_s: float = Field(gt=0.0)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_tokens: int = Field(gt=0)
    num_ctx: int = Field(gt=0)
    think: bool
    keep_alive: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)


class KokoroTtsConfig(BaseModel):
    """Kokoro ONNX TTS engine configuration."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    model_path: Path
    voices_path: Path
    default_voice: str = Field(min_length=1)
    lang_code: str = Field(min_length=1)
    supported_languages: list[str]
    sample_rate: int = Field(gt=0)
    speed: float = Field(gt=0.0)


class PiperTtsConfig(BaseModel):
    """Piper ONNX TTS engine configuration."""

    model_config = ConfigDict(extra="forbid")

    en_model_name: str = Field(min_length=1)
    en_model_path: Path
    en_config_path: Path
    ta_model_name: str = Field(min_length=1)
    ta_model_path: Path
    ta_config_path: Path
    supported_languages: list[str]
    sample_rate: int = Field(gt=0)
    length_scale: float = Field(gt=0.0)


class TtsConfig(BaseModel):
    """Text-to-Speech configuration covering Kokoro and Piper."""

    model_config = ConfigDict(extra="forbid")

    default_backend: Literal["kokoro", "piper"]
    default_language: str = Field(min_length=1)
    output_dir: Path
    kokoro: KokoroTtsConfig
    piper: PiperTtsConfig


class BenchmarkConfig(BaseModel):
    """Benchmarking configuration."""

    model_config = ConfigDict(extra="forbid")

    hardware_name: str = Field(min_length=1)
    output_json_file: Path
    output_markdown_file: Path
    warmup_runs: int = Field(ge=0)
    benchmark_runs: int = Field(gt=0)
    max_stage_latency_ms: float = Field(gt=0.0)


class LoggingConfig(BaseModel):
    """Structured logging configuration."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    json_format: bool
    stdout: bool
    file: bool
    app_log_file: Path
    latency_log_file: Path


class AppConfig(BaseModel):
    """Root application configuration."""

    model_config = ConfigDict(extra="forbid")

    models: ModelsConfig
    audio: AudioConfig
    thresholds: ThresholdsConfig
    paths: PathsConfig
    vad: VadConfig
    stt: SttConfig
    llm: LlmConfig
    tts: TtsConfig
    benchmark: BenchmarkConfig
    logging: LoggingConfig


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
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return cached application configuration loaded from config.yaml."""
    return load_config()
