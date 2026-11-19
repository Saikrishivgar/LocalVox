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
    input_device: int | str | None = None
    output_device: int | str | None = None


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
    wakeword_models_dir: Path = Field(default=Path("data/models/wakeword"))
    logs_dir: Path
    latency_log_file: Path
    app_log_file: Path
    benchmarks_dir: Path
    test_audio_dir: Path
    tts_output_dir: Path
    memory_db_file: Path
    conversations_db_file: Path


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
    repeat_penalty: float = Field(ge=0.5, le=2.0)
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


class BargeInProfileConfig(BaseModel):
    """Sensitivity scaling profile applied to base barge-in thresholds."""

    model_config = ConfigDict(extra="forbid")

    vad_threshold_offset: float = Field(ge=-0.4, le=0.4)
    rms_scale: float = Field(gt=0.0, le=5.0)
    speech_frames_offset: int = Field(ge=-6, le=12)


class PipelineConfig(BaseModel):
    """End-to-end voice pipeline, endpointing, streaming TTS, and barge-in configuration."""

    model_config = ConfigDict(extra="forbid")

    pre_speech_pad_ms: int = Field(ge=0)
    post_speech_pad_ms: int = Field(ge=0)
    vad_hangover_ms: int = Field(ge=0)
    min_utterance_ms: int = Field(ge=0)
    max_utterance_s: float = Field(gt=0.0)
    min_audio_rms: float = Field(ge=0.0)
    listen_timeout_s: float | None = Field(default=None, gt=0.0)
    max_history_turns: int = Field(gt=0)
    max_history_chars: int = Field(gt=0)
    tts_routing: Literal["auto", "kokoro", "piper"]
    stream_llm_output: bool
    streaming_tts_enabled: bool = True
    tts_min_sentence_chars: int = Field(default=16, ge=1)
    tts_max_chunk_chars: int = Field(default=220, ge=20)
    tts_queue_maxsize: int = Field(default=8, ge=1)
    tts_chunk_gap_ms: int = Field(default=35, ge=0)
    barge_in_enabled: bool = True
    barge_in_sensitivity: Literal["low", "medium", "high"] = "medium"
    barge_in_vad_threshold: float = Field(default=0.72, gt=0.0, lt=1.0)
    barge_in_min_speech_ms: int = Field(default=224, ge=32)
    barge_in_min_rms: float = Field(default=0.015, ge=0.0)
    barge_in_echo_ratio: float = Field(default=1.4, gt=0.0)
    echo_suppression_enabled: bool = True
    echo_max_delay_ms: int = Field(default=350, ge=32)
    echo_min_corr: float = Field(default=0.35, ge=0.0, le=1.0)
    echo_reference_window_ms: int = Field(default=250, ge=32)
    echo_tail_hold_ms: int = Field(default=200, ge=0)
    barge_in_profiles: dict[Literal["low", "medium", "high"], BargeInProfileConfig]
    voice_system_prompt: str = Field(min_length=1)
    clear_commands: list[str]
    exit_commands: list[str]

    def effective_barge_in_params(
        self, frame_ms: float = 32.0
    ) -> tuple[float, int, float, float]:
        """Return (vad_threshold, min_speech_frames, min_rms, echo_ratio) for active sensitivity."""
        profile = self.barge_in_profiles[self.barge_in_sensitivity]
        raw_thresh = self.barge_in_vad_threshold + profile.vad_threshold_offset
        vad_thresh = min(0.95, max(0.25, raw_thresh))
        base_frames = max(1, round(self.barge_in_min_speech_ms / frame_ms))
        min_frames = max(2, base_frames + profile.speech_frames_offset)
        min_rms = self.barge_in_min_rms * profile.rms_scale
        echo_ratio = self.barge_in_echo_ratio * profile.rms_scale
        return vad_thresh, min_frames, min_rms, echo_ratio


class MemoryConfig(BaseModel):
    """Persistent local SQLite + FTS5 conversational memory configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    db_path: Path
    auto_extract: bool = True
    async_extraction: bool = True
    max_retrieved_memories: int = Field(default=5, gt=0)
    max_context_chars: int = Field(default=1200, ge=100)
    min_relevance_score: float = Field(default=0.2, ge=0.0, le=1.0)
    min_importance_to_store: float = Field(default=0.35, ge=0.0, le=1.0)
    recency_half_life_days: float = Field(default=30.0, gt=0.0)
    weight_lexical: float = Field(default=0.50, ge=0.0, le=1.0)
    weight_importance: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_recency: float = Field(default=0.20, ge=0.0, le=1.0)
    privacy_filter_enabled: bool = True
    extraction_max_tokens: int = Field(default=384, gt=32)
    extraction_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    pending_confirmation_timeout_s: float = Field(default=30.0, gt=0.0)


class WakeWordConfig(BaseModel):
    """Offline openWakeWord ONNX standby wake-word detection configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    wake_phrase: str = Field(default="Hey Jarvis", min_length=1, max_length=64)
    model_name: str = Field(default="hey_jarvis_v0.1", min_length=1, max_length=64)
    models_dir: Path = Field(default=Path("data/models/wakeword"))
    melspectrogram_model: Path = Field(default=Path("data/models/wakeword/melspectrogram.onnx"))
    embedding_model: Path = Field(default=Path("data/models/wakeword/embedding_model.onnx"))
    classifier_model: Path = Field(default=Path("data/models/wakeword/hey_jarvis_v0.1.onnx"))
    sensitivity: Literal["low", "medium", "high"] = "medium"
    threshold: float = Field(default=0.5, ge=0.1, le=0.95)
    patience_frames: int = Field(default=2, ge=1, le=10)
    cooldown_s: float = Field(default=2.0, ge=0.2, le=15.0)
    min_audio_rms: float = Field(default=0.003, ge=0.0, le=0.5)
    ack_enabled: bool = True
    ack_text: str = Field(default="Yes, I'm listening.", min_length=1, max_length=160)
    activation_grace_ms: int = Field(default=350, ge=0, le=2000)
    pre_roll_ms: int = Field(default=320, ge=0, le=1500)
    conversation_timeout_s: float = Field(default=15.0, ge=2.0, le=180.0)
    execution_provider: Literal["cpu", "coreml"] = "cpu"
    personal_profile_path: Path = Field(
        default=Path("data/models/wakeword/voice_profile.npz")
    )
    personal_enabled: bool = True
    personal_only: bool = False
    personal_threshold: float = Field(default=0.68, ge=0.45, le=0.95)

    def effective_threshold(self) -> float:
        """Compute active confidence threshold combining base threshold and sensitivity preset."""
        offsets = {"low": 0.15, "medium": 0.0, "high": -0.15}
        return round(min(0.95, max(0.1, self.threshold + offsets[self.sensitivity])), 4)

    def effective_personal_threshold(self) -> float:
        """Compute active personalized voice match threshold adjusted by sensitivity preset."""
        offsets = {"low": 0.08, "medium": 0.0, "high": -0.07}
        return round(
            min(0.95, max(0.50, self.personal_threshold + offsets[self.sensitivity])),
            4,
        )


class ServerConfig(BaseModel):
    """Local desktop FastAPI & WebSocket host configuration."""

    model_config = ConfigDict(extra="forbid")

    host: Literal["127.0.0.1", "localhost"] = "127.0.0.1"
    port: int = Field(default=8000, ge=1024, le=65535)
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://127.0.0.1:3000", "http://localhost:3000"]
    )


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
    pipeline: PipelineConfig
    memory: MemoryConfig
    wakeword: WakeWordConfig = Field(default_factory=WakeWordConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


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
