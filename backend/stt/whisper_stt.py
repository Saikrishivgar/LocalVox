"""MLX Whisper Speech-to-Text (STT) wrapper for LocalVox."""

import json
import math
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict
from scipy.signal import resample_poly

from backend.config import AppConfig, get_config
from backend.eval.benchmark import compute_rtf, get_disk_size_mb
from backend.logger import get_logger, stage_timer

if TYPE_CHECKING:
    import mlx.core as mx
    from mlx_whisper import whisper

SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".aiff", ".webm"}
)


class SttResult(BaseModel):
    """Transcription output and timing metrics from MLX Whisper."""

    model_config = ConfigDict(extra="forbid")

    text: str
    language: str | None
    audio_duration_s: float
    load_time_ms: float
    processing_time_ms: float
    rtf: float | None
    model_path: str
    segments: list[dict[str, Any]]


def probe_audio_duration_s(audio_path: Path) -> float:
    """Validate audio file format/readability and return duration in seconds."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty or invalid: {audio_path}")

    suffix = audio_path.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{suffix}'. "
            f"Supported: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    try:
        info = sf.info(str(audio_path))
        if info.frames > 0 and info.samplerate > 0:
            return float(info.duration)
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        duration = float(out)
        if duration > 0.0:
            return duration
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted audio file '{audio_path}': {exc}") from exc

    raise ValueError(f"Audio file has zero duration: {audio_path}")


def validate_local_model_dir(model_path: Path) -> Path:
    """Verify a local Whisper model directory exists with config and weights."""
    if not model_path.exists() or not model_path.is_dir():
        raise FileNotFoundError(
            f"Local Whisper model directory not found: {model_path}. "
            "Download models first before running offline STT inference."
        )
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in Whisper model directory: {model_path}")

    weight_candidates = ("weights.safetensors", "weights.npz", "model.safetensors")
    if not any((model_path / name).exists() for name in weight_candidates):
        raise FileNotFoundError(
            f"No model weight file ({weight_candidates}) found in {model_path}"
        )
    return model_path


def _load_whisper_into_holder(model_dir: Path, dtype: "mx.Dtype") -> "whisper.Whisper":
    """Load Whisper weights from local disk supporting both weights.* and model.safetensors."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mlx_whisper import whisper
    from mlx_whisper.transcribe import ModelHolder

    if ModelHolder.model is not None and ModelHolder.model_path == str(model_dir):
        return ModelHolder.model

    config_data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config_data.pop("model_type", None)
    quantization = config_data.pop("quantization", None)
    model_args = whisper.ModelDimensions(**config_data)

    weight_file = next(
        model_dir / name
        for name in ("weights.safetensors", "weights.npz", "model.safetensors")
        if (model_dir / name).exists()
    )
    weights = mx.load(str(weight_file))
    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        def predicate(path: str, module: nn.Module) -> bool:
            return isinstance(module, (nn.Linear, nn.Embedding)) and f"{path}.scales" in weights

        nn.quantize(model, **quantization, class_predicate=predicate)

    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    ModelHolder.model = model
    ModelHolder.model_path = str(model_dir)
    return model


class MlxWhisperStt:
    """Offline MLX Whisper wrapper for Apple Silicon GPU inference."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config if config is not None else get_config()
        self._stt_cfg = self._config.stt
        self._logger = get_logger("localvox.stt", self._config)
        self._load_times_ms: dict[str, float] = {}

    def model_size_mb(self, model_path: Path | str | None = None) -> float | None:
        """Return total disk size in MiB for the local Whisper model directory."""
        target = Path(model_path) if model_path is not None else self._stt_cfg.local_model_path
        return get_disk_size_mb(target)

    def load(self, model_path: Path | str | None = None) -> float:
        """Load local MLX Whisper weights into memory without network calls."""
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        resolved = validate_local_model_dir(
            Path(model_path) if model_path is not None else self._stt_cfg.local_model_path
        )
        key = str(resolved)
        if ModelHolder.model is not None and ModelHolder.model_path == key:
            return self._load_times_ms.get(key, 0.0)

        with stage_timer("stt_load", log_path=self._config.logging.latency_log_file) as rec:
            _load_whisper_into_holder(resolved, mx.float16)
            rec["model"] = resolved.name

        load_ms = float(rec["latency_ms"])
        self._load_times_ms[key] = load_ms
        self._logger.info("MLX Whisper loaded (model=%s, load_ms=%.2f)", resolved.name, load_ms)
        return load_ms

    def _run_transcribe(
        self,
        audio_input: str | np.ndarray,
        audio_duration_s: float,
        *,
        language: str | None = None,
        model_path: Path | str | None = None,
    ) -> SttResult:
        """Shared transcription path for local audio files and float32 arrays."""
        resolved_model = validate_local_model_dir(
            Path(model_path) if model_path is not None else self._stt_cfg.local_model_path
        )
        load_ms = self.load(resolved_model)

        import mlx_whisper

        chosen_lang = language if language is not None else self._stt_cfg.default_language
        if chosen_lang in ("", "auto"):
            chosen_lang = None

        decode_kwargs: dict[str, Any] = {
            "path_or_hf_repo": str(resolved_model),
            "temperature": self._stt_cfg.temperature,
            "condition_on_previous_text": self._stt_cfg.condition_on_previous_text,
            "fp16": True,
        }
        if chosen_lang is not None:
            decode_kwargs["language"] = chosen_lang

        with stage_timer("stt", log_path=self._config.logging.latency_log_file) as rec:
            start = time.perf_counter()
            raw: dict[str, Any] = mlx_whisper.transcribe(audio_input, **decode_kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            rec["model"] = resolved_model.name
            rec["language"] = raw.get("language") or chosen_lang

        processing_ms = round(elapsed_ms, 2)
        rtf = compute_rtf(processing_ms, audio_duration_s)
        text = str(raw.get("text", "")).strip()
        detected_lang = raw.get("language")

        self._logger.info(
            "STT completed (model=%s, lang=%s, dur_s=%.2f, latency_ms=%.2f, rtf=%s)",
            resolved_model.name,
            detected_lang,
            audio_duration_s,
            processing_ms,
            rtf,
        )
        return SttResult(
            text=text,
            language=str(detected_lang) if detected_lang is not None else None,
            audio_duration_s=round(audio_duration_s, 3),
            load_time_ms=round(load_ms, 2),
            processing_time_ms=processing_ms,
            rtf=rtf,
            model_path=str(resolved_model),
            segments=list(raw.get("segments", [])),
        )

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        language: str | None = None,
        model_path: Path | str | None = None,
    ) -> SttResult:
        """Transcribe a local audio file using MLX Whisper on Apple Silicon."""
        audio_file = Path(audio_path)
        audio_duration_s = probe_audio_duration_s(audio_file)
        return self._run_transcribe(
            str(audio_file),
            audio_duration_s,
            language=language,
            model_path=model_path,
        )

    def transcribe_array(
        self,
        waveform: np.ndarray,
        sample_rate: int | None = None,
        *,
        language: str | None = None,
        model_path: Path | str | None = None,
    ) -> SttResult:
        """Transcribe an in-memory float32 mono waveform without writing a file."""
        target_sr = self._stt_cfg.sample_rate
        source_sr = sample_rate if sample_rate is not None else target_sr
        if source_sr <= 0:
            raise ValueError(f"Sample rate must be positive, got {source_sr}")

        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1, dtype=np.float32)
        audio = audio.reshape(-1)
        if audio.size == 0:
            raise ValueError("Waveform for STT transcription must not be empty.")
        if not np.all(np.isfinite(audio)):
            raise ValueError("Waveform contains non-finite values.")

        audio_duration_s = float(audio.size / source_sr)
        if source_sr != target_sr:
            divisor = math.gcd(int(source_sr), int(target_sr))
            audio = resample_poly(
                audio,
                up=int(target_sr) // divisor,
                down=int(source_sr) // divisor,
            ).astype(np.float32)

        return self._run_transcribe(
            audio,
            audio_duration_s,
            language=language,
            model_path=model_path,
        )
