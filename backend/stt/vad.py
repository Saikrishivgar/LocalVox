"""Silero Voice Activity Detection (VAD) wrapper for LocalVox."""

import math
from pathlib import Path

import numpy as np
import silero_vad
import soundfile as sf
import torch
from pydantic import BaseModel, ConfigDict
from scipy.signal import resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad

from backend.config import AppConfig, get_config
from backend.eval.benchmark import compute_rtf, get_disk_size_mb
from backend.logger import get_logger, stage_timer


class VadSegment(BaseModel):
    """Detected speech interval in seconds."""

    model_config = ConfigDict(extra="forbid")

    start_s: float
    end_s: float


class VadResult(BaseModel):
    """Output metrics and speech regions from VAD inference."""

    model_config = ConfigDict(extra="forbid")

    speech_detected: bool
    segments: list[VadSegment]
    audio_duration_s: float
    load_time_ms: float
    processing_time_ms: float
    rtf: float | None
    sample_rate: int
    model_name: str


def load_audio_mono(audio_path: Path, target_sample_rate: int) -> tuple[np.ndarray, float]:
    """Load an audio file as float32 mono waveform resampled to target_sample_rate."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty or invalid: {audio_path}")

    try:
        samples, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise ValueError(f"Failed to read audio file '{audio_path}': {exc}") from exc

    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if waveform.size == 0 or file_sr <= 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")
    if not np.all(np.isfinite(waveform)):
        raise ValueError(f"Audio file contains non-finite values: {audio_path}")



    duration_s = float(waveform.size / file_sr)
    if file_sr != target_sample_rate:
        divisor = math.gcd(int(file_sr), int(target_sample_rate))
        up = int(target_sample_rate) // divisor
        down = int(file_sr) // divisor
        waveform = resample_poly(waveform, up=up, down=down).astype(np.float32)

    return waveform, duration_s


class SileroVadDetector:
    """Offline Silero VAD wrapper configured via AppConfig."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config if config is not None else get_config()
        self._vad_cfg = self._config.vad
        self._logger = get_logger("localvox.vad", self._config)
        self._model: object | None = None
        self._load_time_ms: float = 0.0

    def model_size_mb(self) -> float | None:
        """Return the bundled Silero VAD model file size in MiB."""
        data_dir = Path(silero_vad.__file__).parent / "data"
        filename = "silero_vad.onnx" if self._vad_cfg.backend == "onnx" else "silero_vad.jit"
        return get_disk_size_mb(data_dir / filename)

    def load(self) -> float:
        """Load the bundled Silero VAD model locally without network access."""
        if self._model is not None:
            return self._load_time_ms

        use_onnx = self._vad_cfg.backend == "onnx"
        if not use_onnx:
            torch.set_num_threads(1)

        with stage_timer("vad_load", log_path=self._config.logging.latency_log_file) as rec:
            self._model = load_silero_vad(onnx=use_onnx)
            rec["backend"] = self._vad_cfg.backend
            rec["model"] = self._vad_cfg.model_name

        self._load_time_ms = float(rec["latency_ms"])
        self._logger.info(
            "Silero VAD loaded (backend=%s, load_ms=%.2f)",
            self._vad_cfg.backend,
            self._load_time_ms,
        )
        return self._load_time_ms

    def detect(
        self,
        audio_path: Path | str,
        *,
        sample_rate: int | None = None,
        threshold: float | None = None,
        min_speech_duration_ms: int | None = None,
        min_silence_duration_ms: int | None = None,
    ) -> VadResult:
        """Detect speech intervals in a local audio file."""
        target_sr = sample_rate if sample_rate is not None else self._vad_cfg.sample_rate
        if target_sr not in (8000, 16000):
            raise ValueError(f"Silero VAD requires 8000 or 16000 Hz sample rate, got {target_sr}")

        prob_threshold = threshold if threshold is not None else self._vad_cfg.threshold
        if not (0.0 <= prob_threshold <= 1.0):
            raise ValueError(f"Speech threshold must be in [0.0, 1.0], got {prob_threshold}")

        min_speech_ms = (
            min_speech_duration_ms
            if min_speech_duration_ms is not None
            else self._vad_cfg.min_speech_duration_ms
        )
        min_silence_ms = (
            min_silence_duration_ms
            if min_silence_duration_ms is not None
            else self._vad_cfg.min_silence_duration_ms
        )

        load_ms = self.load()
        waveform, audio_duration_s = load_audio_mono(Path(audio_path), target_sr)
        tensor = torch.from_numpy(waveform)

        with stage_timer("vad", log_path=self._config.logging.latency_log_file) as rec:
            raw_timestamps = get_speech_timestamps(
                tensor,
                self._model,
                threshold=prob_threshold,
                sampling_rate=target_sr,
                min_speech_duration_ms=min_speech_ms,
                min_silence_duration_ms=min_silence_ms,
                speech_pad_ms=self._vad_cfg.speech_pad_ms,
                window_size_samples=self._vad_cfg.window_size_samples,
                return_seconds=True,
            )
            rec["model"] = self._vad_cfg.model_name
            rec["audio_duration_s"] = round(audio_duration_s, 3)

        processing_ms = float(rec["latency_ms"])
        segments = [
            VadSegment(start_s=round(float(seg["start"]), 3), end_s=round(float(seg["end"]), 3))
            for seg in raw_timestamps
        ]
        rtf = compute_rtf(processing_ms, audio_duration_s)
        self._logger.info(
            "VAD completed (speech=%s, segments=%d, latency_ms=%.2f, rtf=%s)",
            bool(segments),
            len(segments),
            processing_ms,
            rtf,
        )
        return VadResult(
            speech_detected=bool(segments),
            segments=segments,
            audio_duration_s=round(audio_duration_s, 3),
            load_time_ms=round(load_ms, 2),
            processing_time_ms=round(processing_ms, 2),
            rtf=rtf,
            sample_rate=target_sr,
            model_name=self._vad_cfg.model_name,
        )
