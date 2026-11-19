"""Pure Apple Silicon ONNX streaming engine for openWakeWord + 3-sample personalized voice."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Literal, Protocol

import numpy as np
import onnxruntime as ort

from backend.config import AppConfig, WakeWordConfig, get_config
from backend.logger import get_logger, stage_timer
from backend.memory.models import utc_now_iso
from backend.wakeword.enrollment import (
    PersonalizedVoiceProfile,
    extract_template_from_trace,
)
from backend.wakeword.events import (
    WakeWordDetectionEvent,
    WakeWordEngineInfo,
    WakeWordError,
)
from backend.wakeword.models import (
    check_wakeword_models,
    ensure_wakeword_models_exist,
    wakeword_disk_size_mb,
)

# Official openWakeWord v0.5.1 streaming feature constants (16 kHz mono PCM).
OWW_SAMPLE_RATE = 16000
OWW_STEP_SAMPLES = 1280  # 80 ms hop
OWW_MEL_CONTEXT_SAMPLES = 480  # 30 ms lookback (160 * 3)

OWW_MEL_WINDOW_FRAMES = 76
OWW_MEL_BINS = 32
OWW_EMBEDDING_FRAMES = 16
OWW_EMBEDDING_DIM = 96


class WakeWordDetector(Protocol):
    """Protocol implemented by local streaming wake-word detectors and test stubs."""

    def load(self) -> float:
        """Load local wake-word model artifacts and return load time in ms."""

    def reset(self) -> None:
        """Reset streaming feature buffers and patience counters."""

    def process_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int = OWW_SAMPLE_RATE,
    ) -> WakeWordDetectionEvent | None:
        """Consume a mono float32 chunk and return a WakeWordDetectionEvent if triggered."""

    def set_config(self, config: AppConfig) -> None:
        """Apply updated runtime wake-word sensitivity/threshold configuration."""

    def info(self) -> WakeWordEngineInfo:
        """Return live runtime telemetry and disk availability for the engine."""


class OpenWakeWordEngine:
    """Thread-safe local ONNX openWakeWord streaming engine with 3-sample voice matching."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config if config is not None else get_config()
        self._ww_cfg: WakeWordConfig = self._config.wakeword
        self._logger = get_logger("localvox.wakeword", self._config)
        self._lock = threading.Lock()

        self._mel_sess: ort.InferenceSession | None = None
        self._emb_sess: ort.InferenceSession | None = None
        self._cls_sess: ort.InferenceSession | None = None
        self._mel_in_name: str = ""
        self._emb_in_name: str = ""
        self._cls_in_name: str = ""
        self._active_provider: str = "CPUExecutionProvider"

        self._load_time_ms: float = 0.0
        self._last_inference_ms: float | None = None
        self._last_score: float = 0.0
        self._last_personal_score: float = 0.0
        self._total_detections: int = 0
        self._last_detection_at: str | None = None
        self._last_trigger_monotonic: float = 0.0
        self._consecutive_hits: int = 0

        # Streaming audio & neural feature ring buffers (in-memory only; never written to disk).
        self._pending = np.empty(0, dtype=np.float32)
        self._raw_tail = np.zeros(OWW_MEL_CONTEXT_SAMPLES, dtype=np.float32)
        self._mel_buffer = np.ones((OWW_MEL_WINDOW_FRAMES, OWW_MEL_BINS), dtype=np.float32)
        self._emb_buffer = np.zeros((OWW_EMBEDDING_FRAMES, OWW_EMBEDDING_DIM), dtype=np.float32)
        self._baseline_embedding: np.ndarray | None = None
        self._recent_rms: deque[float] = deque(maxlen=12)
        self._mel_steps: deque[np.ndarray] = deque(maxlen=OWW_EMBEDDING_FRAMES)
        self._step_rms: deque[float] = deque(maxlen=OWW_EMBEDDING_FRAMES)

        self._profile: PersonalizedVoiceProfile | None = PersonalizedVoiceProfile.load(
            self._ww_cfg.personal_profile_path
        )

    @property
    def is_loaded(self) -> bool:
        """Return True when all three ONNX sessions are initialized in memory."""
        with self._lock:
            return (
                self._mel_sess is not None
                and self._emb_sess is not None
                and self._cls_sess is not None
            )

    def _resolve_providers(self) -> list[str]:
        available = set(ort.get_available_providers())
        if (
            self._ww_cfg.execution_provider == "coreml"
            and "CoreMLExecutionProvider" in available
        ):
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def set_config(self, config: AppConfig) -> None:
        """Update runtime wake-word thresholds and metadata without unloading ONNX weights."""
        with self._lock:
            prev_path = self._ww_cfg.personal_profile_path
            self._config = config
            self._ww_cfg = config.wakeword
            if self._ww_cfg.personal_profile_path != prev_path:
                self._profile = PersonalizedVoiceProfile.load(
                    self._ww_cfg.personal_profile_path
                )

    def get_personal_profile(self) -> PersonalizedVoiceProfile | None:
        """Return currently loaded personalized voice profile if enrolled."""
        with self._lock:
            return self._profile

    def set_personal_profile(self, profile: PersonalizedVoiceProfile | None) -> None:
        """Install or clear the active personalized 3-sample voice profile in memory."""
        with self._lock:
            self._profile = profile
            self._last_personal_score = 0.0

    def reload_personal_profile(self) -> bool:
        """Reload persisted voice profile from disk; returns True when loaded."""
        with self._lock:
            self._profile = PersonalizedVoiceProfile.load(self._ww_cfg.personal_profile_path)
            return self._profile is not None

    def last_scores(self) -> tuple[float, float, float]:
        """Return (generic_score, personal_score, effective_threshold) for live standby HUD."""
        with self._lock:
            personal_active = bool(self._ww_cfg.personal_enabled and self._profile is not None)
            if personal_active and self._ww_cfg.personal_only:
                thresh = self._ww_cfg.effective_personal_threshold()
            elif personal_active:
                thresh = min(
                    self._ww_cfg.effective_threshold(),
                    self._ww_cfg.effective_personal_threshold(),
                )
            else:
                thresh = self._ww_cfg.effective_threshold()
            return self._last_score, self._last_personal_score, thresh

    def load(self) -> float:
        """Load melspectrogram, embedding, and Hey Jarvis ONNX sessions locally."""
        with self._lock:
            if (
                self._mel_sess is not None
                and self._emb_sess is not None
                and self._cls_sess is not None
            ):
                return self._load_time_ms

            mel_path, emb_path, cls_path = ensure_wakeword_models_exist(self._ww_cfg)
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            providers = self._resolve_providers()

            with stage_timer(
                "wakeword_load",
                log_path=self._config.logging.latency_log_file,
            ) as rec:
                self._mel_sess = ort.InferenceSession(
                    str(mel_path),
                    sess_options=sess_opts,
                    providers=providers,
                )
                self._emb_sess = ort.InferenceSession(
                    str(emb_path),
                    sess_options=sess_opts,
                    providers=providers,
                )
                self._cls_sess = ort.InferenceSession(
                    str(cls_path),
                    sess_options=sess_opts,
                    providers=providers,
                )
                self._mel_in_name = self._mel_sess.get_inputs()[0].name
                self._emb_in_name = self._emb_sess.get_inputs()[0].name
                self._cls_in_name = self._cls_sess.get_inputs()[0].name
                used_providers = self._cls_sess.get_providers()
                self._active_provider = (
                    used_providers[0] if used_providers else "CPUExecutionProvider"
                )
                rec["model"] = self._ww_cfg.model_name
                rec["provider"] = self._active_provider

            self._load_time_ms = float(rec["latency_ms"])
            self._warm_baseline_locked()
            self._reset_buffers_locked(clear_cooldown=True)
            if self._profile is None and self._ww_cfg.personal_profile_path.exists():
                self._profile = PersonalizedVoiceProfile.load(
                    self._ww_cfg.personal_profile_path
                )
            self._logger.info(
                "openWakeWord ONNX loaded (model=%s, provider=%s, load_ms=%.2f, personalized=%s)",
                self._ww_cfg.model_name,
                self._active_provider,
                self._load_time_ms,
                self._profile is not None,
            )
            return self._load_time_ms

    def _warm_baseline_locked(self) -> None:
        if self._emb_sess is None:
            return
        baseline_mel = np.ones(
            (1, OWW_MEL_WINDOW_FRAMES, OWW_MEL_BINS, 1),
            dtype=np.float32,
        )
        emb = self._emb_sess.run(None, {self._emb_in_name: baseline_mel})[0]
        self._baseline_embedding = np.asarray(emb, dtype=np.float32).reshape(OWW_EMBEDDING_DIM)

    def _reset_buffers_locked(self, *, clear_cooldown: bool = False) -> None:
        self._pending = np.empty(0, dtype=np.float32)
        self._raw_tail = np.zeros(OWW_MEL_CONTEXT_SAMPLES, dtype=np.float32)
        self._mel_buffer = np.ones((OWW_MEL_WINDOW_FRAMES, OWW_MEL_BINS), dtype=np.float32)
        if self._baseline_embedding is not None:
            self._emb_buffer = np.tile(self._baseline_embedding, (OWW_EMBEDDING_FRAMES, 1))
        else:
            self._emb_buffer = np.zeros((OWW_EMBEDDING_FRAMES, OWW_EMBEDDING_DIM), dtype=np.float32)
        self._consecutive_hits = 0
        self._last_score = 0.0
        self._last_personal_score = 0.0
        self._recent_rms.clear()
        self._mel_steps.clear()
        self._step_rms.clear()
        if clear_cooldown:
            self._last_trigger_monotonic = 0.0

    def reset(self, *, clear_cooldown: bool = False) -> None:
        """Reset audio/feature sliding buffers and consecutive-hit counters."""
        with self._lock:
            self._reset_buffers_locked(clear_cooldown=clear_cooldown)

    def _infer_step_locked(self, step_samples: np.ndarray) -> tuple[float, float]:
        """Run one 80 ms (1280-sample) openWakeWord ONNX step; returns (score, infer_ms)."""
        assert self._mel_sess is not None
        assert self._emb_sess is not None
        assert self._cls_sess is not None

        t0 = time.perf_counter()
        pcm_window = np.concatenate([self._raw_tail, step_samples])
        self._raw_tail = step_samples[-OWW_MEL_CONTEXT_SAMPLES:].copy()

        pcm_int16_scale = np.clip(pcm_window * 32767.0, -32768.0, 32767.0).astype(np.float32)
        spec_raw = self._mel_sess.run(
            None,
            {self._mel_in_name: pcm_int16_scale[None, :]},
        )[0]
        spec = np.squeeze(spec_raw)
        if spec.ndim == 1:
            spec = spec[None, :]
        spec = (spec / 10.0 + 2.0).astype(np.float32, copy=False)

        n_frames = min(spec.shape[0], OWW_MEL_WINDOW_FRAMES)
        if n_frames > 0:
            self._mel_buffer = np.roll(self._mel_buffer, -n_frames, axis=0)
            self._mel_buffer[-n_frames:, :] = spec[-n_frames:, :OWW_MEL_BINS]
            self._mel_steps.append(np.mean(self._mel_buffer[-n_frames:], axis=0).astype(np.float32))
        else:
            self._mel_steps.append(np.ones(OWW_MEL_BINS, dtype=np.float32))

        emb_raw = self._emb_sess.run(
            None,
            {self._emb_in_name: self._mel_buffer[None, :, :, None]},
        )[0]
        emb = np.asarray(emb_raw, dtype=np.float32).reshape(OWW_EMBEDDING_DIM)
        self._emb_buffer = np.roll(self._emb_buffer, -1, axis=0)
        self._emb_buffer[-1, :] = emb

        score_raw = self._cls_sess.run(
            None,
            {self._cls_in_name: self._emb_buffer[None, :, :]},
        )[0]
        score = float(np.clip(np.asarray(score_raw).reshape(-1)[0], 0.0, 1.0))
        infer_ms = (time.perf_counter() - t0) * 1000.0
        self._last_inference_ms = round(infer_ms, 3)
        self._last_score = round(score, 4)
        return score, infer_ms

    def _score_personal_locked(self) -> float:
        """Compute live personalized 3-sample voice match score in < 0.01 ms."""
        if (
            not self._ww_cfg.personal_enabled
            or self._profile is None
            or self._baseline_embedding is None
            or len(self._step_rms) < 4
        ):
            self._last_personal_score = 0.0
            return 0.0

        centered_win = self._emb_buffer - self._baseline_embedding
        mel_arr = np.asarray(list(self._mel_steps), dtype=np.float32)
        rms_arr = np.asarray(list(self._step_rms), dtype=np.float32)
        p_score = self._profile.score_window(centered_win, mel_arr, rms_arr)
        self._last_personal_score = round(p_score, 4)
        return p_score

    def extract_enrollment_sample(
        self,
        waveform: np.ndarray,
        sample_rate: int = OWW_SAMPLE_RATE,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Extract normalized (1536,) template + (32,) timbre vector without keeping raw audio."""
        if sample_rate != OWW_SAMPLE_RATE:
            raise WakeWordError(
                f"openWakeWord requires {OWW_SAMPLE_RATE} Hz mono audio, got {sample_rate} Hz."
            )
        if not self.is_loaded:
            self.load()

        arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if arr.size < OWW_STEP_SAMPLES:
            raise ValueError("Voice sample is too short. Please say the wake phrase clearly.")

        # Pad with 0.5 s leading silence and 1.0 s trailing silence so the 760 ms embedding
        # receptive window captures the entire wake phrase and its trailing phonetic contour.
        padded = np.concatenate(
            [np.zeros(8000, dtype=np.float32), arr, np.zeros(16000, dtype=np.float32)]
        )
        centered_embs: list[np.ndarray] = []
        step_mels: list[np.ndarray] = []
        step_rms: list[float] = []
        generic_scores: list[float] = []

        with self._lock:
            assert self._baseline_embedding is not None
            base = self._baseline_embedding
            self._reset_buffers_locked(clear_cooldown=True)
            for pos in range(0, padded.size - OWW_STEP_SAMPLES + 1, OWW_STEP_SAMPLES):
                step = padded[pos : pos + OWW_STEP_SAMPLES]
                score, _ = self._infer_step_locked(step)
                centered_embs.append(self._emb_buffer[-1].copy() - base)
                step_mels.append(self._mel_steps[-1].copy())
                step_rms.append(float(np.sqrt(np.mean(np.square(step, dtype=np.float64)))))
                generic_scores.append(score)
            self._reset_buffers_locked(clear_cooldown=True)

        template, timbre = extract_template_from_trace(
            np.asarray(centered_embs, dtype=np.float32),
            np.asarray(step_mels, dtype=np.float32),
            np.asarray(step_rms, dtype=np.float32),
        )
        peak_generic = round(max(generic_scores) if generic_scores else 0.0, 4)
        return template, timbre, peak_generic

    def process_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int = OWW_SAMPLE_RATE,
    ) -> WakeWordDetectionEvent | None:
        """Process arbitrary-length mono float32 audio (e.g., 512 samples) in 80 ms steps."""
        if sample_rate != OWW_SAMPLE_RATE:
            raise WakeWordError(
                f"openWakeWord requires {OWW_SAMPLE_RATE} Hz mono audio, got {sample_rate} Hz."
            )

        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None

        if not self.is_loaded:
            self.load()

        with self._lock:
            self._pending = (
                np.concatenate([self._pending, arr]) if self._pending.size else arr.copy()
            )
            generic_thresh = self._ww_cfg.effective_threshold()
            personal_thresh = self._ww_cfg.effective_personal_threshold()
            patience = max(1, self._ww_cfg.patience_frames)
            min_rms = self._ww_cfg.min_audio_rms
            cooldown_s = self._ww_cfg.cooldown_s
            personal_active = bool(self._ww_cfg.personal_enabled and self._profile is not None)
            personal_only = bool(personal_active and self._ww_cfg.personal_only)

            detected_event: WakeWordDetectionEvent | None = None
            while self._pending.size >= OWW_STEP_SAMPLES:
                step = self._pending[:OWW_STEP_SAMPLES]
                self._pending = self._pending[OWW_STEP_SAMPLES:]

                step_rms = float(np.sqrt(np.mean(np.square(step, dtype=np.float64))))
                self._recent_rms.append(step_rms)
                self._step_rms.append(step_rms)
                window_peak_rms = max(self._recent_rms) if self._recent_rms else step_rms
                generic_score, infer_ms = self._infer_step_locked(step)
                personal_score = self._score_personal_locked()

                now_mono = time.monotonic()
                in_cooldown = (
                    self._last_trigger_monotonic > 0.0
                    and (now_mono - self._last_trigger_monotonic) < cooldown_s
                )

                if in_cooldown:
                    self._consecutive_hits = 0
                    continue

                g_hit = (not personal_only) and (generic_score >= generic_thresh)
                p_hit = personal_active and (personal_score >= personal_thresh)
                hit = g_hit or p_hit

                if hit and window_peak_rms >= min_rms:
                    self._consecutive_hits += 1
                    # A strong personalized match aligns tightly around a 1-2 frame peak; allow
                    # immediate trigger if personalized match clears threshold by >= 0.05 margin.
                    required_hits = (
                        1
                        if (p_hit and personal_score >= min(0.95, personal_thresh + 0.05))
                        else patience
                    )
                    if self._consecutive_hits >= required_hits:
                        self._last_trigger_monotonic = now_mono
                        self._total_detections += 1
                        ts = utc_now_iso()
                        self._last_detection_at = ts

                        matched_by: Literal[
                            "openwakeword_onnx", "personalized_voice", "hybrid"
                        ]
                        if g_hit and p_hit:
                            matched_by = "hybrid"
                            eff_conf = max(generic_score, personal_score)
                            eff_thresh = min(generic_thresh, personal_thresh)
                        elif p_hit:
                            matched_by = "personalized_voice"
                            eff_conf = personal_score
                            eff_thresh = personal_thresh
                        else:
                            matched_by = "openwakeword_onnx"
                            eff_conf = generic_score
                            eff_thresh = generic_thresh

                        detected_event = WakeWordDetectionEvent(
                            wake_phrase=self._ww_cfg.wake_phrase,
                            model_name=self._ww_cfg.model_name,
                            confidence=round(eff_conf, 4),
                            threshold=round(eff_thresh, 4),
                            inference_ms=round(infer_ms, 3),
                            audio_rms=round(window_peak_rms, 5),
                            matched_by=matched_by,
                            generic_confidence=round(generic_score, 4),
                            personal_confidence=(
                                round(personal_score, 4) if personal_active else None
                            ),
                            timestamp=ts,
                            detected_at_monotonic=now_mono,
                        )
                        # Flush feature buffers so trailing wake tail never double-fires.
                        self._reset_buffers_locked(clear_cooldown=False)
                        break
                else:
                    self._consecutive_hits = 0

            return detected_event

    def score_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int = OWW_SAMPLE_RATE,
        *,
        chunk_size: int = 512,
    ) -> tuple[float, list[float], list[float]]:
        """Run isolated offline evaluation over a complete waveform; returns (peak, scores, ms)."""
        if sample_rate != OWW_SAMPLE_RATE:
            raise WakeWordError(
                f"openWakeWord requires {OWW_SAMPLE_RATE} Hz mono audio, got {sample_rate} Hz."
            )
        if not self.is_loaded:
            self.load()

        arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
        scores: list[float] = []
        latencies: list[float] = []
        with self._lock:
            self._reset_buffers_locked(clear_cooldown=True)
            for pos in range(0, arr.size, chunk_size):
                piece = arr[pos : pos + chunk_size]
                self._pending = (
                    np.concatenate([self._pending, piece])
                    if self._pending.size
                    else piece.copy()
                )
                while self._pending.size >= OWW_STEP_SAMPLES:
                    step = self._pending[:OWW_STEP_SAMPLES]
                    self._pending = self._pending[OWW_STEP_SAMPLES:]
                    step_rms = float(np.sqrt(np.mean(np.square(step, dtype=np.float64))))
                    self._step_rms.append(step_rms)
                    score, infer_ms = self._infer_step_locked(step)
                    scores.append(score)
                    latencies.append(infer_ms)
            self._reset_buffers_locked(clear_cooldown=True)
        peak = max(scores) if scores else 0.0
        return peak, scores, latencies

    def info(self) -> WakeWordEngineInfo:
        """Return live engine telemetry and local ONNX artifact readiness."""
        with self._lock:
            loaded = (
                self._mel_sess is not None
                and self._emb_sess is not None
                and self._cls_sess is not None
            )
            models_ok = check_wakeword_models(self._ww_cfg)
            profile = self._profile
            return WakeWordEngineInfo(
                engine="openwakeword-onnx",
                wake_phrase=self._ww_cfg.wake_phrase,
                model_name=self._ww_cfg.model_name,
                model_path=str(self._ww_cfg.classifier_model),
                loaded=loaded,
                models_present=models_ok,
                disk_size_mb=wakeword_disk_size_mb(self._ww_cfg) if models_ok else None,
                execution_provider=self._active_provider,
                last_inference_ms=self._last_inference_ms,
                last_score=self._last_score,
                last_personal_score=self._last_personal_score,
                total_detections=self._total_detections,
                last_detection_at=self._last_detection_at,
                personalized_enrolled=profile is not None,
                personalized_samples=profile.sample_count if profile is not None else 0,
                personal_only=bool(profile is not None and self._ww_cfg.personal_only),
            )
