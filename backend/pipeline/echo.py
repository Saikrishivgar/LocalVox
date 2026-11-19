"""Playback audio reference tracking and lightweight software echo suppression."""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, resample_poly

from backend.config import AppConfig, get_config


def resample_mono(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1-D float32 mono audio cleanly when hardware rate differs from target rate."""
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if source_sr == target_sr or audio.size == 0:
        return audio
    divisor = math.gcd(int(source_sr), int(target_sr))
    up = int(target_sr) // divisor
    down = int(source_sr) // divisor
    return resample_poly(audio, up=up, down=down).astype(np.float32)


def compute_rms(waveform: np.ndarray) -> float:
    """Compute root-mean-square amplitude of a float32 waveform."""
    if waveform.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


class PlaybackReferenceBuffer:
    """Thread-safe circular reference of recently played speaker audio at target_sr."""

    def __init__(self, target_sr: int = 16000, capacity_ms: int = 750) -> None:
        self._target_sr = target_sr
        self._capacity_samples = max(target_sr // 4, int((capacity_ms / 1000.0) * target_sr))
        self._lock = threading.Lock()
        self._chunks: deque[np.ndarray] = deque()
        self._total_samples = 0
        self._last_active_monotonic = 0.0

    @property
    def target_sr(self) -> int:
        """Sample rate used for stored reference audio."""
        return self._target_sr

    def clear(self) -> None:
        """Reset stored reference audio and active playback timestamp."""
        with self._lock:
            self._chunks.clear()
            self._total_samples = 0
            self._last_active_monotonic = 0.0

    def push(self, waveform: np.ndarray, source_sr: int) -> None:
        """Append a newly written speaker playback block, resampled to target_sr."""
        mono = resample_mono(waveform, source_sr, self._target_sr)
        if mono.size == 0:
            return
        rms = compute_rms(mono)
        now = time.monotonic()
        with self._lock:
            self._chunks.append(mono)
            self._total_samples += mono.size
            while self._total_samples > self._capacity_samples and self._chunks:
                dropped = self._chunks.popleft()
                self._total_samples -= dropped.size
            if rms >= 1e-4:
                self._last_active_monotonic = now

    def snapshot(
        self,
        window_samples: int,
        recent_samples: int,
    ) -> tuple[np.ndarray, float, float]:
        """Return (reference_window, recent_rms, age_since_active_s) safely."""
        with self._lock:
            if not self._chunks or self._last_active_monotonic == 0.0:
                return np.empty(0, dtype=np.float32), 0.0, float("inf")
            joined = np.concatenate(list(self._chunks)).astype(np.float32, copy=False)
            age_s = max(0.0, time.monotonic() - self._last_active_monotonic)

        window = joined[-window_samples:] if joined.size > window_samples else joined
        recent = joined[-recent_samples:] if joined.size > recent_samples else joined
        return window, compute_rms(recent), age_s


@dataclass(frozen=True)
class EchoSuppressionResult:
    """Per-frame output and diagnostics from software echo suppression."""

    cleaned_frame: np.ndarray
    raw_rms: float
    residual_rms: float
    reference_rms: float
    correlation: float
    estimated_echo_gain: float
    is_likely_pure_echo: bool


class EchoSuppressor:
    """Lag-aligned software echo suppressor and double-talk classifier for barge-in.

    PortAudio on macOS does not provide kernel VoiceProcessingIO hardware AEC; this class
    combines normalized cross-correlation lag alignment, linear echo projection removal,
    and online speaker-to-mic coupling tracking to reject own-playback echo cleanly when
    enabled while preserving low-volume interruption sensitivity on headphones.
    """

    def __init__(
        self,
        reference: PlaybackReferenceBuffer,
        config: AppConfig | None = None,
    ) -> None:
        self._ref = reference
        self._config = config if config is not None else get_config()
        pipe = self._config.pipeline
        self._sr = self._config.vad.sample_rate
        self._frame_len = self._config.vad.window_size_samples
        frame_ms = (self._frame_len / self._sr) * 1000.0
        _, _, eff_min_rms, eff_echo_ratio = pipe.effective_barge_in_params(frame_ms)
        self._eff_min_rms = eff_min_rms
        self._eff_echo_ratio = eff_echo_ratio
        self._enabled = pipe.echo_suppression_enabled
        self._min_corr = pipe.echo_min_corr
        self._window_samples = max(
            self._frame_len * 2,
            int(((pipe.echo_max_delay_ms / 1000.0) * self._sr) + self._frame_len),
        )
        self._recent_samples = max(
            self._frame_len,
            int((pipe.echo_reference_window_ms / 1000.0) * self._sr),
        )
        self._tail_hold_s = pipe.echo_tail_hold_ms / 1000.0
        self._coupling_ema = 0.0

    def reset(self) -> None:
        """Reset online speaker-to-mic acoustic coupling state."""
        self._coupling_ema = 0.0

    def process_frame(self, mic_frame: np.ndarray) -> EchoSuppressionResult:
        """Analyze and echo-suppress one 16 kHz mono mic frame during playback."""
        frame = np.asarray(mic_frame, dtype=np.float32).reshape(-1)
        raw_rms = compute_rms(frame)
        ref_window, ref_rms, age_s = self._ref.snapshot(
            self._window_samples,
            self._recent_samples,
        )

        if ref_rms < 1e-4 or age_s > self._tail_hold_s:
            return EchoSuppressionResult(
                cleaned_frame=frame,
                raw_rms=raw_rms,
                residual_rms=raw_rms,
                reference_rms=0.0,
                correlation=0.0,
                estimated_echo_gain=0.0,
                is_likely_pure_echo=False,
            )

        cleaned = frame
        best_corr = 0.0
        alpha = 0.0

        if self._enabled and ref_window.size >= frame.size and raw_rms >= 1e-5:
            n = frame.size
            mic_f64 = frame.astype(np.float64, copy=False)
            ref_f64 = ref_window.astype(np.float64, copy=False)
            mic_energy = float(np.dot(mic_f64, mic_f64))
            raw_corr = fftconvolve(ref_f64, mic_f64[::-1], mode="valid")
            cs = np.concatenate(([0.0], np.cumsum(ref_f64 * ref_f64)))
            seg_energies = cs[n : n + raw_corr.size] - cs[: raw_corr.size]
            denom = np.sqrt(np.maximum(seg_energies, 1e-12) * max(mic_energy, 1e-12))
            norm_corr = np.abs(raw_corr) / denom
            best_idx = int(np.argmax(norm_corr))
            best_corr = float(min(1.0, norm_corr[best_idx]))

            if best_corr >= self._min_corr:
                seg = ref_f64[best_idx : best_idx + n]
                seg_energy = float(np.dot(seg, seg)) + 1e-9
                alpha = float(np.clip(float(np.dot(mic_f64, seg)) / seg_energy, -2.5, 2.5))
                cleaned = (mic_f64 - alpha * seg).astype(np.float32)
                observed_gain = min(3.0, raw_rms / max(ref_rms, 1e-5))
                self._coupling_ema = (0.20 * observed_gain) + (0.80 * self._coupling_ema)

        residual_rms = compute_rms(cleaned)
        expected_echo_rms = ref_rms * self._coupling_ema

        if residual_rms < self._eff_min_rms:
            pure_echo = True
        elif not self._enabled:
            pure_echo = False
        elif best_corr >= self._min_corr and residual_rms < (raw_rms * 0.55):
            # Direct playback reflection largely cancelled by delayed reference projection.
            pure_echo = True
        elif expected_echo_rms >= self._eff_min_rms and raw_rms < (
            expected_echo_rms * self._eff_echo_ratio
        ):
            # Reverberant speaker energy below double-talk interruption margin.
            pure_echo = best_corr >= (self._min_corr * 0.7) or residual_rms < (
                expected_echo_rms * 0.9
            )
        else:
            pure_echo = False

        return EchoSuppressionResult(
            cleaned_frame=cleaned,
            raw_rms=raw_rms,
            residual_rms=residual_rms,
            reference_rms=ref_rms,
            correlation=round(best_corr, 4),
            estimated_echo_gain=round(abs(alpha), 4),
            is_likely_pure_echo=pure_echo,
        )

