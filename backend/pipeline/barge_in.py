"""Concurrent user speech barge-in monitor with echo suppression."""

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from backend.config import AppConfig, get_config
from backend.pipeline.audio_io import AudioCaptureSource, compute_waveform_bars
from backend.pipeline.echo import EchoSuppressor
from backend.stt.vad import SileroVadDetector


@dataclass
class BargeInCapture:
    """Result and buffered onset frames captured when user speech interrupts a turn."""

    triggered: bool = False
    prefix_chunks: list[np.ndarray] = field(default_factory=list)
    first_speech_at: float | None = None
    triggered_at: float | None = None
    detect_latency_ms: float | None = None


class BargeInMonitor:
    """Monitor microphone frames during GENERATING/SPEAKING for genuine user speech."""

    def __init__(
        self,
        capture: AudioCaptureSource,
        vad: SileroVadDetector,
        echo_suppressor: EchoSuppressor,
        config: AppConfig | None = None,
    ) -> None:
        self._capture = capture
        self._vad = vad
        self._echo = echo_suppressor
        self._config = config if config is not None else get_config()
        self._sample_rate = self._config.vad.sample_rate
        self._chunk_size = self._config.vad.window_size_samples
        frame_ms = (self._chunk_size / self._sample_rate) * 1000.0
        vad_thresh, min_frames, min_rms, _ = self._config.pipeline.effective_barge_in_params(
            frame_ms
        )
        self._vad_threshold = vad_thresh
        self._min_speech_frames = min_frames
        self._min_rms = min_rms
        self._pre_pad_frames = max(
            1,
            round(self._config.pipeline.pre_speech_pad_ms / frame_ms),
        )

    def watch(
        self,
        *,
        turn_cancel: threading.Event,
        monitor_stop: threading.Event,
        on_barge_in: Callable[[BargeInCapture], None],
        on_audio_level: Callable[[str, float, list[float]], None] | None = None,
    ) -> BargeInCapture:
        """Read incoming mic frames until speech triggers barge-in or monitor stops."""
        self._vad.reset_states()
        ring_buffer: deque[np.ndarray] = deque(
            maxlen=self._pre_pad_frames + self._min_speech_frames
        )
        speech_frames = 0
        first_speech_at: float | None = None
        last_level_at = 0.0

        while not turn_cancel.is_set() and not monitor_stop.is_set():
            chunk = self._capture.read_chunk(timeout_s=0.04)
            if chunk is None:
                # Yield briefly so mocked non-blocking sources do not spin CPU when empty.
                monitor_stop.wait(timeout=0.01)
                continue

            echo_res = self._echo.process_frame(chunk)
            if on_audio_level is not None:
                now = time.monotonic()
                if (now - last_level_at) >= 0.045:
                    last_level_at = now
                    _, bars = compute_waveform_bars(echo_res.cleaned_frame)
                    on_audio_level("mic", round(echo_res.residual_rms, 4), bars)
            ring_buffer.append(
                echo_res.cleaned_frame if echo_res.reference_rms > 0 else chunk
            )
            prob = self._vad.predict_chunk(echo_res.cleaned_frame, self._sample_rate)
            is_user_speech = (
                not echo_res.is_likely_pure_echo
                and echo_res.residual_rms >= self._min_rms
                and prob >= self._vad_threshold
            )

            if is_user_speech:
                if speech_frames == 0:
                    first_speech_at = time.perf_counter()
                speech_frames += 1
                if speech_frames >= self._min_speech_frames:
                    triggered_at = time.perf_counter()
                    detect_ms = (
                        round((triggered_at - first_speech_at) * 1000.0, 2)
                        if first_speech_at is not None
                        else 0.0
                    )
                    capture_res = BargeInCapture(
                        triggered=True,
                        prefix_chunks=list(ring_buffer),
                        first_speech_at=first_speech_at,
                        triggered_at=triggered_at,
                        detect_latency_ms=detect_ms,
                    )
                    on_barge_in(capture_res)
                    return capture_res
            else:
                speech_frames = 0
                first_speech_at = None

        return BargeInCapture(triggered=False)
