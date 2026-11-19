"""Microphone audio capture, streaming Silero VAD endpointing, and playback."""

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import sounddevice as sd
from pydantic import BaseModel, ConfigDict

from backend.config import AppConfig, get_config
from backend.logger import get_logger
from backend.pipeline.echo import PlaybackReferenceBuffer, resample_mono
from backend.stt.vad import SileroVadDetector

if TYPE_CHECKING:
    from backend.pipeline.state import PipelineStateMachine
    from backend.tts.streaming import AudioChunk


class AudioDeviceError(RuntimeError):
    """Raised when audio capture or playback hardware fails or is unavailable."""


class AudioDeviceInfo(BaseModel):
    """Normalized audio device metadata."""

    model_config = ConfigDict(extra="forbid")

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool
    is_default_output: bool


def list_audio_devices() -> list[AudioDeviceInfo]:
    """Query available PortAudio input/output devices on this machine."""
    try:
        raw_devices = sd.query_devices()
        default_in, default_out = sd.default.device
    except Exception as exc:
        raise AudioDeviceError(f"Failed to query audio devices: {exc}") from exc

    devices: list[AudioDeviceInfo] = []
    for idx, info in enumerate(raw_devices):
        devices.append(
            AudioDeviceInfo(
                index=idx,
                name=str(info.get("name", f"Device {idx}")),
                max_input_channels=int(info.get("max_input_channels", 0)),
                max_output_channels=int(info.get("max_output_channels", 0)),
                default_samplerate=float(info.get("default_samplerate", 0.0)),
                is_default_input=(idx == default_in),
                is_default_output=(idx == default_out),
            )
        )
    return devices


def resolve_audio_device(
    device: int | str | None,
    kind: Literal["input", "output"],
) -> int | None:
    """Resolve explicit device index or name substring, or None for system default."""
    if device is None:
        return None

    devices = list_audio_devices()
    channel_attr = "max_input_channels" if kind == "input" else "max_output_channels"
    valid = [d for d in devices if getattr(d, channel_attr) > 0]
    if not valid:
        raise AudioDeviceError(f"No {kind} audio devices available.")

    if isinstance(device, int):
        for d in valid:
            if d.index == device:
                return d.index
        raise AudioDeviceError(f"Audio {kind} device index {device} not found or has no channels.")

    query = device.lower().strip()
    for d in valid:
        if query in d.name.lower():
            return d.index
    raise AudioDeviceError(
        f"Audio {kind} device matching '{device}' not found. Available: {[d.name for d in valid]}"
    )


_resample_mono = resample_mono


def compute_waveform_bars(samples: np.ndarray, num_bars: int = 24) -> tuple[float, list[float]]:
    """Compute normalized RMS level and downsampled bar amplitudes for live visualization."""
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0 or num_bars <= 0:
        return 0.0, []
    rms = float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))
    usable = (arr.size // num_bars) * num_bars
    if usable >= num_bars:
        bins = np.abs(arr[:usable]).reshape(num_bars, -1).mean(axis=1)
        bars = [round(min(1.0, float(v) * 4.0), 3) for v in bins]
    else:
        bars = [round(min(1.0, abs(float(v)) * 4.0), 3) for v in arr[:num_bars]]
    return round(rms, 5), bars


class AudioCaptureSource(Protocol):
    """Protocol for chunked, gateable mono audio sources (real mic or test mock)."""

    @property
    def is_paused(self) -> bool:
        """Return True when capture is gated and incoming audio is ignored."""

    def start(self) -> None:
        """Start capturing audio frames."""

    def stop(self) -> None:
        """Stop capturing audio frames."""

    def pause(self) -> None:
        """Ignore incoming audio and clear queued frames without closing the stream."""

    def resume(self) -> None:
        """Clear stale frames and resume accepting incoming audio."""

    def clear_buffer(self) -> None:
        """Discard all buffered/queued audio frames."""

    def read_chunk(self, timeout_s: float = 0.2) -> np.ndarray | None:
        """Return next chunk of float32 mono samples or None on timeout/pause."""


class AudioPlaybackSink(Protocol):
    """Protocol for interruptible speaker audio playback."""

    def cancel(self) -> None:
        """Cancel active playback if playing."""

    def reset_cancel(self) -> None:
        """Clear cancellation flag before starting new playback."""

    def play(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        stop_event: threading.Event | None = None,
    ) -> tuple[float, bool]:
        """Play audio and return (elapsed_ms, interrupted)."""


class MicrophoneAudioCapture:
    """Continuous, gateable microphone capture backed by PortAudio / sounddevice."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        device: int | str | None = None,
        sample_rate: int | None = None,
        chunk_size: int | None = None,
        frame_processor: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self._config = config if config is not None else get_config()
        self._logger = get_logger("localvox.audio.capture", self._config)
        self._target_sr = sample_rate or self._config.audio.sample_rate
        self._chunk_size = chunk_size or self._config.audio.chunk_size
        self._device_spec = device if device is not None else self._config.audio.input_device
        self._frame_processor = frame_processor
        self._device_idx: int | None = None
        self._capture_sr: int = self._target_sr
        self._stream: sd.InputStream | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._pending = np.empty(0, dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._accepting = threading.Event()

    @property
    def is_paused(self) -> bool:
        """Return True when microphone callback is gated and ignoring audio."""
        return not self._accepting.is_set()

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            self._logger.debug("Microphone stream callback status: %s", status)
        # Drop microphone audio while paused (e.g., during assistant playback/processing).
        if not self._accepting.is_set():
            return

        mono = np.asarray(indata[:, 0] if indata.ndim > 1 else indata, dtype=np.float32).copy()
        if self._capture_sr != self._target_sr:
            mono = _resample_mono(mono, self._capture_sr, self._target_sr)
        if self._frame_processor is not None:
            mono = self._frame_processor(mono)

        if not self._accepting.is_set():
            return

        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            if self._accepting.is_set():
                self._queue.put_nowait(mono)

    def clear_buffer(self) -> None:
        """Flush all queued and pending audio frames safely."""
        with self._buffer_lock:
            self._pending = np.empty(0, dtype=np.float32)
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def pause(self) -> None:
        """Gate microphone capture during playback/processing without closing hardware."""
        self._accepting.clear()
        self.clear_buffer()

    def resume(self) -> None:
        """Flush any stale frames and re-enable microphone capture."""
        self.clear_buffer()
        self._accepting.set()

    def start(self) -> None:
        """Open and start the microphone InputStream."""
        if self._stream is not None:
            return
        self._device_idx = resolve_audio_device(self._device_spec, "input")
        try:
            dev_info = sd.query_devices(self._device_idx, "input")
        except Exception as exc:
            raise AudioDeviceError(f"No usable microphone input device available: {exc}") from exc
        if int(dev_info.get("max_input_channels", 0)) < 1:
            raise AudioDeviceError(f"Selected input device has no input channels: {dev_info}")

        self._capture_sr = self._target_sr
        blocksize = self._chunk_size
        try:
            sd.check_input_settings(
                device=self._device_idx,
                samplerate=self._capture_sr,
                channels=1,
                dtype="float32",
            )
        except Exception:
            fallback_sr = int(dev_info.get("default_samplerate") or 48000)
            self._capture_sr = fallback_sr
            blocksize = max(256, int(round(self._chunk_size * (fallback_sr / self._target_sr))))

        self.clear_buffer()
        self._accepting.set()
        try:
            self._stream = sd.InputStream(
                device=self._device_idx,
                samplerate=self._capture_sr,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._accepting.clear()
            self._stream = None
            raise AudioDeviceError(f"Failed to start microphone stream: {exc}") from exc

        self._logger.info(
            "Microphone capture started (device=%s, capture_sr=%d, target_sr=%d)",
            dev_info.get("name"),
            self._capture_sr,
            self._target_sr,
        )

    def stop(self) -> None:
        """Stop and close the microphone InputStream cleanly."""
        self._accepting.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
                self.clear_buffer()

    def read_chunk(self, timeout_s: float = 0.2) -> np.ndarray | None:
        """Return an exact chunk_size mono float32 frame at target_sr, or None on timeout/pause."""
        if not self._accepting.is_set():
            return None

        while True:
            with self._buffer_lock:
                if not self._accepting.is_set():
                    return None
                if self._pending.size >= self._chunk_size:
                    chunk = self._pending[: self._chunk_size].copy()
                    self._pending = self._pending[self._chunk_size :]
                    return chunk
            try:
                block = self._queue.get(timeout=timeout_s)
            except queue.Empty:
                return None
            with self._buffer_lock:
                if not self._accepting.is_set():
                    return None
                self._pending = np.concatenate([self._pending, block])


class SpeechEndpointDetector:
    """Segment live microphone frames into clean utterances using Silero VAD."""

    def __init__(
        self,
        vad: SileroVadDetector,
        config: AppConfig | None = None,
    ) -> None:
        self._vad = vad
        self._config = config if config is not None else get_config()
        self._sample_rate = self._config.vad.sample_rate
        self._chunk_size = self._config.vad.window_size_samples
        frame_ms = (self._chunk_size / self._sample_rate) * 1000.0
        self._pre_pad_frames = max(1, round(self._config.pipeline.pre_speech_pad_ms / frame_ms))
        self._post_pad_frames = max(
            0, round(self._config.pipeline.post_speech_pad_ms / frame_ms)
        )
        self._min_speech_frames = max(
            1, round(self._config.vad.min_speech_duration_ms / frame_ms)
        )
        self._hangover_frames = max(1, round(self._config.pipeline.vad_hangover_ms / frame_ms))
        self._min_utterance_samples = int(
            (self._config.pipeline.min_utterance_ms / 1000.0) * self._sample_rate
        )
        self._max_utterance_samples = int(
            self._config.pipeline.max_utterance_s * self._sample_rate
        )
        self._threshold = self._config.vad.threshold
        self._min_rms = self._config.pipeline.min_audio_rms

    def capture_utterance(
        self,
        capture: AudioCaptureSource,
        *,
        stop_event: threading.Event | None = None,
        state_machine: "PipelineStateMachine | None" = None,
        on_speech_start: Callable[[], None] | None = None,
        timeout_s: float | None = None,
        initial_chunks: list[np.ndarray] | None = None,
        already_in_speech: bool = False,
        on_audio_level: Callable[[str, float, list[float]], None] | None = None,
    ) -> tuple[np.ndarray, float] | None:
        """Listen until speech starts and ends; return (waveform, vad_ms) or None on stop."""
        self._vad.reset_states()
        ring_buffer: deque[np.ndarray] = deque(
            maxlen=self._pre_pad_frames + self._min_speech_frames
        )
        utterance_chunks: list[np.ndarray] = []
        in_speech = False
        speech_frames = 0
        silence_frames = 0
        vad_time_ms = 0.0
        total_samples = 0
        start_wait = time.monotonic()
        last_level_at = 0.0

        if initial_chunks:
            for seed in initial_chunks:
                t0 = time.perf_counter()
                self._vad.predict_chunk(seed, self._sample_rate)
                vad_time_ms += (time.perf_counter() - t0) * 1000.0
                if already_in_speech:
                    utterance_chunks.append(seed)
                    total_samples += seed.size
                else:
                    ring_buffer.append(seed)
            if already_in_speech:
                in_speech = True
                if on_speech_start is not None:
                    on_speech_start()

        while stop_event is None or not stop_event.is_set():
            if state_machine is not None and not state_machine.can_accept_speech():
                return None

            timed_out = timeout_s is not None and (time.monotonic() - start_wait) >= timeout_s
            if not in_speech and timed_out:
                return None

            chunk = capture.read_chunk(timeout_s=0.1)
            if chunk is None:
                if in_speech and timed_out:
                    break
                continue

            frame_rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
            if on_audio_level is not None:
                now = time.monotonic()
                if (now - last_level_at) >= 0.045:
                    last_level_at = now
                    _, bars = compute_waveform_bars(chunk)
                    on_audio_level("mic", round(frame_rms, 5), bars)
            t0 = time.perf_counter()
            prob = self._vad.predict_chunk(chunk, self._sample_rate)
            vad_time_ms += (time.perf_counter() - t0) * 1000.0
            is_speech = prob >= self._threshold and frame_rms >= (self._min_rms * 0.5)

            if not in_speech:
                ring_buffer.append(chunk)
                if is_speech:
                    speech_frames += 1
                    if speech_frames >= self._min_speech_frames:
                        in_speech = True
                        utterance_chunks.extend(ring_buffer)
                        total_samples = sum(c.size for c in utterance_chunks)
                        silence_frames = 0
                        if on_speech_start is not None:
                            on_speech_start()
                else:
                    speech_frames = 0
                continue

            utterance_chunks.append(chunk)
            total_samples += chunk.size
            if is_speech:
                silence_frames = 0
            else:
                silence_frames += 1

            reached_end = silence_frames >= self._hangover_frames
            reached_cap = total_samples >= self._max_utterance_samples
            if not reached_end and not reached_cap:
                continue

            tail_keep_frames = min(silence_frames, self._post_pad_frames)
            trim_frames = max(0, silence_frames - tail_keep_frames)
            if trim_frames > 0 and trim_frames < len(utterance_chunks):
                utterance_chunks = utterance_chunks[: len(utterance_chunks) - trim_frames]

            waveform = np.concatenate(utterance_chunks).astype(np.float32)
            rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
            if waveform.size >= self._min_utterance_samples and rms >= self._min_rms:
                return waveform, round(vad_time_ms, 2)

            in_speech = False
            speech_frames = 0
            silence_frames = 0
            utterance_chunks.clear()
            ring_buffer.clear()
            self._vad.reset_states()

        if in_speech and utterance_chunks:
            waveform = np.concatenate(utterance_chunks).astype(np.float32)
            rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
            if waveform.size >= self._min_utterance_samples and rms >= self._min_rms:
                return waveform, round(vad_time_ms, 2)
        return None


class SpeakerAudioPlayer:
    """Interruptible audio playback through the configured output device."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        device: int | str | None = None,
        reference_buffer: PlaybackReferenceBuffer | None = None,
        on_audio_level: Callable[[str, float, list[float]], None] | None = None,
    ) -> None:
        self._config = config if config is not None else get_config()
        self._logger = get_logger("localvox.audio.player", self._config)
        self._device_spec = device if device is not None else self._config.audio.output_device
        self._cancel_event = threading.Event()
        self._cancel_lock = threading.Lock()
        self._cancel_requested_at: float | None = None
        self._last_cancel_latency_ms: float | None = None
        self._on_audio_level = on_audio_level
        self._last_level_emit = 0.0
        self._reference_buffer = (
            reference_buffer
            if reference_buffer is not None
            else PlaybackReferenceBuffer(target_sr=self._config.vad.sample_rate)
        )

    @property
    def reference_buffer(self) -> PlaybackReferenceBuffer:
        """Return circular playback reference buffer fed to echo suppression."""
        return self._reference_buffer

    @property
    def last_cancel_latency_ms(self) -> float | None:
        """Return measured ms between cancel request and PortAudio abort completion."""
        return self._last_cancel_latency_ms

    def cancel(self) -> None:
        """Cancel active audio playback immediately."""
        with self._cancel_lock:
            if not self._cancel_event.is_set():
                self._cancel_requested_at = time.perf_counter()
                self._cancel_event.set()

    def reset_cancel(self) -> None:
        """Clear cancellation state before starting new playback."""
        with self._cancel_lock:
            self._cancel_event.clear()
            self._cancel_requested_at = None
            self._last_cancel_latency_ms = None

    def set_audio_level_callback(
        self,
        callback: Callable[[str, float, list[float]], None] | None,
    ) -> None:
        """Register or replace throttled playback waveform level callback."""
        self._on_audio_level = callback

    def _is_cancelled(self, stop_event: threading.Event | None) -> bool:
        return self._cancel_event.is_set() or (stop_event is not None and stop_event.is_set())

    def _record_cancel_completed(self) -> None:
        with self._cancel_lock:
            if self._cancel_requested_at is not None:
                self._last_cancel_latency_ms = round(
                    (time.perf_counter() - self._cancel_requested_at) * 1000.0,
                    2,
                )

    def _resolve_output_sr(self, target_sr: int) -> tuple[int | None, int]:
        device_idx = resolve_audio_device(self._device_spec, "output")
        try:
            dev_info = sd.query_devices(device_idx, "output")
        except Exception as exc:
            raise AudioDeviceError(f"No usable speaker output device available: {exc}") from exc

        try:
            sd.check_output_settings(
                device=device_idx,
                samplerate=target_sr,
                channels=1,
                dtype="float32",
            )
            return device_idx, target_sr
        except Exception:
            fallback_sr = int(dev_info.get("default_samplerate") or 48000)
            return device_idx, fallback_sr

    def _write_blocks(
        self,
        stream: sd.OutputStream,
        audio: np.ndarray,
        play_sr: int,
        block_frames: int,
        stop_event: threading.Event | None,
        *,
        push_reference: bool = True,
    ) -> bool:
        for offset in range(0, audio.size, block_frames):
            if self._is_cancelled(stop_event):
                stream.abort()
                self._record_cancel_completed()
                return True
            chunk = audio[offset : offset + block_frames]
            if push_reference:
                self._reference_buffer.push(chunk, play_sr)
            if self._on_audio_level is not None:
                now = time.monotonic()
                if (now - self._last_level_emit) >= 0.045:
                    self._last_level_emit = now
                    rms, bars = compute_waveform_bars(chunk)
                    self._on_audio_level("speaker", rms, bars)
            stream.write(chunk.reshape(-1, 1))
        return False

    def play(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        stop_event: threading.Event | None = None,
    ) -> tuple[float, bool]:
        """Play float32 mono audio in small blocks; returns (elapsed_ms, interrupted)."""
        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return 0.0, False

        device_idx, play_sr = self._resolve_output_sr(sample_rate)
        if play_sr != sample_rate:
            audio = resample_mono(audio, sample_rate, play_sr)

        block_frames = max(256, int(0.025 * play_sr))
        interrupted = False
        start = time.perf_counter()
        try:
            with sd.OutputStream(
                device=device_idx,
                samplerate=play_sr,
                channels=1,
                dtype="float32",
            ) as stream:
                interrupted = self._write_blocks(
                    stream,
                    audio,
                    play_sr,
                    block_frames,
                    stop_event,
                    push_reference=True,
                )
        except Exception as exc:
            raise AudioDeviceError(f"Failed during speaker audio playback: {exc}") from exc
        finally:
            if self._on_audio_level is not None:
                self._on_audio_level("idle", 0.0, [])
            if interrupted:
                self._reference_buffer.clear()

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        return elapsed_ms, interrupted

    def play_stream(
        self,
        chunk_queue: "queue.Queue[AudioChunk | None]",
        *,
        stop_event: threading.Event | None = None,
        on_chunk_start: "Callable[[AudioChunk], None] | None" = None,
    ) -> tuple[float, bool]:
        """Play streamed AudioChunk items seamlessly on one OutputStream until sentinel."""
        interrupted = False
        stream: sd.OutputStream | None = None
        device_idx: int | None = None
        play_sr: int | None = None
        block_frames = 512
        gap_ms = self._config.pipeline.tts_chunk_gap_ms
        start: float | None = None
        played_any = False

        try:
            while True:
                if self._is_cancelled(stop_event):
                    interrupted = True
                    if stream is not None:
                        stream.abort()
                        self._record_cancel_completed()
                    break

                try:
                    item = chunk_queue.get(timeout=0.04)
                except queue.Empty:
                    continue

                if item is None:
                    break

                audio = np.asarray(item.waveform, dtype=np.float32).reshape(-1)
                if audio.size == 0:
                    continue

                if stream is None:
                    device_idx, play_sr = self._resolve_output_sr(item.sample_rate)
                    block_frames = max(256, int(0.025 * play_sr))
                    stream = sd.OutputStream(
                        device=device_idx,
                        samplerate=play_sr,
                        channels=1,
                        dtype="float32",
                    )
                    stream.start()
                    start = time.perf_counter()

                assert play_sr is not None
                if played_any and gap_ms > 0:
                    gap_samples = int((gap_ms / 1000.0) * play_sr)
                    if gap_samples > 0:
                        interrupted = self._write_blocks(
                            stream,
                            np.zeros(gap_samples, dtype=np.float32),
                            play_sr,
                            block_frames,
                            stop_event,
                            push_reference=True,
                        )
                        if interrupted:
                            break

                if on_chunk_start is not None:
                    on_chunk_start(item)

                chunk_audio = (
                    resample_mono(audio, item.sample_rate, play_sr)
                    if item.sample_rate != play_sr
                    else audio
                )
                played_any = True
                interrupted = self._write_blocks(
                    stream,
                    chunk_audio,
                    play_sr,
                    block_frames,
                    stop_event,
                    push_reference=True,
                )
                if interrupted:
                    break
        except Exception as exc:
            raise AudioDeviceError(f"Failed during streaming speaker playback: {exc}") from exc
        finally:
            if self._on_audio_level is not None:
                self._on_audio_level("idle", 0.0, [])
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            if interrupted:
                self._reference_buffer.clear()

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2) if start is not None else 0.0
        return elapsed_ms, interrupted
