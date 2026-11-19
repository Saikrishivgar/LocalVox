"""End-to-end offline voice pipeline orchestrator for LocalVox Phase 3."""

import inspect
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from backend.config import AppConfig, get_config
from backend.eval.benchmark import get_process_memory_mb
from backend.llm.ollama_client import LlmResult, OllamaClient, OllamaError
from backend.logger import get_logger, stage_timer
from backend.memory.manager import MemoryManager
from backend.pipeline.audio_io import (
    AudioCaptureSource,
    AudioDeviceError,
    AudioPlaybackSink,
    MicrophoneAudioCapture,
    SpeakerAudioPlayer,
    SpeechEndpointDetector,
    compute_waveform_bars,
)
from backend.pipeline.barge_in import BargeInCapture, BargeInMonitor
from backend.pipeline.conversation import ConversationManager
from backend.pipeline.echo import EchoSuppressor, PlaybackReferenceBuffer
from backend.pipeline.state import PipelineState, PipelineStateMachine, TurnMetrics
from backend.stt.vad import SileroVadDetector
from backend.stt.whisper_stt import MlxWhisperStt
from backend.tts.router import TtsRouter
from backend.tts.streaming import AudioChunk, SentenceBuffer
from backend.tts.types import TtsResult, UnsupportedLanguageError
from backend.wakeword.detector import OpenWakeWordEngine, WakeWordDetector
from backend.wakeword.events import WakeWordDetectionEvent


def _drain_queue(target_queue: queue.Queue[Any]) -> None:
    """Empty a queue cleanly without blocking."""
    while True:
        try:
            target_queue.get_nowait()
        except queue.Empty:
            break


def _put_sentinel(target_queue: queue.Queue[Any], sentinel: Any = None) -> None:
    """Ensure a worker shutdown sentinel is placed even if bounded queue was full."""
    try:
        target_queue.put_nowait(sentinel)
    except queue.Full:
        _drain_queue(target_queue)
        target_queue.put_nowait(sentinel)


class VoicePipeline:
    """Interruptible streaming pipeline: Mic <-> VAD -> STT -> LLM -> TTS -> Speaker."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        capture: AudioCaptureSource | None = None,
        player: AudioPlaybackSink | None = None,
        vad: SileroVadDetector | None = None,
        stt: MlxWhisperStt | None = None,
        llm: OllamaClient | None = None,
        tts_router: TtsRouter | None = None,
        conversation: ConversationManager | None = None,
        state_machine: PipelineStateMachine | None = None,
        memory_manager: MemoryManager | None = None,
        wakeword_detector: WakeWordDetector | None = None,
        wake_word_mode: bool = False,
        language: str | None = None,
        tts_backend: str | None = None,
        on_transcript: Callable[[str, str | None], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        on_response: Callable[[str], None] | None = None,
        on_turn_complete: Callable[[TurnMetrics], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_barge_in: Callable[[BargeInCapture], None] | None = None,
        on_audio_level: Callable[[str, float, list[float]], None] | None = None,
        on_wake_word: Callable[[WakeWordDetectionEvent], None] | None = None,
    ) -> None:
        self._config = config if config is not None else get_config()
        self._logger = get_logger("localvox.pipeline", self._config)
        self._vad = vad if vad is not None else SileroVadDetector(self._config)
        self._stt = stt if stt is not None else MlxWhisperStt(self._config)
        self._llm = llm if llm is not None else OllamaClient(self._config)
        self._tts = tts_router if tts_router is not None else TtsRouter(self._config)
        self._capture = (
            capture if capture is not None else MicrophoneAudioCapture(self._config)
        )
        self._player = player if player is not None else SpeakerAudioPlayer(self._config)
        self._conversation = (
            conversation
            if conversation is not None
            else ConversationManager(
                self._config.pipeline.voice_system_prompt,
                max_turns=self._config.pipeline.max_history_turns,
                max_chars=self._config.pipeline.max_history_chars,
            )
        )
        self._state = state_machine if state_machine is not None else PipelineStateMachine()
        self._endpoint_detector = SpeechEndpointDetector(self._vad, self._config)
        ref_buf = getattr(self._player, "reference_buffer", None)
        if not isinstance(ref_buf, PlaybackReferenceBuffer):
            ref_buf = PlaybackReferenceBuffer(target_sr=self._config.vad.sample_rate)
        self._reference_buffer = ref_buf
        self._echo_suppressor = EchoSuppressor(self._reference_buffer, self._config)
        self._language = language if language is not None else self._config.stt.default_language
        self._tts_backend = tts_backend
        self._on_transcript = on_transcript
        self._on_token = on_token
        self._on_response = on_response
        self._on_turn_complete = on_turn_complete
        self._on_error = on_error
        self._on_barge_in = on_barge_in
        self._on_audio_level = on_audio_level
        self._on_wake_word = on_wake_word
        self._wakeword: WakeWordDetector | None = wakeword_detector
        self._wake_word_mode = wake_word_mode
        self._wake_word_active = False
        self._just_woke_up = False
        self._cached_ack_audio: tuple[np.ndarray, int] | None = None
        self._last_wake_to_listen_ms: float | None = None
        if hasattr(self._player, "set_audio_level_callback"):
            self._player.set_audio_level_callback(on_audio_level)
        self._turn_counter = 0
        self._pending_barge_in_chunks: list[np.ndarray] | None = None
        self._foreground_llm_busy = threading.Event()
        self._execution_lock = threading.Lock()
        if memory_manager is not None:
            self._memory: MemoryManager | None = memory_manager
        elif self._config.memory.enabled:
            self._memory = MemoryManager(
                self._config,
                ollama=self._llm,
                foreground_busy=self._foreground_llm_busy,
            )
        else:
            self._memory = None

    @property
    def state_machine(self) -> PipelineStateMachine:
        """Return the pipeline state machine."""
        return self._state

    @property
    def memory_manager(self) -> MemoryManager | None:
        """Return persistent local memory manager if enabled."""
        return self._memory

    @property
    def conversation(self) -> ConversationManager:
        """Return the active session conversation manager."""
        return self._conversation

    @property
    def wakeword_detector(self) -> WakeWordDetector:
        """Return lazy or injected offline openWakeWord detector."""
        if self._wakeword is None:
            self._wakeword = OpenWakeWordEngine(self._config)
        return self._wakeword

    @property
    def wake_word_active(self) -> bool:
        """Return True while the hands-free wake-word standby/session loop is running."""
        return self._wake_word_active

    @property
    def last_wake_to_listen_ms(self) -> float | None:
        """Return elapsed ms from the most recent wake detection to LISTENING transition."""
        return self._last_wake_to_listen_ms

    def set_config(self, config: AppConfig) -> None:
        """Apply updated runtime configuration settings safely across components."""
        self._config = config
        self._language = config.stt.default_language
        self._vad._vad_cfg = config.vad
        self._llm._llm_cfg = config.llm
        self._tts._config = config
        if hasattr(self._tts, "_kokoro") and self._tts._kokoro is not None:
            self._tts._kokoro._config = config
            self._tts._kokoro._kokoro_cfg = config.tts.kokoro
        if hasattr(self._tts, "_piper") and self._tts._piper is not None:
            self._tts._piper._config = config
            self._tts._piper._piper_cfg = config.tts.piper
        self._conversation._max_turns = config.pipeline.max_history_turns
        self._conversation._max_chars = config.pipeline.max_history_chars
        if self._memory is None and config.memory.enabled:
            self._memory = MemoryManager(
                config,
                ollama=self._llm,
                foreground_busy=self._foreground_llm_busy,
            )
        if self._memory is not None:
            self._memory._config = config
            self._memory._store._config = config
            self._memory._retriever._config = config
            self._memory._extractor._config = config
            self._memory._commands._config = config
        if self._wakeword is not None:
            self._wakeword.set_config(config)
        self._cached_ack_audio = None
        self._endpoint_detector = SpeechEndpointDetector(self._vad, self._config)
        self._echo_suppressor = EchoSuppressor(self._reference_buffer, self._config)

    @property
    def echo_suppressor(self) -> EchoSuppressor:
        """Return the active software echo suppressor."""
        return self._echo_suppressor

    def warmup(self, *, require_llm: bool = True) -> None:
        """Load and verify all local AI components once so turns reuse warm models."""
        self._vad.load()
        self._stt.load()
        try:
            if hasattr(self._llm, "warmup"):
                self._llm.warmup()
            else:
                self._llm.verify_model_exists()
        except OllamaError as exc:
            if require_llm:
                raise
            self._logger.warning(
                "Ollama not reachable during wake-word warmup (%s); standby remains active.",
                exc,
            )
        self._tts.preload(self._tts_backend)
        self._logger.info("LocalVox Phase 3 pipeline warmed up and ready offline.")

    def cancel_playback(self) -> None:
        """Cancel active LLM generation, TTS synthesis, and audio playback cleanly."""
        self._state.cancel_playback()
        if hasattr(self._player, "cancel"):
            self._player.cancel()

    def _matches_command(self, text: str, commands: list[str]) -> bool:
        normalized = " ".join(text.lower().strip(" .!?,:;").split())
        return any(cmd.lower() == normalized for cmd in commands)

    def _reset_listening_audio(self) -> None:
        """Discard stale buffered audio and reset VAD state before resuming listening."""
        if hasattr(self._capture, "clear_buffer"):
            self._capture.clear_buffer()
        self._vad.reset_states()
        self._echo_suppressor.reset()

    def _invoke_llm_chat(
        self,
        messages: list[dict[str, str]],
        *,
        on_token: Callable[[str], None] | None,
        cancel_event: threading.Event,
    ) -> Any:
        """Call llm.chat while keeping compatibility with test stubs lacking cancel_event."""
        chat_fn = self._llm.chat
        kwargs: dict[str, Any] = {"on_token": on_token}
        try:
            sig = inspect.signature(chat_fn)
            if "cancel_event" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["cancel_event"] = cancel_event
        except (TypeError, ValueError):
            kwargs["cancel_event"] = cancel_event
        return chat_fn(messages, **kwargs)

    def _speak_text(
        self,
        text: str,
        *,
        stt_language: str | None = None,
    ) -> tuple[str, str, float, float, bool]:
        """Synthesize and play a fixed status/command phrase cleanly."""
        if hasattr(self._capture, "pause"):
            self._capture.pause()
        self._state.transition(PipelineState.SPEAKING, "synthesizing_and_playing")
        waveform, sample_rate, tts_res = self._tts.synthesize(
            text,
            stt_language=stt_language,
            backend_override=self._tts_backend,
        )
        self._state.clear_barge_in()
        if hasattr(self._player, "reset_cancel"):
            self._player.reset_cancel()
        playback_ms, interrupted = self._player.play(
            waveform,
            sample_rate,
            stop_event=self._state.barge_in_event,
        )
        return (
            tts_res.backend,
            tts_res.language,
            tts_res.processing_time_ms,
            playback_ms,
            interrupted,
        )

    def _play_chunk_queue(
        self,
        audio_queue: queue.Queue[AudioChunk | None],
        *,
        turn_cancel: threading.Event,
        on_chunk_start: Callable[[AudioChunk], None],
    ) -> tuple[float, bool]:
        """Play ordered AudioChunk queue via play_stream or AudioPlaybackSink.play fallback."""
        if hasattr(self._player, "play_stream"):
            return self._player.play_stream(
                audio_queue,
                stop_event=turn_cancel,
                on_chunk_start=on_chunk_start,
            )

        # Fallback for minimal mock sinks implementing only AudioPlaybackSink.play().
        start: float | None = None
        interrupted = False
        while True:
            if turn_cancel.is_set() or self._state.stop_event.is_set():
                interrupted = True
                break
            try:
                item = audio_queue.get(timeout=0.02)
            except queue.Empty:
                continue
            if item is None:
                break
            if start is None:
                start = time.perf_counter()
            on_chunk_start(item)
            _, chunk_interrupted = self._player.play(
                item.waveform,
                item.sample_rate,
                stop_event=turn_cancel,
            )
            if chunk_interrupted or turn_cancel.is_set():
                interrupted = True
                break
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2) if start is not None else 0.0
        return elapsed_ms, interrupted

    def _execute_streaming_turn(
        self,
        *,
        transcript: str,
        stt_language: str | None,
        speech_end_at: float,
        extra_context: str | None = None,
        speak: bool = True,
    ) -> dict[str, Any]:
        """Run overlapping LLM streaming, optional incremental TTS playback, and barge-in."""
        gen_id, turn_cancel = self._state.begin_generation()
        monitor_stop = threading.Event()
        if hasattr(self._player, "reset_cancel"):
            self._player.reset_cancel()

        self._state.transition(PipelineState.GENERATING, "llm_inference")
        messages = self._conversation.build_messages(transcript, extra_context=extra_context)

        sentence_buffer = SentenceBuffer(
            min_sentence_chars=self._config.pipeline.tts_min_sentence_chars,
            max_chunk_chars=self._config.pipeline.tts_max_chunk_chars,
        )
        sentence_queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
        audio_queue: queue.Queue[AudioChunk | None] = queue.Queue(
            maxsize=self._config.pipeline.tts_queue_maxsize
        )

        worker_errors: list[Exception] = []
        error_lock = threading.Lock()
        llm_holder: list[Any] = []
        tts_results: list[TtsResult] = []
        spoken_texts: list[str] = []
        first_audible_at: list[float] = []
        barge_in_holder: list[BargeInCapture] = []
        state_lock = threading.Lock()

        def record_error(exc: Exception) -> None:
            with error_lock:
                worker_errors.append(exc)
            turn_cancel.set()
            monitor_stop.set()
            if hasattr(self._player, "cancel"):
                self._player.cancel()

        def enqueue_sentence(seq_idx: int, chunk_text: str) -> None:
            while not turn_cancel.is_set() and not self._state.stop_event.is_set():
                try:
                    sentence_queue.put((seq_idx, chunk_text), timeout=0.05)
                    return
                except queue.Full:
                    continue

        def llm_worker() -> None:
            seq_idx = 0
            tokens_seen = 0

            def handle_token(delta: str) -> None:
                nonlocal seq_idx, tokens_seen
                if turn_cancel.is_set() or not self._state.is_generation_current(gen_id):
                    return
                tokens_seen += 1
                if speak and self._config.pipeline.stream_llm_output and self._on_token is not None:
                    self._on_token(delta)
                elif not speak and self._on_token is not None:
                    self._on_token(delta)
                if speak and self._config.pipeline.streaming_tts_enabled:
                    for ready in sentence_buffer.push(delta):
                        enqueue_sentence(seq_idx, ready)
                        seq_idx += 1

            try:
                res = self._invoke_llm_chat(
                    messages,
                    on_token=handle_token,
                    cancel_event=turn_cancel,
                )
                llm_holder.append(res)
                if (
                    speak
                    and not turn_cancel.is_set()
                    and not getattr(res, "cancelled", False)
                    and self._state.is_generation_current(gen_id)
                ):
                    full_text = str(res.response_text).strip()
                    if not full_text:
                        raise RuntimeError("Local LLM returned an empty response.")
                    if self._config.pipeline.streaming_tts_enabled:
                        if tokens_seen == 0:
                            for ready in sentence_buffer.push(full_text):
                                enqueue_sentence(seq_idx, ready)
                                seq_idx += 1
                        for tail in sentence_buffer.flush():
                            enqueue_sentence(seq_idx, tail)
                            seq_idx += 1
                    else:
                        enqueue_sentence(seq_idx, full_text)
                        seq_idx += 1
                elif (
                    not speak
                    and not turn_cancel.is_set()
                    and not getattr(res, "cancelled", False)
                    and not str(res.response_text).strip()
                ):
                    raise RuntimeError("Local LLM returned an empty response.")
            except Exception as exc:
                record_error(exc)
            finally:
                if speak:
                    _put_sentinel(sentence_queue, None)

        def tts_worker() -> None:
            try:
                while True:
                    if turn_cancel.is_set() or self._state.stop_event.is_set():
                        break
                    try:
                        item = sentence_queue.get(timeout=0.04)
                    except queue.Empty:
                        continue
                    if item is None:
                        break

                    seq_idx, chunk_text = item
                    if turn_cancel.is_set() or not self._state.is_generation_current(gen_id):
                        break

                    waveform, sr, tts_res = self._tts.synthesize(
                        chunk_text,
                        stt_language=stt_language,
                        backend_override=self._tts_backend,
                    )
                    if turn_cancel.is_set() or not self._state.is_generation_current(gen_id):
                        break

                    tts_results.append(tts_res)
                    with state_lock:
                        if (
                            self._state.state == PipelineState.GENERATING
                            and not turn_cancel.is_set()
                            and not self._state.stop_event.is_set()
                        ):
                            self._state.transition(
                                PipelineState.SPEAKING,
                                "streaming_tts_playback",
                            )

                    chunk = AudioChunk(
                        seq_id=seq_idx,
                        text=chunk_text,
                        waveform=waveform,
                        sample_rate=sr,
                        tts_result=tts_res,
                    )
                    while not turn_cancel.is_set() and not self._state.stop_event.is_set():
                        try:
                            audio_queue.put(chunk, timeout=0.04)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                record_error(exc)
            finally:
                if turn_cancel.is_set():
                    _drain_queue(audio_queue)
                _put_sentinel(audio_queue, None)

        barge_thread: threading.Thread | None = None

        def handle_barge_in(info: BargeInCapture) -> None:
            barge_in_holder.append(info)
            self._state.request_barge_in()
            turn_cancel.set()
            monitor_stop.set()
            if hasattr(self._player, "cancel"):
                self._player.cancel()
            with state_lock:
                if self._state.state in (
                    PipelineState.GENERATING,
                    PipelineState.SPEAKING,
                ):
                    self._state.transition(PipelineState.INTERRUPTED, "user_barge_in")
            if self._on_barge_in is not None:
                self._on_barge_in(info)

        def on_chunk_start(chunk: AudioChunk) -> None:
            spoken_texts.append(chunk.text)
            if not first_audible_at:
                first_audible_at.append(time.perf_counter())

        if speak and self._config.pipeline.barge_in_enabled and not self._capture.is_paused:
            self._echo_suppressor.reset()
            monitor = BargeInMonitor(
                self._capture,
                self._vad,
                self._echo_suppressor,
                self._config,
            )
            barge_thread = threading.Thread(
                target=monitor.watch,
                kwargs={
                    "turn_cancel": turn_cancel,
                    "monitor_stop": monitor_stop,
                    "on_barge_in": handle_barge_in,
                    "on_audio_level": self._on_audio_level,
                },
                name="localvox-barge-in",
                daemon=True,
            )

        tts_thread = (
            threading.Thread(target=tts_worker, name="localvox-tts-worker", daemon=True)
            if speak
            else None
        )
        llm_thread = threading.Thread(target=llm_worker, name="localvox-llm-worker", daemon=True)

        self._foreground_llm_busy.set()
        if tts_thread is not None:
            tts_thread.start()
        llm_thread.start()
        if barge_thread is not None:
            barge_thread.start()

        playback_ms = 0.0
        playback_interrupted = False
        cancel_done_at: float | None = None
        try:
            if speak:
                playback_ms, playback_interrupted = self._play_chunk_queue(
                    audio_queue,
                    turn_cancel=turn_cancel,
                    on_chunk_start=on_chunk_start,
                )
                if playback_interrupted or barge_in_holder:
                    cancel_done_at = time.perf_counter()
            else:
                llm_thread.join()
        finally:
            monitor_stop.set()
            if worker_errors or self._state.stop_event.is_set() or barge_in_holder:
                turn_cancel.set()
            if turn_cancel.is_set():
                _drain_queue(sentence_queue)
                if speak:
                    _put_sentinel(sentence_queue, None)
                _drain_queue(audio_queue)
                if speak:
                    _put_sentinel(audio_queue, None)
            llm_thread.join(timeout=5.0)
            self._foreground_llm_busy.clear()
            if tts_thread is not None:
                tts_thread.join(timeout=5.0)
            if barge_thread is not None:
                barge_thread.join(timeout=5.0)

        if worker_errors:
            raise worker_errors[0]

        barge_info = barge_in_holder[0] if barge_in_holder else None
        barge_triggered = barge_info is not None
        interrupted = bool(
            playback_interrupted
            or barge_triggered
            or turn_cancel.is_set()
            or self._state.barge_in_event.is_set()
        )
        if barge_info is not None and barge_info.prefix_chunks:
            self._pending_barge_in_chunks = barge_info.prefix_chunks

        llm_res = llm_holder[0] if llm_holder else LlmResult(
            model_name=self._config.llm.model_name,
            response_text="",
            thinking_text="",
            ttft_ms=0.0,
            total_time_ms=0.0,
            load_duration_ms=None,
            prompt_tokens=0,
            completion_tokens=0,
            tokens_per_sec=0.0,
            cancelled=interrupted,
        )

        reply_text = str(llm_res.response_text).strip()
        spoken_joined = " ".join(spoken_texts).strip()
        if interrupted:
            recorded_reply = (
                f"{spoken_joined} [interrupted]"
                if spoken_joined
                else "[interrupted before speaking]"
            )
            self._conversation.add_turn(transcript, recorded_reply)
            display_reply = spoken_joined or reply_text
        else:
            recorded_reply = reply_text
            self._conversation.add_turn(transcript, recorded_reply)
            display_reply = reply_text

        if display_reply and self._on_response is not None:
            self._on_response(display_reply)

        tts_total_ms = round(sum(r.processing_time_ms for r in tts_results), 2)
        tts_ttfa_ms = round(tts_results[0].processing_time_ms, 2) if tts_results else 0.0
        tts_backend = tts_results[0].backend if tts_results else self._config.tts.default_backend
        tts_language = (
            tts_results[0].language
            if tts_results
            else (stt_language or self._config.tts.default_language)
        )
        first_audio_latency_ms = (
            round((first_audible_at[0] - speech_end_at) * 1000.0, 2)
            if first_audible_at
            else 0.0
        )

        playback_cancel_ms: float | None = None
        if barge_triggered:
            player_cancel_ms = getattr(self._player, "last_cancel_latency_ms", None)
            if player_cancel_ms is not None:
                playback_cancel_ms = float(player_cancel_ms)
            elif (
                barge_info is not None
                and barge_info.triggered_at is not None
                and cancel_done_at is not None
            ):
                playback_cancel_ms = round(
                    max(0.0, (cancel_done_at - barge_info.triggered_at) * 1000.0),
                    2,
                )

        return {
            "response_text": display_reply,
            "tts_backend": tts_backend,
            "tts_language": tts_language,
            "llm_ttft_ms": float(llm_res.ttft_ms),
            "llm_total_ms": float(llm_res.total_time_ms),
            "llm_tokens_per_sec": float(llm_res.tokens_per_sec),
            "tts_ms": tts_total_ms,
            "tts_ttfa_ms": tts_ttfa_ms,
            "tts_chunks": len(tts_results),
            "playback_ms": playback_ms,
            "first_audio_latency_ms": first_audio_latency_ms,
            "interrupted": interrupted,
            "barge_in_triggered": barge_triggered,
            "interruption_detect_ms": barge_info.detect_latency_ms if barge_info else None,
            "playback_cancel_ms": playback_cancel_ms,
        }

    def _process_transcript(
        self,
        transcript: str,
        *,
        detected_language: str | None,
        vad_ms: float,
        stt_ms: float,
        speech_end_to_stt_ms: float,
        turn_start: float,
        utt_end_ts: float,
        rec: dict[str, Any],
        speak: bool = True,
        return_state: PipelineState | None = PipelineState.LISTENING,
    ) -> TurnMetrics | None:
        """Execute shared command routing, memory retrieval/extraction, LLM, and optional TTS."""
        barge_in_active = self._config.pipeline.barge_in_enabled

        if self._matches_command(transcript, self._config.pipeline.clear_commands):
            self._conversation.clear()
            reply = "Conversation cleared. Ready for your next question."
            if self._on_response is not None:
                self._on_response(reply)
            if speak:
                backend, tts_lang, tts_ms, playback_ms, interrupted = self._speak_text(
                    reply, stt_language="en"
                )
            else:
                backend, tts_lang, tts_ms, playback_ms, interrupted = (
                    self._config.tts.default_backend,
                    "en",
                    0.0,
                    0.0,
                    False,
                )
            self._turn_counter += 1
            total_ms = round((time.perf_counter() - turn_start) * 1000.0, 2)
            metrics = TurnMetrics(
                turn_id=self._turn_counter,
                transcript=transcript,
                detected_language=detected_language,
                response_text=reply,
                tts_backend=backend,
                tts_language=tts_lang,
                vad_ms=vad_ms,
                stt_ms=stt_ms,
                llm_ttft_ms=0.0,
                llm_total_ms=0.0,
                llm_tokens_per_sec=0.0,
                tts_ms=tts_ms,
                playback_ms=playback_ms,
                response_latency_ms=round(stt_ms + tts_ms, 2),
                total_turn_ms=total_ms,
                memory_mb=get_process_memory_mb(),
                interrupted=interrupted,
                speech_end_to_stt_ms=speech_end_to_stt_ms,
                tts_ttfa_ms=tts_ms,
                tts_chunks=1 if speak else 0,
            )
            if self._on_turn_complete is not None:
                self._on_turn_complete(metrics)
            if return_state is not None and not self._state.stop_event.is_set():
                self._state.transition(return_state, "conversation_cleared")
            return metrics

        if self._matches_command(transcript, self._config.pipeline.exit_commands):
            reply = "Goodbye! Happy studying."
            if self._on_response is not None:
                self._on_response(reply)
            if speak:
                backend, tts_lang, tts_ms, playback_ms, interrupted = self._speak_text(
                    reply, stt_language="en"
                )
            else:
                backend, tts_lang, tts_ms, playback_ms, interrupted = (
                    self._config.tts.default_backend,
                    "en",
                    0.0,
                    0.0,
                    False,
                )
            self._state.request_stop()
            self._turn_counter += 1
            total_ms = round((time.perf_counter() - turn_start) * 1000.0, 2)
            return TurnMetrics(
                turn_id=self._turn_counter,
                transcript=transcript,
                detected_language=detected_language,
                response_text=reply,
                tts_backend=backend,
                tts_language=tts_lang,
                vad_ms=vad_ms,
                stt_ms=stt_ms,
                llm_ttft_ms=0.0,
                llm_total_ms=0.0,
                llm_tokens_per_sec=0.0,
                tts_ms=tts_ms,
                playback_ms=playback_ms,
                response_latency_ms=round(stt_ms + tts_ms, 2),
                total_turn_ms=total_ms,
                memory_mb=get_process_memory_mb(),
                interrupted=interrupted,
                speech_end_to_stt_ms=speech_end_to_stt_ms,
                tts_ttfa_ms=tts_ms,
                tts_chunks=1 if speak else 0,
            )

        recent_pairs = [(t.user, t.assistant) for t in self._conversation.turns[-2:]]
        if self._memory is not None:
            mem_cmd = self._memory.handle_command(
                transcript,
                recent_turns=recent_pairs,
            )
            if mem_cmd is not None:
                reply = mem_cmd.response_text
                self._conversation.add_turn(transcript, reply)
                if self._on_response is not None:
                    self._on_response(reply)
                if speak:
                    backend, tts_lang, tts_ms, playback_ms, interrupted = self._speak_text(
                        reply,
                        stt_language=detected_language or "en",
                    )
                else:
                    backend, tts_lang, tts_ms, playback_ms, interrupted = (
                        self._config.tts.default_backend,
                        detected_language or "en",
                        0.0,
                        0.0,
                        False,
                    )
                self._turn_counter += 1
                total_ms = round((time.perf_counter() - turn_start) * 1000.0, 2)
                metrics = TurnMetrics(
                    turn_id=self._turn_counter,
                    transcript=transcript,
                    detected_language=detected_language,
                    response_text=reply,
                    tts_backend=backend,
                    tts_language=tts_lang,
                    vad_ms=vad_ms,
                    stt_ms=stt_ms,
                    llm_ttft_ms=0.0,
                    llm_total_ms=0.0,
                    llm_tokens_per_sec=0.0,
                    tts_ms=tts_ms,
                    playback_ms=playback_ms,
                    response_latency_ms=round(stt_ms + tts_ms, 2),
                    total_turn_ms=total_ms,
                    memory_mb=get_process_memory_mb(),
                    interrupted=interrupted,
                    speech_end_to_stt_ms=speech_end_to_stt_ms,
                    tts_ttfa_ms=tts_ms,
                    tts_chunks=1 if speak else 0,
                )
                if self._on_turn_complete is not None:
                    self._on_turn_complete(metrics)
                if return_state is not None and not self._state.stop_event.is_set():
                    self._state.transition(return_state, f"memory_{mem_cmd.action}")
                return metrics

        memory_context: str | None = None
        mem_retrieval_ms = 0.0
        mem_count = 0
        mem_chars = 0
        if self._memory is not None:
            _, memory_context, ret_metrics = self._memory.retrieve_context(transcript)
            mem_retrieval_ms = ret_metrics.retrieval_ms
            mem_count = ret_metrics.memories_retrieved
            mem_chars = ret_metrics.context_chars

        if speak and barge_in_active and hasattr(self._capture, "clear_buffer"):
            # Discard frames accumulated while Whisper was executing STT.
            self._capture.clear_buffer()

        turn_out = self._execute_streaming_turn(
            transcript=transcript,
            stt_language=detected_language,
            speech_end_at=utt_end_ts,
            extra_context=memory_context,
            speak=speak,
        )
        if self._memory is not None and turn_out["response_text"]:
            self._memory.extract_turn(
                transcript,
                turn_out["response_text"],
                recent_turns=recent_pairs,
            )
        self._turn_counter += 1
        total_ms = round((time.perf_counter() - turn_start) * 1000.0, 2)
        response_ms = round(stt_ms + turn_out["llm_total_ms"] + turn_out["tts_ms"], 2)
        rec["turn_id"] = self._turn_counter
        rec["stt_ms"] = stt_ms
        rec["llm_ms"] = turn_out["llm_total_ms"]
        rec["tts_ms"] = turn_out["tts_ms"]
        rec["first_audio_latency_ms"] = turn_out["first_audio_latency_ms"]
        rec["interrupted"] = turn_out["interrupted"]
        rec["memory_retrieval_ms"] = mem_retrieval_ms
        rec["memories_retrieved"] = mem_count
        rec["memory_context_chars"] = mem_chars

        metrics = TurnMetrics(
            turn_id=self._turn_counter,
            transcript=transcript,
            detected_language=detected_language,
            response_text=turn_out["response_text"],
            tts_backend=turn_out["tts_backend"],
            tts_language=turn_out["tts_language"],
            vad_ms=vad_ms,
            stt_ms=stt_ms,
            llm_ttft_ms=turn_out["llm_ttft_ms"],
            llm_total_ms=turn_out["llm_total_ms"],
            llm_tokens_per_sec=turn_out["llm_tokens_per_sec"],
            tts_ms=turn_out["tts_ms"],
            playback_ms=turn_out["playback_ms"],
            response_latency_ms=response_ms,
            total_turn_ms=total_ms,
            memory_mb=get_process_memory_mb(),
            interrupted=turn_out["interrupted"],
            speech_end_to_stt_ms=speech_end_to_stt_ms,
            tts_ttfa_ms=turn_out["tts_ttfa_ms"],
            first_audio_latency_ms=turn_out["first_audio_latency_ms"],
            interruption_detect_ms=turn_out["interruption_detect_ms"],
            playback_cancel_ms=turn_out["playback_cancel_ms"],
            tts_chunks=turn_out["tts_chunks"] if speak else 0,
            barge_in_triggered=turn_out["barge_in_triggered"],
            memory_retrieval_ms=mem_retrieval_ms,
            memories_retrieved=mem_count,
            memory_context_chars=mem_chars,
        )
        if self._on_turn_complete is not None:
            self._on_turn_complete(metrics)
        if return_state is not None and not self._state.stop_event.is_set():
            next_reason = (
                "barge_in_resume_listening"
                if turn_out["barge_in_triggered"]
                else "turn_complete"
            )
            self._state.transition(return_state, next_reason)
        return metrics

    def process_utterance(
        self,
        utterance: np.ndarray,
        *,
        vad_ms: float = 0.0,
        sample_rate: int | None = None,
        speech_end_at: float | None = None,
        _transition_to_listening: bool = True,
    ) -> TurnMetrics | None:
        """Run STT -> Conversation/LLM -> Streaming TTS -> Playback for one detected utterance."""
        sr = sample_rate if sample_rate is not None else self._config.audio.sample_rate
        turn_start = time.perf_counter()
        utt_end_ts = speech_end_at if speech_end_at is not None else turn_start
        barge_in_active = self._config.pipeline.barge_in_enabled

        if not barge_in_active and hasattr(self._capture, "pause"):
            self._capture.pause()

        with self._execution_lock:
            try:
                latency_path = self._config.logging.latency_log_file
                with stage_timer("pipeline_turn", log_path=latency_path) as rec:
                    self._state.transition(PipelineState.PROCESSING, "stt_transcribe")
                    stt_res = self._stt.transcribe_array(utterance, sr, language=self._language)
                    speech_end_to_stt_ms = round((time.perf_counter() - utt_end_ts) * 1000.0, 2)
                    transcript = stt_res.text.strip()
                    if self._just_woke_up:
                        self._just_woke_up = False
                        transcript = self._sanitize_wake_transcript(transcript)
                        if not transcript:
                            self._logger.info(
                                "Wake-phrase-only utterance ignored; staying in LISTENING."
                            )
                            self._state.transition(
                                PipelineState.LISTENING, "wake_phrase_ignored"
                            )
                            return None
                    if not transcript:
                        self._logger.info("Empty STT transcript ignored; returning to LISTENING.")
                        self._state.transition(PipelineState.LISTENING, "empty_transcript")
                        return None

                    if self._on_transcript is not None:
                        self._on_transcript(transcript, stt_res.language)

                    return self._process_transcript(
                        transcript,
                        detected_language=stt_res.language,
                        vad_ms=vad_ms,
                        stt_ms=stt_res.processing_time_ms,
                        speech_end_to_stt_ms=speech_end_to_stt_ms,
                        turn_start=turn_start,
                        utt_end_ts=utt_end_ts,
                        rec=rec,
                        speak=True,
                        return_state=(
                            PipelineState.LISTENING if _transition_to_listening else None
                        ),
                    )

            except (
                AudioDeviceError,
                OllamaError,
                UnsupportedLanguageError,
                FileNotFoundError,
                ValueError,
                RuntimeError,
            ) as exc:
                self._pending_barge_in_chunks = None
                if not self._state.stop_event.is_set():
                    self._state.transition(PipelineState.ERROR, str(exc))
                self._logger.error("Pipeline turn error: %s", exc)
                if self._on_error is not None:
                    self._on_error(exc)
                if not self._state.stop_event.is_set():
                    self._state.transition(PipelineState.LISTENING, "error_recovered")
                return None
            finally:
                if self._pending_barge_in_chunks is None:
                    self._reset_listening_audio()

    def process_text(
        self,
        text: str,
        *,
        speak: bool = True,
        language: str | None = None,
    ) -> TurnMetrics | None:
        """Process a desktop text input turn through memory, LLM, and optional TTS."""
        transcript = text.strip()
        if not transcript:
            return None
        turn_start = time.perf_counter()

        with self._execution_lock:
            was_stopped = self._state.stop_event.is_set()
            if self._state.state in (PipelineState.IDLE, PipelineState.ERROR):
                self._state.stop_event.clear()
                self._state.clear_barge_in()
            try:
                if self._on_transcript is not None:
                    self._on_transcript(transcript, language)
                latency_path = self._config.logging.latency_log_file
                with stage_timer("pipeline_text_turn", log_path=latency_path) as rec:
                    self._state.transition(PipelineState.PROCESSING, "text_input")
                    return self._process_transcript(
                        transcript,
                        detected_language=language,
                        vad_ms=0.0,
                        stt_ms=0.0,
                        speech_end_to_stt_ms=0.0,
                        turn_start=turn_start,
                        utt_end_ts=turn_start,
                        rec=rec,
                        speak=speak,
                        return_state=PipelineState.IDLE,
                    )
            except (
                AudioDeviceError,
                OllamaError,
                UnsupportedLanguageError,
                FileNotFoundError,
                ValueError,
                RuntimeError,
            ) as exc:
                if not self._state.stop_event.is_set():
                    self._state.transition(PipelineState.ERROR, str(exc))
                self._logger.error("Text pipeline error: %s", exc)
                if self._on_error is not None:
                    self._on_error(exc)
                if not self._state.stop_event.is_set():
                    self._state.transition(PipelineState.IDLE, "text_error_recovered")
                raise
            finally:
                if was_stopped and self._state.state not in (
                    PipelineState.IDLE,
                    PipelineState.STOPPING,
                ):
                    self._state.transition(PipelineState.IDLE, "text_turn_finished")

    def _sanitize_wake_transcript(self, transcript: str) -> str:
        """Prevent bare wake phrases from triggering an unintended LLM turn after activation."""
        raw = transcript.strip()
        if not raw:
            return ""
        phrase = self._config.wakeword.wake_phrase.strip()
        if not phrase:
            return raw
        tokens = [re.escape(tok) for tok in re.findall(r"[A-Za-z0-9\u0B80-\u0BFF]+", phrase)]
        if not tokens:
            return raw
        pattern = r"^\s*" + r"[\s,.:;!?-]*".join(tokens) + r"[\s,.:;!?-]*"
        stripped = re.sub(pattern, "", raw, count=1, flags=re.IGNORECASE).strip()
        if stripped != raw:
            return stripped
        # Handle common phonetic STT variants of "Hey Jarvis" right after wake activation.
        if phrase.lower() == "hey jarvis":
            variant = re.sub(
                r"^\s*(?:hey|hi|ok|okay)?[\s,.:;!?-]*(?:jarvis|jarvish|jarvas|javis|jarves)[\s,.:;!?-]*",
                "",
                raw,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
            return variant
        return raw

    def capture_enrollment_utterance(self, *, timeout_s: float = 6.5) -> np.ndarray | None:
        """Capture one short wake-phrase utterance on the shared mic stream for voice enrollment."""
        self._vad.load()
        started_locally = False
        if not getattr(self._capture, "is_running", False):
            self._capture.start()
            started_locally = True
        try:
            if hasattr(self._capture, "pause"):
                self._capture.pause()
            self._reset_listening_audio()
            if hasattr(self._capture, "resume"):
                self._capture.resume()

            # Use a snappier endpointing profile (shorter silence hangover & lower VAD threshold)
            # so short wake phrases like "Hey Jarvis" store immediately after the user stops.
            enroll_cfg = self._config.model_copy(deep=True)
            enroll_cfg.vad.threshold = min(enroll_cfg.vad.threshold, 0.35)
            enroll_cfg.vad.min_speech_duration_ms = min(
                enroll_cfg.vad.min_speech_duration_ms, 128
            )
            enroll_cfg.pipeline.vad_hangover_ms = min(
                enroll_cfg.pipeline.vad_hangover_ms, 480
            )
            enroll_cfg.pipeline.min_utterance_ms = min(
                enroll_cfg.pipeline.min_utterance_ms, 192
            )
            enroll_cfg.pipeline.max_utterance_s = min(
                enroll_cfg.pipeline.max_utterance_s, 4.5
            )
            enroll_detector = SpeechEndpointDetector(self._vad, enroll_cfg)

            captured = enroll_detector.capture_utterance(
                self._capture,
                stop_event=threading.Event(),
                state_machine=None,
                timeout_s=timeout_s,
                on_audio_level=self._on_audio_level,
            )
            if captured is None:
                return None
            waveform, _ = captured
            return waveform
        finally:
            if hasattr(self._capture, "pause"):
                self._capture.pause()
            self._reset_listening_audio()
            if started_locally:
                self._capture.stop()

    def _speak_wake_ack(self) -> None:
        """Play short local acknowledgment prompt with microphone paused to prevent echo."""
        ack_text = self._config.wakeword.ack_text.strip()
        if not ack_text or self._state.stop_event.is_set():
            return
        if self._cached_ack_audio is None:
            waveform, sample_rate, _ = self._tts.synthesize(
                ack_text,
                stt_language="en",
                backend_override=self._tts_backend,
            )
            self._cached_ack_audio = (waveform, sample_rate)
        waveform, sample_rate = self._cached_ack_audio
        if hasattr(self._player, "reset_cancel"):
            self._player.reset_cancel()
        self._state.transition(PipelineState.SPEAKING, "wake_ack")
        self._player.play(waveform, sample_rate, stop_event=self._state.stop_event)

    def wait_for_wake_word(
        self,
        *,
        timeout_s: float | None = None,
    ) -> tuple[WakeWordDetectionEvent, list[np.ndarray]] | None:
        """Listen in SLEEPING standby on the shared mic stream until wake phrase triggers."""
        detector = self.wakeword_detector
        detector.load()
        detector.reset()

        if hasattr(self._capture, "pause"):
            self._capture.pause()
        self._reset_listening_audio()
        self._state.clear_barge_in()
        if hasattr(self._capture, "resume"):
            self._capture.resume()

        self._state.transition(PipelineState.SLEEPING, "standby_wakeword")

        sr = self._config.audio.sample_rate
        chunk_size = self._config.audio.chunk_size
        frame_ms = (chunk_size / sr) * 1000.0
        pre_roll_frames = max(1, round(self._config.wakeword.pre_roll_ms / max(frame_ms, 1.0)))
        pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames)
        start_wait = time.monotonic()
        last_level_at = 0.0

        while not self._state.stop_event.is_set():
            if not self._config.wakeword.enabled:
                return None
            if timeout_s is not None and (time.monotonic() - start_wait) >= timeout_s:
                return None

            chunk = self._capture.read_chunk(timeout_s=0.1)
            if chunk is None:
                continue

            pre_roll.append(chunk)
            if self._on_audio_level is not None:
                now = time.monotonic()
                if (now - last_level_at) >= 0.06:
                    last_level_at = now
                    rms, bars = compute_waveform_bars(chunk)
                    self._on_audio_level("mic", rms, bars)

            event = detector.process_chunk(chunk, sample_rate=sr)
            if event is not None:
                return event, list(pre_roll)

        return None

    def _handle_wake_activation(
        self,
        event: WakeWordDetectionEvent,
        pre_roll_chunks: list[np.ndarray],
    ) -> list[np.ndarray] | None:
        """Handle WAKE_DETECTED -> optional ack -> grace period before LISTENING."""
        wake_start = time.perf_counter()
        self._state.transition(PipelineState.WAKE_DETECTED, "wake_word_detected")
        if self._on_wake_word is not None:
            self._on_wake_word(event)

        if hasattr(self._capture, "pause"):
            self._capture.pause()
        self._reset_listening_audio()

        used_ack = False
        if self._config.wakeword.ack_enabled and self._config.wakeword.ack_text.strip():
            used_ack = True
            self._speak_wake_ack()

        grace_ms = self._config.wakeword.activation_grace_ms
        if grace_ms > 0 and not self._state.stop_event.is_set():
            self._state.stop_event.wait(timeout=grace_ms / 1000.0)

        if self._state.stop_event.is_set():
            return None

        self._last_wake_to_listen_ms = round((time.perf_counter() - wake_start) * 1000.0, 2)
        if used_ack:
            return None
        return pre_roll_chunks if pre_roll_chunks else None

    def run_once(
        self,
        *,
        timeout_s: float | None = None,
        pre_roll_chunks: list[np.ndarray] | None = None,
    ) -> TurnMetrics | None:
        """Capture one utterance from the audio source and process a single turn."""
        seed_chunks = self._pending_barge_in_chunks
        self._pending_barge_in_chunks = None
        effective_timeout = (
            timeout_s if timeout_s is not None else self._config.pipeline.listen_timeout_s
        )

        if seed_chunks:
            self._state.clear_barge_in()
            self._echo_suppressor.reset()
            if self._capture.is_paused and hasattr(self._capture, "resume"):
                self._capture.resume()
            self._state.transition(PipelineState.LISTENING, "barge_in_follow_up")
            captured = self._endpoint_detector.capture_utterance(
                self._capture,
                stop_event=self._state.stop_event,
                state_machine=self._state,
                timeout_s=effective_timeout,
                initial_chunks=seed_chunks,
                already_in_speech=True,
                on_audio_level=self._on_audio_level,
            )
        else:
            if hasattr(self._capture, "pause"):
                self._capture.pause()
            self._reset_listening_audio()
            self._state.clear_barge_in()
            if hasattr(self._capture, "resume"):
                self._capture.resume()
            self._state.transition(PipelineState.LISTENING, "awaiting_speech")
            captured = self._endpoint_detector.capture_utterance(
                self._capture,
                stop_event=self._state.stop_event,
                state_machine=self._state,
                timeout_s=effective_timeout,
                initial_chunks=pre_roll_chunks,
                already_in_speech=False,
                on_audio_level=self._on_audio_level,
            )

        speech_end_at = time.perf_counter()
        if not self._config.pipeline.barge_in_enabled and hasattr(self._capture, "pause"):
            self._capture.pause()
        if captured is None:
            return None
        utterance, vad_ms = captured
        return self.process_utterance(
            utterance,
            vad_ms=vad_ms,
            sample_rate=self._config.audio.sample_rate,
            speech_end_at=speech_end_at,
            _transition_to_listening=False,
        )

    def run(
        self,
        *,
        max_turns: int | None = None,
        warmup: bool = True,
        shutdown_memory: bool = True,
        wake_word_mode: bool | None = None,
    ) -> list[TurnMetrics]:
        """Run continuous voice loop (optionally gated by offline Hey Jarvis wake word)."""
        completed: list[TurnMetrics] = []
        use_wake = self._wake_word_mode if wake_word_mode is None else wake_word_mode
        self._state.stop_event.clear()
        self._state.clear_barge_in()
        if use_wake:
            self.wakeword_detector.load()
            self._wake_word_active = True
        if warmup:
            self.warmup(require_llm=not use_wake)

        self._capture.start()
        if hasattr(self._capture, "pause"):
            self._capture.pause()
        try:
            if not use_wake:
                while not self._state.stop_event.is_set():
                    if max_turns is not None and len(completed) >= max_turns:
                        break
                    turn_metrics = self.run_once()
                    if turn_metrics is not None:
                        completed.append(turn_metrics)
            else:
                while not self._state.stop_event.is_set():
                    if max_turns is not None and len(completed) >= max_turns:
                        break
                    if not self._config.wakeword.enabled:
                        break

                    detected = self.wait_for_wake_word()
                    if detected is None or self._state.stop_event.is_set():
                        break

                    event, pre_roll_chunks = detected
                    seed_pre_roll = self._handle_wake_activation(event, pre_roll_chunks)
                    if self._state.stop_event.is_set():
                        break

                    self._just_woke_up = True
                    last_activity = time.monotonic()

                    while not self._state.stop_event.is_set():
                        if max_turns is not None and len(completed) >= max_turns:
                            break
                        if not self._config.wakeword.enabled:
                            break

                        timeout_s = self._config.wakeword.conversation_timeout_s
                        elapsed = time.monotonic() - last_activity
                        remaining = timeout_s - elapsed
                        if remaining <= 0.0 and self._pending_barge_in_chunks is None:
                            self._state.transition(
                                PipelineState.RETURNING_TO_STANDBY,
                                "conversation_timeout",
                            )
                            self._reset_listening_audio()
                            self.wakeword_detector.reset()
                            break

                        turn_metrics = self.run_once(
                            timeout_s=max(0.15, remaining),
                            pre_roll_chunks=seed_pre_roll,
                        )
                        seed_pre_roll = None

                        if turn_metrics is not None:
                            completed.append(turn_metrics)
                            last_activity = time.monotonic()
                            self._just_woke_up = False
                        elif self._pending_barge_in_chunks is not None:
                            last_activity = time.monotonic()
                            continue
                        elif not self._state.stop_event.is_set():
                            if (time.monotonic() - last_activity) >= timeout_s:
                                self._state.transition(
                                    PipelineState.RETURNING_TO_STANDBY,
                                    "conversation_timeout",
                                )
                                self._reset_listening_audio()
                                self.wakeword_detector.reset()
                                break
        except KeyboardInterrupt:
            self._state.request_stop()
        finally:
            self._wake_word_active = False
            self._just_woke_up = False
            self._pending_barge_in_chunks = None
            self._state.request_stop()
            if hasattr(self._player, "cancel"):
                self._player.cancel()
            self._capture.stop()
            self._reference_buffer.clear()
            self._vad.reset_states()
            if self._wakeword is not None:
                self._wakeword.reset()
            if shutdown_memory and self._memory is not None:
                self._memory.shutdown(wait=True, timeout_s=5.0)
            self._state.transition(PipelineState.IDLE, "pipeline_stopped")
        return completed
