"""Thread-safe LocalVox Companion Runtime coordinating voice pipeline, memory, and UI state."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psutil
import yaml

from backend.app.history import (
    ConversationHistoryStore,
    ConversationSessionDetail,
    ConversationSessionSummary,
)
from backend.app.schemas import (
    ChatResponse,
    ConversationContinueResponse,
    ConversationListResponse,
    EngineStatus,
    HealthResponse,
    LiveMessageItem,
    MemoryCreateRequest,
    MemoryListResponse,
    MemoryMutationResponse,
    MemoryUpdateRequest,
    SettingsPayload,
    SettingsResponse,
    StateSnapshotResponse,
    SystemMonitorResponse,
    VoiceControlResponse,
)
from backend.config import AppConfig, _resolve_config_path, get_config
from backend.eval.benchmark import (
    detect_hardware_summary,
    get_disk_size_mb,
    get_process_memory_mb,
)
from backend.llm.ollama_client import OllamaError, _match_model_name
from backend.logger import get_logger
from backend.memory.manager import MemoryManager
from backend.memory.models import (
    MemoryCandidate,
    MemoryCategory,
    MemorySource,
    utc_now_iso,
)
from backend.pipeline.audio_io import AudioDeviceError
from backend.pipeline.barge_in import BargeInCapture
from backend.pipeline.conversation import ConversationTurn
from backend.pipeline.state import PipelineState, PipelineStateMachine, TurnMetrics
from backend.pipeline.voice_pipeline import VoicePipeline
from backend.tts.piper_tts import PiperTtsEngine
from backend.wakeword.enrollment import (
    MAX_SAMPLE_DURATION_S,
    MIN_SAMPLE_DURATION_S,
    MIN_SAMPLE_RMS,
    REQUIRED_ENROLLMENT_SAMPLES,
    EnrollmentSampleResult,
    PersonalizedVoiceProfile,
    VoiceProfileSummary,
)
from backend.wakeword.events import WakeWordDetectionEvent


class WebSocketBroadcaster:
    """Non-blocking fan-out hub pushing pipeline thread events onto asyncio WebSocket queues."""

    def __init__(self, max_queue_size: int = 128) -> None:
        self._max_queue_size = max_queue_size
        self._lock = threading.Lock()
        self._clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Bind or clear the active FastAPI asyncio event loop."""
        with self._lock:
            self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a connected WebSocket consumer queue."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._max_queue_size)
        with self._lock:
            self._clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Unregister a disconnected WebSocket consumer queue."""
        with self._lock:
            self._clients.discard(q)

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        """Schedule a thread-safe WebSocket broadcast without ever blocking caller threads."""
        event = {"type": event_type, "timestamp": utc_now_iso(), **payload}
        with self._lock:
            loop = self._loop
            has_clients = bool(self._clients)
        if loop is None or loop.is_closed() or not has_clients:
            return
        try:
            loop.call_soon_threadsafe(self._dispatch_on_loop, event)
        except RuntimeError:
            pass

    def _dispatch_on_loop(self, event: dict[str, Any]) -> None:
        with self._lock:
            queues = list(self._clients)
        is_audio = event.get("type") == "audio_level"
        for q in queues:
            if q.full():
                if is_audio:
                    # Drop telemetry frames first so control/transcript events never stall.
                    continue
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                continue


class CompanionRuntime:
    """Desktop orchestration layer wrapping VoicePipeline, MemoryManager, and HistoryStore."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        config_path: Path | str | None = None,
        pipeline: VoicePipeline | None = None,
        history_store: ConversationHistoryStore | None = None,
    ) -> None:
        self._config_path = _resolve_config_path(config_path)
        self._config = (
            config
            if config is not None
            else (
                get_config()
                if config_path is None
                else AppConfig.model_validate(
                    yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
                )
            )
        )
        self._logger = get_logger("localvox.companion", self._config)
        self._broadcaster = WebSocketBroadcaster()
        self._history = (
            history_store
            if history_store is not None
            else ConversationHistoryStore(self._config.paths.conversations_db_file)
        )

        # Prime CPU delta baseline once without sleeping.
        psutil.cpu_percent(interval=None)

        self._live_lock = threading.Lock()
        self._voice_lock = threading.Lock()
        self._settings_lock = threading.Lock()
        self._enroll_lock = threading.Lock()
        self._pending_enroll_templates: list[np.ndarray] = []
        self._pending_enroll_timbres: list[np.ndarray] = []
        self._active_session_id = uuid.uuid4().hex[:16]
        self._messages: deque[LiveMessageItem] = deque(maxlen=200)
        self._streaming_partial: str = ""
        self._last_metrics: dict[str, Any] | None = None
        self._last_wake_event: WakeWordDetectionEvent | None = None
        self._voice_thread: threading.Thread | None = None
        self._voice_wake_mode: bool = False
        self._mic_active = False
        self._warmed_up = False
        self._standalone_memory: MemoryManager | None = None

        if pipeline is None:
            sm = PipelineStateMachine(on_change=self._on_state_change)
            self._pipeline = VoicePipeline(
                config=self._config,
                state_machine=sm,
                wake_word_mode=self._config.wakeword.enabled,
                on_transcript=self._on_transcript,
                on_token=self._on_token,
                on_response=self._on_response,
                on_turn_complete=self._on_turn_complete,
                on_error=self._on_error,
                on_barge_in=self._on_barge_in,
                on_audio_level=self._on_audio_level,
                on_wake_word=self._on_wake_word,
            )
        else:
            self._pipeline = pipeline
            self._hook_injected_pipeline(pipeline)

    def _hook_injected_pipeline(self, pipeline: VoicePipeline) -> None:
        """Attach non-blocking UI observer callbacks onto an injected pipeline instance."""
        prev_state_cb = pipeline.state_machine._on_change

        def wrapped_state(prev: PipelineState, new: PipelineState, reason: str) -> None:
            if prev_state_cb is not None:
                prev_state_cb(prev, new, reason)
            self._on_state_change(prev, new, reason)

        pipeline.state_machine._on_change = wrapped_state

        prev_transcript = pipeline._on_transcript
        prev_token = pipeline._on_token
        prev_response = pipeline._on_response
        prev_complete = pipeline._on_turn_complete
        prev_error = pipeline._on_error
        prev_barge_in = pipeline._on_barge_in
        prev_level = pipeline._on_audio_level
        prev_wake = pipeline._on_wake_word

        def wrapped_transcript(text: str, lang: str | None) -> None:
            if prev_transcript is not None:
                prev_transcript(text, lang)
            self._on_transcript(text, lang)

        def wrapped_token(token: str) -> None:
            if prev_token is not None:
                prev_token(token)
            self._on_token(token)

        def wrapped_response(text: str) -> None:
            if prev_response is not None:
                prev_response(text)
            self._on_response(text)

        def wrapped_complete(metrics: TurnMetrics) -> None:
            if prev_complete is not None:
                prev_complete(metrics)
            self._on_turn_complete(metrics)

        def wrapped_error(exc: Exception) -> None:
            if prev_error is not None:
                prev_error(exc)
            self._on_error(exc)

        def wrapped_barge_in(capture: BargeInCapture) -> None:
            if prev_barge_in is not None:
                prev_barge_in(capture)
            self._on_barge_in(capture)

        def wrapped_level(source: str, rms: float, bars: list[float]) -> None:
            if prev_level is not None:
                prev_level(source, rms, bars)
            self._on_audio_level(source, rms, bars)

        def wrapped_wake(event: WakeWordDetectionEvent) -> None:
            if prev_wake is not None:
                prev_wake(event)
            self._on_wake_word(event)

        pipeline._on_transcript = wrapped_transcript
        pipeline._on_token = wrapped_token
        pipeline._on_response = wrapped_response
        pipeline._on_turn_complete = wrapped_complete
        pipeline._on_error = wrapped_error
        pipeline._on_barge_in = wrapped_barge_in
        pipeline._on_audio_level = wrapped_level
        pipeline._on_wake_word = wrapped_wake
        if hasattr(pipeline._player, "set_audio_level_callback"):
            pipeline._player.set_audio_level_callback(wrapped_level)

    @property
    def config(self) -> AppConfig:
        """Return active AppConfig."""
        return self._config

    @property
    def broadcaster(self) -> WebSocketBroadcaster:
        """Return WebSocket event hub."""
        return self._broadcaster

    @property
    def mic_active(self) -> bool:
        """Return True while the background microphone pipeline loop is active."""
        return bool(
            self._mic_active
            and self._voice_thread is not None
            and self._voice_thread.is_alive()
        )

    def _get_memory_manager(self) -> MemoryManager:
        if self._pipeline.memory_manager is not None:
            return self._pipeline.memory_manager
        if self._standalone_memory is None:
            self._standalone_memory = MemoryManager(
                self._config,
                ollama=self._pipeline._llm,
                foreground_busy=self._pipeline._foreground_llm_busy,
            )
        return self._standalone_memory

    # ------------------------------------------------------------------
    # Pipeline observer callbacks (invoked on pipeline / worker threads)
    # ------------------------------------------------------------------

    def _on_state_change(
        self,
        prev_state: PipelineState,
        new_state: PipelineState,
        reason: str,
    ) -> None:
        self._broadcaster.publish(
            "state_changed",
            {
                "state": new_state.value,
                "previous_state": prev_state.value,
                "reason": reason,
                "mic_active": self.mic_active,
            },
        )

    def _on_transcript(self, text: str, language: str | None) -> None:
        item = LiveMessageItem(
            id=uuid.uuid4().hex[:12],
            role="user",
            content=text,
            language=language,
            created_at=utc_now_iso(),
        )
        with self._live_lock:
            self._streaming_partial = ""
            self._messages.append(item)
        self._broadcaster.publish("transcript", {"message": item.model_dump()})

    def _on_token(self, token: str) -> None:
        with self._live_lock:
            self._streaming_partial += token
            partial = self._streaming_partial
        self._broadcaster.publish("response_token", {"token": token, "partial": partial})

    def _on_response(self, text: str) -> None:
        item = LiveMessageItem(
            id=uuid.uuid4().hex[:12],
            role="assistant",
            content=text,
            language=None,
            created_at=utc_now_iso(),
        )
        with self._live_lock:
            self._streaming_partial = ""
            self._messages.append(item)
        self._broadcaster.publish("response_complete", {"message": item.model_dump()})

    def _on_turn_complete(self, metrics: TurnMetrics) -> None:
        metrics_dict = metrics.model_dump()
        with self._live_lock:
            self._last_metrics = metrics_dict
            session_id = self._active_session_id
            for msg in reversed(self._messages):
                if msg.role == "assistant" and msg.content == metrics.response_text:
                    msg.language = metrics.tts_language
                    msg.metrics = metrics_dict
                    break
        try:
            self._history.record_turn(
                session_id,
                metrics.transcript,
                metrics.response_text,
                language=metrics.detected_language or metrics.tts_language,
                metrics=metrics_dict,
            )
        except Exception as exc:  # pragma: no cover - defensive disk error guard
            self._logger.error("Failed to persist conversation turn: %s", exc)
        self._broadcaster.publish(
            "turn_complete",
            {
                "session_id": session_id,
                "metrics": metrics_dict,
            },
        )

    def _on_barge_in(self, capture: BargeInCapture) -> None:
        with self._live_lock:
            self._streaming_partial = ""
        self._broadcaster.publish(
            "barge_in",
            {"detect_latency_ms": capture.detect_latency_ms},
        )

    def _on_error(self, exc: Exception) -> None:
        self._broadcaster.publish("error", {"message": str(exc)})

    def _on_audio_level(self, source: str, rms: float, bars: list[float]) -> None:
        payload: dict[str, Any] = {"source": source, "rms": rms, "bars": bars}
        if (
            source == "mic"
            and self._pipeline.state_machine.state == PipelineState.SLEEPING
            and hasattr(self._pipeline.wakeword_detector, "last_scores")
        ):
            generic_s, personal_s, thresh = self._pipeline.wakeword_detector.last_scores()
            payload["wake_score"] = round(max(generic_s, personal_s), 4)
            payload["generic_score"] = generic_s
            payload["personal_score"] = personal_s
            payload["wake_threshold"] = thresh
        self._broadcaster.publish("audio_level", payload)

    def _on_wake_word(self, event: WakeWordDetectionEvent) -> None:
        with self._live_lock:
            self._last_wake_event = event
        self._broadcaster.publish(
            "wake_detected",
            {"event": event.model_dump()},
        )

    # ------------------------------------------------------------------
    # Companion & Voice Control
    # ------------------------------------------------------------------

    def get_health(self) -> HealthResponse:
        """Return health probe payload without side effects."""
        return HealthResponse(
            status="ok",
            pipeline_state=self._pipeline.state_machine.state.value,
            mic_active=self.mic_active,
        )

    def get_state_snapshot(self) -> StateSnapshotResponse:
        """Return current pipeline state, live transcript items, and wake-word telemetry."""
        with self._live_lock:
            messages = [msg.model_copy(deep=True) for msg in self._messages]
            partial = self._streaming_partial
            last_metrics = dict(self._last_metrics) if self._last_metrics is not None else None
            last_wake = self._last_wake_event
            session_id = self._active_session_id
        ww_info = self._pipeline.wakeword_detector.info()
        return StateSnapshotResponse(
            pipeline_state=self._pipeline.state_machine.state.value,
            mic_active=self.mic_active,
            session_id=session_id,
            streaming_partial=partial,
            messages=messages,
            last_metrics=last_metrics,
            wakeword_enabled=self._config.wakeword.enabled,
            wakeword_active=self.mic_active and self._voice_wake_mode,
            wake_phrase=self._config.wakeword.wake_phrase,
            last_wake_event=last_wake,
            wakeword_engine=ww_info,
            voice_profile=self.get_voice_profile(),
        )

    def send_text(
        self,
        text: str,
        *,
        speak: bool = True,
        language: str | None = None,
    ) -> ChatResponse:
        """Execute a chat turn synchronously in worker thread and optionally resume mic."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Message text cannot be empty.")

        was_mic_active = self.mic_active
        prev_mode: Literal["wakeword", "direct"] = (
            "wakeword" if self._voice_wake_mode else "direct"
        )
        if was_mic_active:
            self.stop_voice()

        try:
            metrics = self._pipeline.process_text(
                clean_text,
                speak=speak,
                language=language,
            )
        finally:
            if was_mic_active:
                self.start_voice(mode=prev_mode)

        with self._live_lock:
            session_id = self._active_session_id

        response_text = metrics.response_text if metrics is not None else ""
        metrics_dict = metrics.model_dump() if metrics is not None else None
        return ChatResponse(
            session_id=session_id,
            response_text=response_text,
            metrics=metrics_dict,
            pipeline_state=self._pipeline.state_machine.state.value,
        )

    def _run_voice_worker(self, *, wake_word_mode: bool) -> None:
        try:
            self._pipeline.run(
                warmup=not self._warmed_up,
                shutdown_memory=False,
                wake_word_mode=wake_word_mode,
            )
            self._warmed_up = True
        except (AudioDeviceError, OllamaError, Exception) as exc:
            self._logger.error("Voice loop error: %s", exc)
            self._broadcaster.publish("error", {"message": str(exc)})
        finally:
            with self._voice_lock:
                self._mic_active = False
                self._voice_wake_mode = False
            self._broadcaster.publish(
                "mic_status",
                {"mic_active": False, "wakeword_active": False},
            )

    def start_voice(
        self,
        *,
        mode: Literal["auto", "direct", "wakeword"] = "auto",
    ) -> VoiceControlResponse:
        """Start shared-microphone loop in hands-free wakeword standby or direct listen mode."""
        with self._voice_lock:
            if self.mic_active:
                return VoiceControlResponse(
                    status="already_running",
                    pipeline_state=self._pipeline.state_machine.state.value,
                    mic_active=True,
                    message="Microphone listening loop is already active.",
                )
            models_ok = self._pipeline.wakeword_detector.info().models_present
            if mode == "wakeword":
                use_wake = True
            elif mode == "direct":
                use_wake = False
            else:
                use_wake = bool(self._config.wakeword.enabled and models_ok)

            self._mic_active = True
            self._voice_wake_mode = use_wake
            self._voice_thread = threading.Thread(
                target=self._run_voice_worker,
                kwargs={"wake_word_mode": use_wake},
                name="localvox-companion-voice",
                daemon=True,
            )
            self._voice_thread.start()

        self._broadcaster.publish(
            "mic_status",
            {"mic_active": True, "wakeword_active": use_wake},
        )
        msg = (
            f"Hands-free standby active — say '{self._config.wakeword.wake_phrase}'."
            if use_wake
            else "Continuous microphone listening started."
        )
        return VoiceControlResponse(
            status="started",
            pipeline_state=self._pipeline.state_machine.state.value,
            mic_active=True,
            message=msg,
        )

    def set_wakeword_monitoring(self, enabled: bool) -> VoiceControlResponse:
        """Enable or immediately disable wake-word standby monitoring with privacy mic stop."""
        current = self.get_settings().settings
        updated = current.model_copy(update={"wakeword_enabled": enabled})
        self.update_settings(updated)
        if not enabled:
            return self.stop_voice()
        if self.mic_active and not self._voice_wake_mode:
            self.stop_voice()
        return self.start_voice(mode="wakeword")

    # ------------------------------------------------------------------
    # Personalized 3-Sample Voice Enrollment (100% local, zero raw audio saved)
    # ------------------------------------------------------------------

    def get_voice_profile(self) -> VoiceProfileSummary:
        """Return live personalized 3-sample wake-word voice profile status."""
        ww_cfg = self._config.wakeword
        detector = self._pipeline.wakeword_detector
        profile = (
            detector.get_personal_profile()
            if hasattr(detector, "get_personal_profile")
            else PersonalizedVoiceProfile.load(ww_cfg.personal_profile_path)
        )
        in_progress = len(self._pending_enroll_templates)
        return VoiceProfileSummary(
            enrolled=profile is not None,
            samples_recorded=profile.sample_count if profile is not None else 0,
            required_samples=REQUIRED_ENROLLMENT_SAMPLES,
            in_progress_samples=in_progress,
            wake_phrase=(
                profile.wake_phrase if profile is not None else ww_cfg.wake_phrase
            ),
            profile_path=str(ww_cfg.personal_profile_path),
            updated_at=profile.updated_at if profile is not None else None,
            personal_enabled=ww_cfg.personal_enabled,
            personal_only=ww_cfg.personal_only,
            personal_threshold=ww_cfg.effective_personal_threshold(),
            consistency_score=profile.consistency_score if profile is not None else None,
        )

    def start_voice_enrollment(self) -> VoiceProfileSummary:
        """Reset any partial enrollment attempt and prepare for 3 fresh voice samples."""
        with self._enroll_lock:
            self._pending_enroll_templates.clear()
            self._pending_enroll_timbres.clear()
        summary = self.get_voice_profile()
        self._broadcaster.publish("voice_profile_updated", {"profile": summary.model_dump()})
        return summary

    def record_voice_enrollment_sample(
        self,
        *,
        timeout_s: float = 6.5,
    ) -> EnrollmentSampleResult:
        """Record one wake-phrase sample on the shared mic, extract embeddings, discard raw PCM."""
        with self._enroll_lock:
            detector = self._pipeline.wakeword_detector
            if not hasattr(detector, "extract_enrollment_sample"):
                raise RuntimeError(
                    "Active wake-word detector does not support personalized voice enrollment."
                )

            if len(self._pending_enroll_templates) >= REQUIRED_ENROLLMENT_SAMPLES:
                self._pending_enroll_templates.clear()
                self._pending_enroll_timbres.clear()

            sample_index = len(self._pending_enroll_templates) + 1
            was_mic_active = self.mic_active
            prev_mode: Literal["wakeword", "direct"] = (
                "wakeword" if self._voice_wake_mode else "direct"
            )
            if was_mic_active:
                self.stop_voice()

            self._broadcaster.publish(
                "enrollment_status",
                {
                    "recording": True,
                    "sample_index": sample_index,
                    "required_samples": REQUIRED_ENROLLMENT_SAMPLES,
                },
            )
            try:
                waveform = self._pipeline.capture_enrollment_utterance(timeout_s=timeout_s)
                if waveform is None or waveform.size == 0:
                    raise ValueError(
                        f"No speech heard within {timeout_s:.0f}s. "
                        f"Speak '{self._config.wakeword.wake_phrase}' clearly and try again."
                    )

                sr = self._config.audio.sample_rate
                duration_s = round(float(waveform.size / sr), 3)
                audio_rms = round(
                    float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))),
                    5,
                )
                if duration_s < MIN_SAMPLE_DURATION_S:
                    raise ValueError(
                        "Sample was too short. Please say the full phrase clearly."
                    )
                if duration_s > MAX_SAMPLE_DURATION_S:
                    raise ValueError(
                        "Sample was too long. Say only the wake phrase (e.g., 'Hey Jarvis')."
                    )
                if audio_rms < MIN_SAMPLE_RMS:
                    raise ValueError(
                        "Voice was too quiet. Move closer to the microphone and speak clearly."
                    )

                template, timbre, generic_score = detector.extract_enrollment_sample(
                    waveform,
                    sample_rate=sr,
                )
                # Raw audio is never saved; discard immediately after mathematical extraction.
                del waveform

                consistency_prev: float | None = None
                if self._pending_enroll_templates:
                    sims = [
                        float(np.clip(np.dot(template, prev), 0.0, 1.0))
                        for prev in self._pending_enroll_templates
                    ]
                    consistency_prev = round(float(np.mean(sims)), 4)

                self._pending_enroll_templates.append(template)
                self._pending_enroll_timbres.append(timbre)
                self._logger.info(
                    "Voice enrollment sample %d/%d captured (dur_s=%.2f, rms=%.4f, generic=%.3f)",
                    sample_index,
                    REQUIRED_ENROLLMENT_SAMPLES,
                    duration_s,
                    audio_rms,
                    generic_score,
                )

                completed = len(self._pending_enroll_templates) >= REQUIRED_ENROLLMENT_SAMPLES
                if completed:
                    profile = PersonalizedVoiceProfile.from_samples(
                        self._pending_enroll_templates,
                        self._pending_enroll_timbres,
                        wake_phrase=self._config.wakeword.wake_phrase,
                    )
                    profile.save(self._config.wakeword.personal_profile_path)
                    if hasattr(detector, "set_personal_profile"):
                        detector.set_personal_profile(profile)
                    self._pending_enroll_templates.clear()
                    self._pending_enroll_timbres.clear()
                    consistency_pct = int(profile.consistency_score * 100)
                    msg = (
                        f"Voice profile saved! LocalVox now recognizes your voice for "
                        f"'{profile.wake_phrase}' (consistency {consistency_pct}%)."
                    )
                else:
                    remaining = REQUIRED_ENROLLMENT_SAMPLES - sample_index
                    msg = (
                        f"Sample {sample_index} of {REQUIRED_ENROLLMENT_SAMPLES} captured. "
                        f"Say '{self._config.wakeword.wake_phrase}' {remaining} more "
                        f"{'time' if remaining == 1 else 'times'}."
                    )

                summary = self.get_voice_profile()
                self._broadcaster.publish(
                    "voice_profile_updated",
                    {"profile": summary.model_dump()},
                )
                return EnrollmentSampleResult(
                    sample_index=sample_index,
                    required_samples=REQUIRED_ENROLLMENT_SAMPLES,
                    completed=completed,
                    duration_s=duration_s,
                    audio_rms=audio_rms,
                    generic_score=generic_score,
                    consistency_with_previous=consistency_prev,
                    message=msg,
                    profile=summary,
                )
            finally:
                self._broadcaster.publish(
                    "enrollment_status",
                    {
                        "recording": False,
                        "sample_index": sample_index,
                        "required_samples": REQUIRED_ENROLLMENT_SAMPLES,
                    },
                )
                if was_mic_active:
                    self.start_voice(mode=prev_mode)

    def delete_voice_profile(self) -> VoiceProfileSummary:
        """Delete local personalized voice embeddings and reset matcher cleanly."""
        with self._enroll_lock:
            self._pending_enroll_templates.clear()
            self._pending_enroll_timbres.clear()
            PersonalizedVoiceProfile.delete(self._config.wakeword.personal_profile_path)
            detector = self._pipeline.wakeword_detector
            if hasattr(detector, "set_personal_profile"):
                detector.set_personal_profile(None)
        summary = self.get_voice_profile()
        self._broadcaster.publish("voice_profile_updated", {"profile": summary.model_dump()})
        return summary

    def stop_voice(self) -> VoiceControlResponse:
        """Stop continuous microphone listening cleanly without closing persistent memory."""
        thread: threading.Thread | None
        with self._voice_lock:
            thread = self._voice_thread
            was_running = self.mic_active
            if was_running:
                self._pipeline.cancel_playback()
                self._pipeline.state_machine.request_stop()

        if thread is not None and thread.is_alive():
            thread.join(timeout=4.0)

        with self._voice_lock:
            self._mic_active = False
            self._voice_wake_mode = False
            self._voice_thread = None

        self._broadcaster.publish(
            "mic_status",
            {"mic_active": False, "wakeword_active": False},
        )
        return VoiceControlResponse(
            status="stopped" if was_running else "already_stopped",
            pipeline_state=self._pipeline.state_machine.state.value,
            mic_active=False,
            message="Microphone listening stopped.",
        )

    def stop_speaking(self) -> VoiceControlResponse:
        """Immediately cancel active LLM/TTS speech playback cleanly."""
        self._pipeline.cancel_playback()
        self._broadcaster.publish(
            "playback_stopped",
            {"pipeline_state": self._pipeline.state_machine.state.value},
        )
        return VoiceControlResponse(
            status="cancelled",
            pipeline_state=self._pipeline.state_machine.state.value,
            mic_active=self.mic_active,
            message="Active assistant speech stopped.",
        )

    # ------------------------------------------------------------------
    # Memory Dashboard API
    # ------------------------------------------------------------------

    def list_memories(
        self,
        *,
        query: str = "",
        category: MemoryCategory | None = None,
        limit: int = 100,
    ) -> MemoryListResponse:
        """Return total/per-category counts and matching SQLite memory rows."""
        mem = self._get_memory_manager()
        category_counts = {
            cat.value: mem.store.count(category=cat)
            for cat in MemoryCategory
        }
        total_count = sum(category_counts.values())
        clean_query = query.strip()
        match_scores: dict[str, float] = {}
        if clean_query:
            hits = mem.retriever.search(
                clean_query,
                category=category,
                limit=limit,
                min_score=0.05,
            )
            items = [hit.record for hit in hits]
            match_scores = {hit.record.id: round(hit.score, 4) for hit in hits}
        else:
            items = mem.store.list_memories(category=category, limit=limit)
        return MemoryListResponse(
            total_count=total_count,
            category_counts=category_counts,
            items=items,
            match_scores=match_scores,
        )

    def create_memory(self, payload: MemoryCreateRequest) -> MemoryMutationResponse:
        """Explicitly store a new memory record after privacy validation."""
        mem = self._get_memory_manager()
        candidate = MemoryCandidate(
            category=payload.category,
            subject=payload.subject,
            content=payload.content,
            keywords=payload.keywords,
            importance=payload.importance,
            confidence=1.0,
            source=MemorySource.USER_EXPLICIT,
            confirmed_by_user=True,
            session_id=self._active_session_id,
        )
        record, action, _ = mem.store.upsert(candidate)
        self._broadcaster.publish(
            "memory_changed",
            {"action": action, "memory_id": record.id},
        )
        return MemoryMutationResponse(status="ok", action=action, record=record)

    def update_memory(
        self,
        memory_id: str,
        payload: MemoryUpdateRequest,
    ) -> MemoryMutationResponse:
        """Update an existing SQLite memory record and reindex FTS5."""
        mem = self._get_memory_manager()
        updated = mem.store.update(
            memory_id,
            content=payload.content,
            category=payload.category,
            subject=payload.subject,
            keywords=payload.keywords,
            importance=payload.importance,
            source=MemorySource.USER_EXPLICIT,
            confirmed_by_user=True,
        )
        self._broadcaster.publish(
            "memory_changed",
            {"action": "updated", "memory_id": updated.id},
        )
        return MemoryMutationResponse(status="ok", action="updated", record=updated)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete one persisted memory record permanently."""
        mem = self._get_memory_manager()
        deleted = mem.store.delete(memory_id)
        if deleted:
            self._broadcaster.publish(
                "memory_changed",
                {"action": "deleted", "memory_id": memory_id},
            )
        return deleted

    def clear_all_memories(self, confirm_phrase: str) -> MemoryMutationResponse:
        """Wipe all memories only after matching exact confirmation phrase."""
        if confirm_phrase != "CLEAR_ALL_MEMORIES":
            raise ValueError("Confirmation phrase must equal CLEAR_ALL_MEMORIES.")
        mem = self._get_memory_manager()
        deleted_count = mem.store.clear_all(confirmed=True)
        self._broadcaster.publish(
            "memory_changed",
            {"action": "cleared_all", "deleted_count": deleted_count},
        )
        return MemoryMutationResponse(
            status="ok",
            action="cleared_all",
            deleted_count=deleted_count,
        )

    # ------------------------------------------------------------------
    # Conversation History API
    # ------------------------------------------------------------------

    def list_conversations(self, *, query: str = "", limit: int = 50) -> ConversationListResponse:
        """Return persisted real conversation sessions (never fabricated)."""
        items = self._history.list_sessions(query=query, limit=limit)
        return ConversationListResponse(
            items=items,
            active_session_id=self._active_session_id,
        )

    def get_conversation(self, session_id: str) -> ConversationSessionDetail | None:
        """Return complete message transcript for one saved session."""
        return self._history.get_session(session_id)

    def new_conversation(self) -> ConversationSessionSummary:
        """Start a fresh empty conversation session in memory until first turn arrives."""
        new_id = uuid.uuid4().hex[:16]
        self._pipeline.conversation.clear()
        with self._live_lock:
            self._active_session_id = new_id
            self._messages.clear()
            self._streaming_partial = ""
            self._last_metrics = None
        now = utc_now_iso()
        self._broadcaster.publish("session_changed", {"session_id": new_id, "loaded_turns": 0})
        return ConversationSessionSummary(
            id=new_id,
            title="New Conversation",
            created_at=now,
            updated_at=now,
            turn_count=0,
            preview="",
        )

    def continue_conversation(self, session_id: str) -> ConversationContinueResponse | None:
        """Load persisted session messages back into active ConversationManager and UI."""
        detail = self._history.get_session(session_id)
        if detail is None:
            return None

        turns: list[tuple[str, str]] = []
        pending_user: str | None = None
        live_items: list[LiveMessageItem] = []

        for msg in detail.messages:
            live_items.append(
                LiveMessageItem(
                    id=f"hist-{msg.id}",
                    role="user" if msg.role == "user" else "assistant",
                    content=msg.content,
                    language=msg.language,
                    created_at=msg.created_at,
                    metrics=msg.metrics,
                )
            )
            if msg.role == "user":
                pending_user = msg.content
            elif msg.role == "assistant" and pending_user is not None:
                turns.append((pending_user, msg.content))
                pending_user = None

        conv = self._pipeline.conversation
        conv.clear()
        max_turns = self._config.pipeline.max_history_turns
        loaded_pairs = turns[-max_turns:]
        for user_txt, assistant_txt in loaded_pairs:
            conv._turns.append(ConversationTurn(user=user_txt, assistant=assistant_txt))

        with self._live_lock:
            self._active_session_id = detail.session.id
            self._messages.clear()
            for item in live_items[-200:]:
                self._messages.append(item)
            self._streaming_partial = ""
            self._last_metrics = next(
                (msg.metrics for msg in reversed(live_items) if msg.metrics),
                None,
            )

        self._broadcaster.publish(
            "session_changed",
            {"session_id": detail.session.id, "loaded_turns": len(loaded_pairs)},
        )
        return ConversationContinueResponse(
            detail=detail,
            loaded_turns=len(loaded_pairs),
            active_session_id=detail.session.id,
        )

    def delete_conversation(self, session_id: str) -> bool:
        """Delete a saved conversation session permanently."""
        deleted = self._history.delete_session(session_id)
        if deleted:
            with self._live_lock:
                is_active = self._active_session_id == session_id
            if is_active:
                self.new_conversation()
            self._broadcaster.publish("session_deleted", {"session_id": session_id})
        return deleted

    # ------------------------------------------------------------------
    # System Monitor API (real telemetry only)
    # ------------------------------------------------------------------

    def _read_recent_latencies(self, limit: int = 25) -> list[dict[str, Any]]:
        path = self._config.logging.latency_log_file
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        records: list[dict[str, Any]] = []
        for raw in lines[-limit:]:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    records.append(parsed)
            except json.JSONDecodeError:
                continue
        records.reverse()
        return records

    def _is_whisper_loaded(self) -> bool:
        mod = sys.modules.get("mlx_whisper.transcribe")
        if mod is None:
            return False
        holder = getattr(mod, "ModelHolder", None)
        if holder is None or getattr(holder, "model", None) is None:
            return False
        try:
            resolved = str(self._config.stt.local_model_path.resolve())
        except OSError:
            return False
        return str(getattr(holder, "model_path", "")) == resolved

    def _list_ollama_model_names(self) -> list[str]:
        raw_models = self._pipeline._llm.list_models()
        names: list[str] = []
        for item in raw_models:
            if isinstance(item, str) and item.strip():
                names.append(item.strip())
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("model") or "").strip()
                if name:
                    names.append(name)
        return names

    def get_system_monitor(self) -> SystemMonitorResponse:
        """Collect live hardware, local AI model, SQLite, worker, and latency telemetry."""
        vm = psutil.virtual_memory()
        cpu_pct = round(float(psutil.cpu_percent(interval=None)), 1)
        ram_total_gb = round(vm.total / (1024**3), 2)
        ram_used_gb = round((vm.total - vm.available) / (1024**3), 2)
        ram_pct = round(float(vm.percent), 1)
        rss_mb = get_process_memory_mb()

        llm = self._pipeline._llm
        ollama_connected = False
        available_models: list[str] = []
        qwen_available = False
        llm_mem_mb: float | None = None
        llm_detail = "Ollama daemon unreachable on local host."
        try:
            available_models = self._list_ollama_model_names()
            ollama_connected = True
            qwen_available = (
                _match_model_name(self._config.llm.model_name, available_models) is not None
            )
            llm_mem_mb = llm.get_running_memory_mb(self._config.llm.model_name)
            if qwen_available:
                llm_detail = (
                    "Loaded in local Ollama memory."
                    if llm_mem_mb is not None
                    else "Available offline in Ollama."
                )
            else:
                llm_detail = f"Model {self._config.llm.model_name} not pulled in Ollama."
        except OllamaError as exc:
            llm_detail = str(exc)

        llm_status = EngineStatus(
            name="Ollama Qwen LLM",
            status=(
                "loaded"
                if (ollama_connected and llm_mem_mb is not None)
                else ("ready" if qwen_available else ("missing" if ollama_connected else "offline"))
            ),
            detail=llm_detail,
            model_name=self._config.llm.model_name,
            loaded=llm_mem_mb is not None,
            memory_mb=llm_mem_mb,
        )

        vad = self._pipeline._vad
        vad_loaded = getattr(vad, "_model", None) is not None
        vad_size = vad.model_size_mb()
        vad_status = EngineStatus(
            name="Silero VAD",
            status="loaded" if vad_loaded else ("ready" if vad_size is not None else "missing"),
            detail="Loaded in memory." if vad_loaded else "Local bundled model ready.",
            model_name=self._config.vad.model_name,
            loaded=vad_loaded,
            disk_size_mb=vad_size,
        )

        stt = self._pipeline._stt
        whisper_dir = self._config.stt.local_model_path
        whisper_exists = whisper_dir.exists()
        whisper_loaded = self._is_whisper_loaded()
        whisper_size = stt.model_size_mb() if whisper_exists else None
        stt_status = EngineStatus(
            name="MLX Whisper STT",
            status=(
                "loaded"
                if whisper_loaded
                else ("ready" if whisper_exists else "missing")
            ),
            detail=(
                "Loaded in MLX ModelHolder."
                if whisper_loaded
                else ("Apple Silicon weights ready." if whisper_exists else "Weights missing.")
            ),
            model_name=self._config.stt.model_name,
            loaded=whisper_loaded,
            disk_size_mb=whisper_size,
        )

        kokoro_cfg = self._config.tts.kokoro
        kokoro_files_ok = kokoro_cfg.model_path.exists() and kokoro_cfg.voices_path.exists()
        kokoro_inst = getattr(self._pipeline._tts, "_kokoro", None)
        kokoro_loaded = (
            kokoro_inst is not None and getattr(kokoro_inst, "_engine", None) is not None
        )
        kokoro_size: float | None = None
        if kokoro_files_ok:
            m_mb = get_disk_size_mb(kokoro_cfg.model_path)
            v_mb = get_disk_size_mb(kokoro_cfg.voices_path)
            if m_mb is not None and v_mb is not None:
                kokoro_size = round(m_mb + v_mb, 2)
        kokoro_status = EngineStatus(
            name="Kokoro English TTS",
            status="loaded" if kokoro_loaded else ("ready" if kokoro_files_ok else "missing"),
            detail=(
                f"Active voice: {kokoro_cfg.default_voice}"
                if kokoro_files_ok
                else "Kokoro ONNX files missing."
            ),
            model_name=kokoro_cfg.model_name,
            loaded=kokoro_loaded,
            disk_size_mb=kokoro_size,
        )

        piper_cfg = self._config.tts.piper
        piper_files_ok = piper_cfg.ta_model_path.exists() and piper_cfg.ta_config_path.exists()
        piper_inst = getattr(self._pipeline._tts, "_piper", None)
        piper_loaded = bool(piper_inst is not None and getattr(piper_inst, "_voices", {}))
        piper_size = PiperTtsEngine(self._config).model_size_mb("ta") if piper_files_ok else None
        piper_status = EngineStatus(
            name="Piper Tamil TTS",
            status="loaded" if piper_loaded else ("ready" if piper_files_ok else "missing"),
            detail=(
                "Tamil ONNX voice ready locally."
                if piper_files_ok
                else "Piper Tamil ONNX model missing."
            ),
            model_name=piper_cfg.ta_model_name,
            loaded=piper_loaded,
            disk_size_mb=piper_size,
        )

        mem = self._get_memory_manager()
        integrity = mem.check_integrity()
        db_size = get_disk_size_mb(mem.database.db_path)
        sqlite_status = EngineStatus(
            name="SQLite + FTS5 Memory",
            status="loaded" if integrity.get("ok") else "error",
            detail=(
                f"{integrity.get('memory_count', 0)} memories indexed (FTS5 verified)."
                if integrity.get("ok")
                else "SQLite integrity check failed."
            ),
            model_name="SQLite WAL + FTS5",
            loaded=bool(integrity.get("ok")),
            disk_size_mb=db_size,
        )

        worker = mem.worker
        last_out = worker.last_outcome
        extraction_summary: dict[str, Any] = {
            "completed_jobs": worker.completed_jobs,
            "queue_size": worker._queue.qsize(),
            "idle": worker._idle_event.is_set(),
            "error_count": len(worker.errors),
            "last_outcome": (
                {
                    "candidates_parsed": last_out.candidates_parsed,
                    "actions": last_out.actions,
                    "skipped_privacy": last_out.skipped_privacy,
                    "skipped_low_value": last_out.skipped_low_value,
                    "extraction_ms": last_out.extraction_ms,
                    "write_ms": last_out.write_ms,
                }
                if last_out is not None
                else None
            ),
        }

        ww_info = self._pipeline.wakeword_detector.info()
        voice_tag = (
            f" · Personal Voice ({ww_info.personalized_samples}/3)"
            if ww_info.personalized_enrolled
            else ""
        )
        ww_status = EngineStatus(
            name=f"openWakeWord ({ww_info.wake_phrase})",
            status=(
                "loaded"
                if ww_info.loaded
                else ("ready" if ww_info.models_present else "missing")
            ),
            detail=(
                f"{ww_info.execution_provider} · "
                f"{ww_info.last_inference_ms:.2f} ms/frame · "
                f"{ww_info.total_detections} detections{voice_tag}"
                if ww_info.loaded and ww_info.last_inference_ms is not None
                else (
                    f"ONNX artifacts ready ({ww_info.execution_provider}){voice_tag}."
                    if ww_info.models_present
                    else "Run `make download-models` to fetch openWakeWord ONNX files."
                )
            ),
            model_name=ww_info.model_name,
            loaded=ww_info.loaded,
            disk_size_mb=ww_info.disk_size_mb,
        )

        return SystemMonitorResponse(
            hardware=detect_hardware_summary(),
            cpu_percent=cpu_pct,
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            ram_percent=ram_pct,
            process_rss_mb=rss_mb,
            ollama_connected=ollama_connected,
            qwen_available=qwen_available,
            available_ollama_models=available_models,
            engines=[
                llm_status,
                ww_status,
                vad_status,
                stt_status,
                kokoro_status,
                piper_status,
                sqlite_status,
            ],
            sqlite_integrity=integrity,
            sqlite_size_mb=db_size,
            extraction_worker=extraction_summary,
            recent_latencies=self._read_recent_latencies(),
            pipeline_state=self._pipeline.state_machine.state.value,
            mic_active=self.mic_active,
        )

    # ------------------------------------------------------------------
    # Settings API (allowlisted config fields only, paths protected)
    # ------------------------------------------------------------------

    def _list_kokoro_voices_safely(self) -> list[str]:
        kokoro_inst = getattr(self._pipeline._tts, "_kokoro", None)
        if kokoro_inst is not None and getattr(kokoro_inst, "_engine", None) is not None:
            return kokoro_inst.available_voices()
        voices_path = self._config.tts.kokoro.voices_path
        if voices_path.exists():
            try:
                with np.load(voices_path, allow_pickle=False) as archive:
                    return sorted(str(k) for k in archive.keys())
            except Exception:
                pass
        return [self._config.tts.kokoro.default_voice]

    def get_settings(self) -> SettingsResponse:
        """Return safe subset of active settings plus locally available voices and models."""
        cfg = self._config
        payload = SettingsPayload(
            kokoro_voice=cfg.tts.kokoro.default_voice,
            kokoro_speed=cfg.tts.kokoro.speed,
            piper_length_scale=cfg.tts.piper.length_scale,
            stt_default_language=cfg.stt.default_language,
            tts_routing=cfg.pipeline.tts_routing,
            streaming_tts_enabled=cfg.pipeline.streaming_tts_enabled,
            barge_in_enabled=cfg.pipeline.barge_in_enabled,
            barge_in_sensitivity=cfg.pipeline.barge_in_sensitivity,
            llm_model_name=cfg.llm.model_name,
            llm_temperature=cfg.llm.temperature,
            max_history_turns=cfg.pipeline.max_history_turns,
            memory_enabled=cfg.memory.enabled,
            memory_auto_extract=cfg.memory.auto_extract,
            memory_privacy_filter_enabled=cfg.memory.privacy_filter_enabled,
            wakeword_enabled=cfg.wakeword.enabled,
            wake_phrase=cfg.wakeword.wake_phrase,
            wakeword_sensitivity=cfg.wakeword.sensitivity,
            wakeword_threshold=cfg.wakeword.threshold,
            wakeword_conversation_timeout_s=cfg.wakeword.conversation_timeout_s,
            wakeword_ack_enabled=cfg.wakeword.ack_enabled,
            wakeword_activation_grace_ms=cfg.wakeword.activation_grace_ms,
            wakeword_personal_enabled=cfg.wakeword.personal_enabled,
            wakeword_personal_only=cfg.wakeword.personal_only,
            wakeword_personal_threshold=cfg.wakeword.personal_threshold,
        )
        try:
            ollama_models = self._list_ollama_model_names()
        except OllamaError:
            ollama_models = [cfg.llm.model_name]
        return SettingsResponse(
            settings=payload,
            available_kokoro_voices=self._list_kokoro_voices_safely(),
            available_ollama_models=ollama_models,
            wakeword_engine=self._pipeline.wakeword_detector.info(),
            voice_profile=self.get_voice_profile(),
            mic_active=self.mic_active,
        )

    def update_settings(self, payload: SettingsPayload) -> SettingsResponse:
        """Validate, apply in-memory, and persist allowlisted desktop settings to YAML."""
        should_stop_standby = False
        should_start_standby = False
        with self._settings_lock:
            prev_wake_enabled = self._config.wakeword.enabled
            if self._config_path.exists():
                raw = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
            else:
                raw = self._config.model_dump(mode="json")

            raw["tts"]["kokoro"]["default_voice"] = payload.kokoro_voice
            raw["tts"]["kokoro"]["speed"] = payload.kokoro_speed
            raw["tts"]["piper"]["length_scale"] = payload.piper_length_scale
            raw["stt"]["default_language"] = payload.stt_default_language
            raw["pipeline"]["tts_routing"] = payload.tts_routing
            raw["pipeline"]["streaming_tts_enabled"] = payload.streaming_tts_enabled
            raw["pipeline"]["barge_in_enabled"] = payload.barge_in_enabled
            raw["pipeline"]["barge_in_sensitivity"] = payload.barge_in_sensitivity
            raw["llm"]["model_name"] = payload.llm_model_name
            raw["llm"]["temperature"] = payload.llm_temperature
            raw["pipeline"]["max_history_turns"] = payload.max_history_turns
            raw["memory"]["enabled"] = payload.memory_enabled
            raw["memory"]["auto_extract"] = payload.memory_auto_extract
            raw["memory"]["privacy_filter_enabled"] = payload.memory_privacy_filter_enabled

            ww_raw = raw.setdefault("wakeword", self._config.wakeword.model_dump(mode="json"))
            ww_raw["enabled"] = payload.wakeword_enabled
            ww_raw["wake_phrase"] = payload.wake_phrase
            ww_raw["sensitivity"] = payload.wakeword_sensitivity
            ww_raw["threshold"] = payload.wakeword_threshold
            ww_raw["conversation_timeout_s"] = payload.wakeword_conversation_timeout_s
            ww_raw["ack_enabled"] = payload.wakeword_ack_enabled
            ww_raw["activation_grace_ms"] = payload.wakeword_activation_grace_ms
            ww_raw["personal_enabled"] = payload.wakeword_personal_enabled
            ww_raw["personal_only"] = payload.wakeword_personal_only
            ww_raw["personal_threshold"] = payload.wakeword_personal_threshold

            new_config = AppConfig.model_validate(raw)

            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._config_path.with_suffix(".yaml.tmp")
            tmp_path.write_text(
                yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            os.replace(tmp_path, self._config_path)

            get_config.cache_clear()
            self._config = new_config
            self._pipeline.set_config(new_config)
            if self._standalone_memory is not None:
                self._standalone_memory._config = new_config
                self._standalone_memory._store._config = new_config
                self._standalone_memory._retriever._config = new_config
                self._standalone_memory._extractor._config = new_config
                self._standalone_memory._commands._config = new_config

            if prev_wake_enabled and not payload.wakeword_enabled and self._voice_wake_mode:
                should_stop_standby = True
            elif (
                not prev_wake_enabled
                and payload.wakeword_enabled
                and not self.mic_active
                and self._pipeline.wakeword_detector.info().models_present
            ):
                should_start_standby = True

        if should_stop_standby:
            self.stop_voice()
        elif should_start_standby:
            self.start_voice(mode="wakeword")

        response = self.get_settings()
        self._broadcaster.publish("settings_updated", {"settings": response.settings.model_dump()})
        return response

    def shutdown(self) -> None:
        """Stop voice loop cleanly and drain persistent SQLite stores without data loss."""
        self.stop_voice()
        if self._pipeline.memory_manager is not None:
            self._pipeline.memory_manager.shutdown(wait=True, timeout_s=5.0)
        if self._standalone_memory is not None:
            self._standalone_memory.shutdown(wait=True, timeout_s=5.0)
        self._history.close()
