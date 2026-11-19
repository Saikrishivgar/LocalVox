"""Pipeline state machine, turn metrics, and barge-in coordination."""

import threading
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class PipelineState(StrEnum):
    """Execution states of the LocalVox voice assistant pipeline."""

    IDLE = "IDLE"
    SLEEPING = "SLEEPING"
    WAKE_DETECTED = "WAKE_DETECTED"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    GENERATING = "GENERATING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    RETURNING_TO_STANDBY = "RETURNING_TO_STANDBY"
    ERROR = "ERROR"
    STOPPING = "STOPPING"


class InvalidStateTransitionError(ValueError):
    """Raised when an invalid pipeline state transition is requested."""


VALID_TRANSITIONS: dict[PipelineState, frozenset[PipelineState]] = {
    PipelineState.IDLE: frozenset(
        {
            PipelineState.IDLE,
            PipelineState.SLEEPING,
            PipelineState.LISTENING,
            PipelineState.PROCESSING,
            PipelineState.ERROR,
            PipelineState.STOPPING,
        }
    ),
    PipelineState.SLEEPING: frozenset(
        {
            PipelineState.SLEEPING,
            PipelineState.WAKE_DETECTED,
            PipelineState.LISTENING,
            PipelineState.PROCESSING,
            PipelineState.ERROR,
            PipelineState.STOPPING,
            PipelineState.IDLE,
        }
    ),
    PipelineState.WAKE_DETECTED: frozenset(
        {
            PipelineState.WAKE_DETECTED,
            PipelineState.LISTENING,
            PipelineState.SPEAKING,
            PipelineState.RETURNING_TO_STANDBY,
            PipelineState.SLEEPING,
            PipelineState.ERROR,
            PipelineState.STOPPING,
            PipelineState.IDLE,
        }
    ),
    PipelineState.LISTENING: frozenset(
        {
            PipelineState.LISTENING,
            PipelineState.PROCESSING,
            PipelineState.INTERRUPTED,
            PipelineState.RETURNING_TO_STANDBY,
            PipelineState.SLEEPING,
            PipelineState.ERROR,
            PipelineState.STOPPING,
            PipelineState.IDLE,
        }
    ),
    PipelineState.PROCESSING: frozenset(
        {
            PipelineState.PROCESSING,
            PipelineState.GENERATING,
            PipelineState.SPEAKING,
            PipelineState.LISTENING,
            PipelineState.INTERRUPTED,
            PipelineState.RETURNING_TO_STANDBY,
            PipelineState.SLEEPING,
            PipelineState.ERROR,
            PipelineState.STOPPING,
            PipelineState.IDLE,
        }
    ),
    PipelineState.GENERATING: frozenset(
        {
            PipelineState.GENERATING,
            PipelineState.SPEAKING,
            PipelineState.INTERRUPTED,
            PipelineState.LISTENING,
            PipelineState.RETURNING_TO_STANDBY,
            PipelineState.SLEEPING,
            PipelineState.ERROR,
            PipelineState.STOPPING,
            PipelineState.IDLE,
        }
    ),
    PipelineState.SPEAKING: frozenset(
        {
            PipelineState.SPEAKING,
            PipelineState.LISTENING,
            PipelineState.INTERRUPTED,
            PipelineState.RETURNING_TO_STANDBY,
            PipelineState.SLEEPING,
            PipelineState.ERROR,
            PipelineState.STOPPING,
            PipelineState.IDLE,
        }
    ),
    PipelineState.INTERRUPTED: frozenset(
        {
            PipelineState.INTERRUPTED,
            PipelineState.LISTENING,
            PipelineState.PROCESSING,
            PipelineState.RETURNING_TO_STANDBY,
            PipelineState.SLEEPING,
            PipelineState.ERROR,
            PipelineState.STOPPING,
            PipelineState.IDLE,
        }
    ),
    PipelineState.RETURNING_TO_STANDBY: frozenset(
        {
            PipelineState.RETURNING_TO_STANDBY,
            PipelineState.SLEEPING,
            PipelineState.IDLE,
            PipelineState.ERROR,
            PipelineState.STOPPING,
        }
    ),
    PipelineState.ERROR: frozenset(
        {
            PipelineState.ERROR,
            PipelineState.SLEEPING,
            PipelineState.LISTENING,
            PipelineState.RETURNING_TO_STANDBY,
            PipelineState.STOPPING,
            PipelineState.IDLE,
        }
    ),
    PipelineState.STOPPING: frozenset(
        {
            PipelineState.STOPPING,
            PipelineState.IDLE,
            PipelineState.ERROR,
        }
    ),
}


class TurnMetrics(BaseModel):
    """End-to-end latency and metadata recorded for a single voice turn."""

    model_config = ConfigDict(extra="forbid")

    turn_id: int
    transcript: str
    detected_language: str | None
    response_text: str
    tts_backend: str
    tts_language: str
    vad_ms: float
    stt_ms: float
    llm_ttft_ms: float
    llm_total_ms: float
    llm_tokens_per_sec: float
    tts_ms: float
    playback_ms: float
    response_latency_ms: float
    total_turn_ms: float
    memory_mb: float
    interrupted: bool = False
    speech_end_to_stt_ms: float = 0.0
    tts_ttfa_ms: float = 0.0
    first_audio_latency_ms: float = 0.0
    interruption_detect_ms: float | None = None
    playback_cancel_ms: float | None = None
    tts_chunks: int = 1
    barge_in_triggered: bool = False
    memory_retrieval_ms: float = 0.0
    memories_retrieved: int = 0
    memory_context_chars: int = 0


class PipelineStateMachine:
    """Thread-safe pipeline state manager with transition validation and turn cancellation."""

    def __init__(
        self,
        on_change: Callable[[PipelineState, PipelineState, str], None] | None = None,
    ) -> None:
        self._state = PipelineState.IDLE
        self._lock = threading.Lock()
        self._on_change = on_change
        self.stop_event = threading.Event()
        self.barge_in_event = threading.Event()
        self._generation_id = 0
        self._active_cancel_event: threading.Event | None = None

    @property
    def state(self) -> PipelineState:
        """Return the current pipeline state."""
        with self._lock:
            return self._state

    @property
    def generation_id(self) -> int:
        """Return the monotonic active turn generation identifier."""
        with self._lock:
            return self._generation_id

    def begin_generation(self) -> tuple[int, threading.Event]:
        """Advance turn generation and bind a fresh per-turn cancellation event."""
        with self._lock:
            self._generation_id += 1
            self.barge_in_event.clear()
            cancel_event = threading.Event()
            if self.stop_event.is_set():
                cancel_event.set()
            self._active_cancel_event = cancel_event
            return self._generation_id, cancel_event

    def is_generation_current(self, gen_id: int) -> bool:
        """Return True if gen_id matches the active turn and cancellation was not signalled."""
        with self._lock:
            active_cancelled = (
                self._active_cancel_event.is_set()
                if self._active_cancel_event is not None
                else False
            )
            return (
                self._generation_id == gen_id
                and not active_cancelled
                and not self.barge_in_event.is_set()
                and not self.stop_event.is_set()
            )

    def transition(self, new_state: PipelineState, reason: str = "") -> PipelineState:
        """Transition atomically to new_state after validating edge legality."""
        with self._lock:
            prev = self._state
            allowed = VALID_TRANSITIONS.get(prev, frozenset())
            if new_state not in allowed:
                raise InvalidStateTransitionError(
                    f"Invalid pipeline state transition: {prev.value} -> {new_state.value}"
                    + (f" ({reason})" if reason else "")
                )
            self._state = new_state
        if self._on_change is not None and prev != new_state:
            self._on_change(prev, new_state, reason)
        return new_state

    def request_barge_in(self) -> None:
        """Signal active playback or generation to interrupt cleanly."""
        with self._lock:
            self.barge_in_event.set()
            if self._active_cancel_event is not None:
                self._active_cancel_event.set()

    def cancel_playback(self) -> None:
        """Cancel active TTS playback and active generation cleanly."""
        self.request_barge_in()

    def clear_barge_in(self) -> None:
        """Reset barge-in flag before starting a new listening or speaking phase."""
        self.barge_in_event.clear()

    def can_accept_speech(self) -> bool:
        """Return True only when the pipeline is LISTENING and not stopping."""
        return self.state == PipelineState.LISTENING and not self.stop_event.is_set()

    def is_standby(self) -> bool:
        """Return True when the pipeline is in SLEEPING wake-word standby."""
        return self.state == PipelineState.SLEEPING and not self.stop_event.is_set()

    def request_stop(self) -> None:
        """Signal the pipeline loop and active generation/playback to stop."""
        with self._lock:
            self.stop_event.set()
            self.barge_in_event.set()
            if self._active_cancel_event is not None:
                self._active_cancel_event.set()
        if self.state not in (PipelineState.IDLE, PipelineState.STOPPING):
            self.transition(PipelineState.STOPPING, "stop_requested")
