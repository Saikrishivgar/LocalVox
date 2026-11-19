"""Comprehensive Phase 6 automated tests covering all 20 wake-word & hands-free requirements."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.runtime import CompanionRuntime
from backend.config import AppConfig, get_config
from backend.llm.ollama_client import LlmResult, OllamaClient, OllamaError
from backend.pipeline.echo import resample_mono
from backend.pipeline.state import (
    InvalidStateTransitionError,
    PipelineState,
    PipelineStateMachine,
)
from backend.pipeline.voice_pipeline import VoicePipeline
from backend.stt.whisper_stt import MlxWhisperStt, SttResult
from backend.tts.kokoro_tts import KokoroTtsEngine
from backend.wakeword import (
    OWW_STEP_SAMPLES,
    OpenWakeWordEngine,
    WakeWordDetectionEvent,
    WakeWordEngineInfo,
    WakeWordError,
    WakeWordModelMissingError,
    check_wakeword_models,
    ensure_wakeword_models_exist,
)
from tests.test_phase3 import (
    LiveQueueCapture,
    StreamRecordingPlayer,
    _fake_vad,
    _make_tone_frames,
    _mock_tts_engines,
)


def _phase6_config(tmp_path: Path, *, use_real_wake_models: bool = True) -> tuple[Path, AppConfig]:
    """Build isolated Phase 6 AppConfig inside tmp_path with fast deterministic test timings."""
    raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    db_file = str(tmp_path / "memory.db")
    conv_file = str(tmp_path / "conversations.db")
    latency_file = str(tmp_path / "latency.jsonl")
    log_file = str(tmp_path / "app.log")

    raw["logging"]["app_log_file"] = log_file
    raw["logging"]["latency_log_file"] = latency_file
    raw["paths"]["app_log_file"] = log_file
    raw["paths"]["latency_log_file"] = latency_file
    raw["paths"]["memory_db_file"] = db_file
    raw["paths"]["conversations_db_file"] = conv_file
    raw["memory"]["db_path"] = db_file
    raw["memory"]["enabled"] = True
    raw["memory"]["auto_extract"] = False
    raw["tts"]["output_dir"] = str(tmp_path / "tts_out")
    raw["tts"]["kokoro"]["model_path"] = str(tmp_path / "kokoro.onnx")
    raw["tts"]["kokoro"]["voices_path"] = str(tmp_path / "voices.bin")
    raw["tts"]["piper"]["en_model_path"] = str(tmp_path / "en.onnx")
    raw["tts"]["piper"]["en_config_path"] = str(tmp_path / "en.onnx.json")
    raw["tts"]["piper"]["ta_model_path"] = str(tmp_path / "ta.onnx")
    raw["tts"]["piper"]["ta_config_path"] = str(tmp_path / "ta.onnx.json")
    raw["pipeline"]["pre_speech_pad_ms"] = 64
    raw["pipeline"]["post_speech_pad_ms"] = 64
    raw["pipeline"]["vad_hangover_ms"] = 128
    raw["pipeline"]["min_utterance_ms"] = 96
    raw["vad"]["min_speech_duration_ms"] = 64
    raw["pipeline"]["barge_in_enabled"] = True
    raw["pipeline"]["barge_in_min_speech_ms"] = 96
    raw["pipeline"]["streaming_tts_enabled"] = True
    raw["pipeline"]["tts_min_sentence_chars"] = 16
    raw["pipeline"]["tts_chunk_gap_ms"] = 0

    if not use_real_wake_models:
        ww_dir = tmp_path / "missing_wakeword"
        raw["paths"]["wakeword_models_dir"] = str(ww_dir)
        raw["wakeword"]["models_dir"] = str(ww_dir)
        raw["wakeword"]["melspectrogram_model"] = str(ww_dir / "melspectrogram.onnx")
        raw["wakeword"]["embedding_model"] = str(ww_dir / "embedding_model.onnx")
        raw["wakeword"]["classifier_model"] = str(ww_dir / "hey_jarvis_v0.1.onnx")

    raw["wakeword"]["enabled"] = True
    raw["wakeword"]["ack_enabled"] = True
    raw["wakeword"]["activation_grace_ms"] = 20
    raw["wakeword"]["pre_roll_ms"] = 128
    raw["wakeword"]["conversation_timeout_s"] = 2.0
    raw["wakeword"]["personal_profile_path"] = str(tmp_path / "voice_profile.npz")
    raw["wakeword"]["personal_enabled"] = True
    raw["wakeword"]["personal_only"] = False
    raw["wakeword"]["personal_threshold"] = 0.72

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg_path, AppConfig.model_validate(raw)


class StubWakeWordDetector:
    """Deterministic WakeWordDetector stub triggering on high-amplitude chunks (> 0.65)."""

    def __init__(
        self,
        config: AppConfig,
        *,
        trigger_amp: float = 0.65,
        cooldown_s: float = 1.0,
    ) -> None:
        self._cfg = config
        self._trigger_amp = trigger_amp
        self._cooldown_s = cooldown_s
        self.loaded = False
        self.reset_count = 0
        self.chunks_seen = 0
        self.detections = 0
        self._last_trigger = 0.0
        self._lock = threading.Lock()

    def load(self) -> float:
        self.loaded = True
        return 1.2

    def reset(self) -> None:
        with self._lock:
            self.reset_count += 1

    def set_config(self, config: AppConfig) -> None:
        with self._lock:
            self._cfg = config

    def process_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int = 16000,
    ) -> WakeWordDetectionEvent | None:
        del sample_rate
        with self._lock:
            self.chunks_seen += 1
            peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
            now = time.monotonic()
            if peak >= self._trigger_amp and (now - self._last_trigger) >= self._cooldown_s:
                self._last_trigger = now
                self.detections += 1
                return WakeWordDetectionEvent(
                    wake_phrase=self._cfg.wakeword.wake_phrase,
                    model_name=self._cfg.wakeword.model_name,
                    confidence=0.98,
                    threshold=self._cfg.wakeword.effective_threshold(),
                    inference_ms=0.8,
                    audio_rms=round(float(np.sqrt(np.mean(np.square(chunk)))), 5),
                    detected_at_monotonic=now,
                )
            return None

    def info(self) -> WakeWordEngineInfo:
        return WakeWordEngineInfo(
            engine="openwakeword-onnx-stub",
            wake_phrase=self._cfg.wakeword.wake_phrase,
            model_name=self._cfg.wakeword.model_name,
            model_path=str(self._cfg.wakeword.classifier_model),
            loaded=self.loaded,
            models_present=True,
            disk_size_mb=3.51,
            execution_provider="CPUExecutionProvider",
            last_inference_ms=0.8 if self.chunks_seen else None,
            total_detections=self.detections,
        )


def _mock_llm(reply: str = "You are working on LocalVox Phase 6.") -> MagicMock:
    llm = MagicMock(spec=OllamaClient)
    llm.list_models.return_value = [{"name": "qwen3:8b", "size": 5200000000}]
    llm.get_running_memory_mb.return_value = 5000.0
    llm.warmup.return_value = None
    llm.verify_model_exists.return_value = {"name": "qwen3:8b"}

    def fake_chat(
        messages: list[dict[str, str]],
        *,
        on_token: object = None,
        cancel_event: threading.Event | None = None,
        **_kwargs: object,
    ) -> LlmResult:
        del messages
        words = reply.split(" ")
        emitted: list[str] = []
        for idx, word in enumerate(words):
            if cancel_event is not None and cancel_event.is_set():
                break
            piece = word + (" " if idx < len(words) - 1 else "")
            emitted.append(piece)
            if callable(on_token):
                on_token(piece)
            time.sleep(0.005)
        return LlmResult(
            model_name="qwen3:8b",
            response_text="".join(emitted),
            thinking_text="",
            ttft_ms=18.0,
            total_time_ms=65.0,
            load_duration_ms=0.0,
            prompt_tokens=20,
            completion_tokens=len(words),
            tokens_per_sec=50.0,
            cancelled=bool(cancel_event is not None and cancel_event.is_set()),
        )

    llm.chat.side_effect = fake_chat
    return llm


def _mock_stt(transcript: str, language: str = "en") -> MagicMock:
    stt = MagicMock(spec=MlxWhisperStt)
    stt.load.return_value = 2.0
    stt.model_size_mb.return_value = 400.0
    stt.transcribe_array.return_value = SttResult(
        text=transcript,
        language=language,
        audio_duration_s=0.6,
        load_time_ms=0.0,
        processing_time_ms=22.0,
        rtf=0.04,
        model_path="whisper-large-v3-turbo-8bit",
        segments=[],
    )
    return stt


# ---------------------------------------------------------------------------
# Cases 1 & 2: Initialization & Model Availability
# ---------------------------------------------------------------------------


def test_wakeword_engine_initialization_and_model_availability(tmp_path: Path) -> None:
    """Cases 1 & 2: Engine initializes lazily, reports missing artifacts, and loads local ONNX."""
    _, missing_cfg = _phase6_config(tmp_path, use_real_wake_models=False)
    assert check_wakeword_models(missing_cfg.wakeword) is False
    with pytest.raises(WakeWordModelMissingError):
        ensure_wakeword_models_exist(missing_cfg.wakeword)

    missing_engine = OpenWakeWordEngine(missing_cfg)
    assert missing_engine.is_loaded is False
    assert missing_engine.info().models_present is False
    with pytest.raises(WakeWordModelMissingError):
        missing_engine.load()

    _, real_cfg = _phase6_config(tmp_path, use_real_wake_models=True)
    if not check_wakeword_models(real_cfg.wakeword):
        pytest.skip("Local openWakeWord ONNX artifacts not downloaded.")

    engine = OpenWakeWordEngine(real_cfg)
    assert engine.is_loaded is False
    load_ms = engine.load()
    assert load_ms >= 0.0
    assert engine.is_loaded is True
    info = engine.info()
    assert info.models_present is True
    assert info.loaded is True
    assert info.disk_size_mb is not None and info.disk_size_mb > 2.5

    # Invalid sample rate is rejected cleanly.
    with pytest.raises(WakeWordError):
        engine.process_chunk(np.zeros(512, dtype=np.float32), sample_rate=48000)


# ---------------------------------------------------------------------------
# Cases 3, 4, 5, 6, 7: Real Positive, Negative, Threshold, False-Positive, Cooldown
# ---------------------------------------------------------------------------


def test_real_openwakeword_positive_negative_and_duplicate_cooldown(tmp_path: Path) -> None:
    """Cases 3, 4, 7: Real ONNX model detects 'Hey Jarvis', rejects negatives & duplicates."""
    _, cfg = _phase6_config(tmp_path, use_real_wake_models=True)
    real_kokoro_cfg = get_config()
    if not (
        check_wakeword_models(cfg.wakeword)
        and real_kokoro_cfg.tts.kokoro.model_path.exists()
        and real_kokoro_cfg.tts.kokoro.voices_path.exists()
    ):
        pytest.skip("Local openWakeWord or Kokoro ONNX models not present.")

    tts = KokoroTtsEngine(real_kokoro_cfg)
    pos_wav, pos_sr, _ = tts.synthesize_array("Hey Jarvis.", voice="af_heart")
    pos_16k = resample_mono(pos_wav, pos_sr, 16000)
    neg_wav, neg_sr, _ = tts.synthesize_array("What projects am I working on?", voice="af_heart")
    neg_16k = resample_mono(neg_wav, neg_sr, 16000)

    engine = OpenWakeWordEngine(cfg)
    engine.load()

    # Stream positive audio in 512-sample (32 ms) mic frames; must emit exactly 1 event (cooldown).
    padded_pos = np.concatenate(
        [np.zeros(16000, dtype=np.float32), pos_16k, np.zeros(24000, dtype=np.float32)]
    )
    events: list[WakeWordDetectionEvent] = []
    for idx in range(0, padded_pos.size, 512):
        ev = engine.process_chunk(padded_pos[idx : idx + 512], sample_rate=16000)
        if ev is not None:
            events.append(ev)

    assert len(events) == 1
    assert events[0].confidence >= 0.85
    assert events[0].wake_phrase == "Hey Jarvis"
    assert events[0].inference_ms > 0.0

    # Immediate repeat inside cooldown window must NOT emit a duplicate event.
    dup_events: list[WakeWordDetectionEvent] = []
    for idx in range(0, pos_16k.size, 512):
        ev = engine.process_chunk(pos_16k[idx : idx + 512], sample_rate=16000)
        if ev is not None:
            dup_events.append(ev)
    assert dup_events == []

    # Negative utterance + silence + white noise must yield 0 activations.
    engine.reset(clear_cooldown=True)
    rng = np.random.default_rng(7)
    neg_stream = np.concatenate(
        [
            np.zeros(16000, dtype=np.float32),
            neg_16k,
            np.zeros(12000, dtype=np.float32),
            rng.normal(0.0, 0.015, 16000).astype(np.float32),
        ]
    )
    neg_events = [
        ev
        for idx in range(0, neg_stream.size, 512)
        if (ev := engine.process_chunk(neg_stream[idx : idx + 512], sample_rate=16000)) is not None
    ]
    assert neg_events == []


def test_threshold_sensitivity_and_false_activation_guards(tmp_path: Path) -> None:
    """Cases 5 & 6: Sensitivity shifts threshold; single-frame spike & low RMS are suppressed."""
    _, cfg = _phase6_config(tmp_path, use_real_wake_models=False)
    cfg.wakeword.threshold = 0.50
    cfg.wakeword.sensitivity = "low"
    assert cfg.wakeword.effective_threshold() == 0.65
    cfg.wakeword.sensitivity = "medium"
    assert cfg.wakeword.effective_threshold() == 0.50
    cfg.wakeword.sensitivity = "high"
    assert cfg.wakeword.effective_threshold() == 0.35

    # Mock ONNX sessions inside OpenWakeWordEngine to test patience & RMS gating deterministically.
    cfg.wakeword.patience_frames = 2
    cfg.wakeword.min_audio_rms = 0.01
    engine = OpenWakeWordEngine(cfg)
    engine._mel_sess = object()  # type: ignore[assignment]
    engine._emb_sess = object()  # type: ignore[assignment]
    engine._cls_sess = object()  # type: ignore[assignment]

    score_seq = iter([0.92, 0.10, 0.90, 0.91])
    engine._infer_step_locked = lambda _step: (next(score_seq), 0.7)  # type: ignore[method-assign]

    # Single-frame spike (0.92 followed by 0.10) must NOT fire when patience_frames == 2.
    loud = np.full(OWW_STEP_SAMPLES, 0.2, dtype=np.float32)
    assert engine.process_chunk(loud) is None
    assert engine.process_chunk(loud) is None

    # Low-RMS quiet signal must NOT fire even with two consecutive high scores.
    quiet = np.full(OWW_STEP_SAMPLES, 0.0005, dtype=np.float32)
    engine.reset(clear_cooldown=True)
    score_seq_quiet = iter([0.95, 0.96])
    engine._infer_step_locked = lambda _step: (next(score_seq_quiet), 0.7)  # type: ignore[method-assign]
    assert engine.process_chunk(quiet) is None
    assert engine.process_chunk(quiet) is None

    # Two consecutive loud frames above threshold DO trigger cleanly.
    engine.reset(clear_cooldown=True)
    score_seq_ok = iter([0.88, 0.91])
    engine._infer_step_locked = lambda _step: (next(score_seq_ok), 0.7)  # type: ignore[method-assign]
    assert engine.process_chunk(loud) is None
    ev = engine.process_chunk(loud)
    assert ev is not None
    assert ev.confidence == 0.91


# ---------------------------------------------------------------------------
# Cases 8, 9, 10, 11, 12, 14, 17, 20: State Machine, Standby -> Wake -> Turn -> Timeout, Privacy
# ---------------------------------------------------------------------------


def test_state_machine_phase6_transitions_and_guards() -> None:
    """Cases 8, 9, 10, 11: Validate all Phase 6 states & illegal transitions."""
    sm = PipelineStateMachine()
    assert sm.state == PipelineState.IDLE
    assert sm.is_standby() is False

    sm.transition(PipelineState.SLEEPING, "standby")
    assert sm.is_standby() is True
    assert sm.can_accept_speech() is False

    sm.transition(PipelineState.WAKE_DETECTED, "hey_jarvis")
    sm.transition(PipelineState.SPEAKING, "wake_ack")
    sm.transition(PipelineState.LISTENING, "active_session")
    assert sm.can_accept_speech() is True

    sm.transition(PipelineState.PROCESSING, "stt")
    sm.transition(PipelineState.GENERATING, "llm")
    sm.transition(PipelineState.SPEAKING, "tts")
    sm.transition(PipelineState.INTERRUPTED, "barge_in")
    sm.transition(PipelineState.LISTENING, "follow_up")
    sm.transition(PipelineState.RETURNING_TO_STANDBY, "timeout")
    sm.transition(PipelineState.SLEEPING, "back_to_sleep")
    assert sm.is_standby() is True

    sm.transition(PipelineState.STOPPING, "shutdown")
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(PipelineState.LISTENING, "illegal_after_stop")


def test_hands_free_wake_to_conversation_timeout_and_privacy(tmp_path: Path) -> None:
    """Cases 8-12, 14, 20: Standby -> wake -> turn -> timeout -> standby + privacy checks."""
    _, cfg = _phase6_config(tmp_path, use_real_wake_models=False)
    cfg.wakeword.conversation_timeout_s = 2.0
    cfg.wakeword.ack_enabled = True
    cfg.pipeline.barge_in_enabled = False

    capture = LiveQueueCapture()
    player = StreamRecordingPlayer(chunk_delay_s=0.005)
    vad = _fake_vad(cfg)
    stt = _mock_stt("Hey Jarvis, what projects am I working on?", language="en")
    llm = _mock_llm("You are building LocalVox offline.")
    tts_router, _, _ = _mock_tts_engines(cfg)
    ww = StubWakeWordDetector(cfg, trigger_amp=0.65)

    states: list[PipelineState] = []
    wake_events: list[WakeWordDetectionEvent] = []
    sm = PipelineStateMachine(on_change=lambda _p, new, _r: states.append(new))

    pipeline = VoicePipeline(
        config=cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=tts_router,
        state_machine=sm,
        wakeword_detector=ww,
        wake_word_mode=True,
        on_wake_word=wake_events.append,
    )

    # Phase A (Privacy in SLEEPING): feed ambient speech AFTER entering SLEEPING standby.
    ambient_frames = _make_tone_frames(15, amp=0.15)

    def feed_sequence() -> None:
        # Wait until pipeline is in SLEEPING before feeding ambient audio.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and sm.state != PipelineState.SLEEPING:
            time.sleep(0.01)
        capture.push_chunks(ambient_frames)

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and ww.chunks_seen < 12:
            time.sleep(0.02)

        # Verify privacy: ambient speech during SLEEPING NEVER reached Whisper or Ollama!
        assert stt.transcribe_array.call_count == 0
        assert llm.chat.call_count == 0

        # Trigger wake word ("Hey Jarvis!").
        capture.push_chunks(_make_tone_frames(4, amp=0.85))

        # Wait until LISTENING after wake ack, then speak request.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and sm.state != PipelineState.LISTENING:
            time.sleep(0.02)
        capture.push_chunks(_make_tone_frames(8, amp=0.25) + _make_tone_frames(10, amp=0.0))

        # Wait for inactivity timeout -> RETURNING_TO_STANDBY -> SLEEPING, then stop cleanly.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and PipelineState.RETURNING_TO_STANDBY not in states:
            time.sleep(0.05)
        time.sleep(0.1)
        sm.request_stop()

    feeder = threading.Thread(target=feed_sequence, daemon=True)
    feeder.start()
    completed = pipeline.run(warmup=False, wake_word_mode=True)
    feeder.join(timeout=3.0)

    assert len(wake_events) == 1
    assert len(completed) == 1
    # Leading wake phrase prefix was cleanly stripped before reaching Qwen3.
    assert completed[0].transcript == "what projects am I working on?"
    assert PipelineState.SLEEPING in states
    assert PipelineState.WAKE_DETECTED in states
    assert PipelineState.LISTENING in states
    assert PipelineState.SPEAKING in states
    assert PipelineState.RETURNING_TO_STANDBY in states
    assert sm.state == PipelineState.IDLE
    assert capture.stopped is True
    # Zero background audio recordings written to disk.
    assert list(tmp_path.rglob("*.wav")) == []


def test_bare_wake_phrase_ignored_and_preroll_buffered_without_ack(tmp_path: Path) -> None:
    """Cases 9 & 12: Bare 'Hey Jarvis' utterance is ignored; pre-roll preserved when ack off."""
    _, cfg = _phase6_config(tmp_path, use_real_wake_models=False)
    cfg.wakeword.ack_enabled = False
    cfg.wakeword.activation_grace_ms = 0
    cfg.wakeword.pre_roll_ms = 128

    capture = LiveQueueCapture()
    player = StreamRecordingPlayer(chunk_delay_s=0.005)
    vad = _fake_vad(cfg)
    stt = _mock_stt("Hey Jarvis!", language="en")
    llm = _mock_llm("Should not be called.")
    tts_router, _, _ = _mock_tts_engines(cfg)
    ww = StubWakeWordDetector(cfg, trigger_amp=0.65)

    pipeline = VoicePipeline(
        config=cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=tts_router,
        wakeword_detector=ww,
        wake_word_mode=True,
    )
    pipeline._just_woke_up = True
    pipeline.state_machine.transition(PipelineState.LISTENING, "test_listen")
    utterance = np.concatenate(_make_tone_frames(8, amp=0.2))
    res = pipeline.process_utterance(utterance, vad_ms=5.0, sample_rate=16000)
    assert res is None
    assert llm.chat.call_count == 0
    assert pipeline.state_machine.state == PipelineState.LISTENING

    # Verify wait_for_wake_word returns bounded pre-roll frames when wake triggers.
    pipeline.state_machine.transition(PipelineState.IDLE, "reset")
    capture.start()

    def push_after_sleep() -> None:
        deadline = time.monotonic() + 2.0
        while (
            time.monotonic() < deadline
            and pipeline.state_machine.state != PipelineState.SLEEPING
        ):
            time.sleep(0.01)
        capture.push_chunks(_make_tone_frames(6, amp=0.1) + _make_tone_frames(2, amp=0.85))

    pusher = threading.Thread(target=push_after_sleep, daemon=True)
    pusher.start()
    detected = pipeline.wait_for_wake_word(timeout_s=2.0)
    pusher.join(timeout=2.0)
    assert detected is not None
    ev, pre_roll = detected
    assert ev.confidence >= 0.9
    assert 1 <= len(pre_roll) <= 5
    capture.stop()


# ---------------------------------------------------------------------------
# Case 13: Thread Safety
# ---------------------------------------------------------------------------


def test_wakeword_detector_thread_safety(tmp_path: Path) -> None:
    """Case 13: Concurrent process_chunk, info, set_config, and reset calls stay race-free."""
    _, cfg = _phase6_config(tmp_path, use_real_wake_models=True)
    if not check_wakeword_models(cfg.wakeword):
        pytest.skip("Local openWakeWord ONNX artifacts not downloaded.")

    engine = OpenWakeWordEngine(cfg)
    engine.load()
    chunk = np.zeros(512, dtype=np.float32)

    def worker(idx: int) -> None:
        for step in range(15):
            engine.process_chunk(chunk)
            if step % 5 == 0:
                _ = engine.info()
            if idx == 0 and step == 7:
                engine.set_config(cfg)
            if idx == 1 and step == 10:
                engine.reset()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, i) for i in range(4)]
        for fut in futures:
            fut.result()

    assert engine.is_loaded is True


# ---------------------------------------------------------------------------
# Case 16: Barge-In Compatibility Inside Wake-Word Session
# ---------------------------------------------------------------------------


def test_barge_in_compatibility_during_wakeword_session(tmp_path: Path) -> None:
    """Case 16: Speaking over assistant during wake-activated session triggers barge-in cleanly."""
    _, cfg = _phase6_config(tmp_path, use_real_wake_models=False)
    cfg.wakeword.ack_enabled = False
    cfg.wakeword.activation_grace_ms = 0
    cfg.pipeline.barge_in_enabled = True
    cfg.pipeline.barge_in_min_rms = 0.02
    cfg.pipeline.barge_in_min_speech_ms = 64

    capture = LiveQueueCapture()
    interrupted_once = threading.Event()

    def inject_barge_in(_chunk: object) -> None:
        if not interrupted_once.is_set():
            interrupted_once.set()
            capture.push_chunks(_make_tone_frames(10, amp=0.35))

    player = StreamRecordingPlayer(chunk_delay_s=0.04, on_playing_chunk=inject_barge_in)
    vad = _fake_vad(cfg)
    stt_calls = iter(["Explain neural networks slowly.", "Stop and summarize in one line."])
    stt = MagicMock(spec=MlxWhisperStt)
    stt.load.return_value = 1.0
    stt.transcribe_array.side_effect = lambda *_a, **_kw: SttResult(
        text=next(stt_calls, "Done."),
        language="en",
        audio_duration_s=0.5,
        load_time_ms=0.0,
        processing_time_ms=10.0,
        rtf=0.02,
        model_path="whisper",
        segments=[],
    )
    llm = _mock_llm(
        "Neural networks learn layered representations. "
        "Backpropagation updates weights iteratively across training batches."
    )
    tts_router, _, _ = _mock_tts_engines(cfg)
    ww = StubWakeWordDetector(cfg, trigger_amp=0.80)

    sm = PipelineStateMachine()
    pipeline = VoicePipeline(
        config=cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=tts_router,
        state_machine=sm,
        wakeword_detector=ww,
        wake_word_mode=True,
    )

    def driver() -> None:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and sm.state != PipelineState.SLEEPING:
            time.sleep(0.01)
        capture.push_chunks(_make_tone_frames(3, amp=0.90))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and sm.state != PipelineState.LISTENING:
            time.sleep(0.01)
        capture.push_chunks(_make_tone_frames(6, amp=0.25) + _make_tone_frames(8, amp=0.0))
        # After barge-in triggers, supply trailing silence so Turn 2 completes.
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and not interrupted_once.is_set():
            time.sleep(0.01)
        time.sleep(0.15)
        capture.push_chunks(_make_tone_frames(12, amp=0.0))

    t = threading.Thread(target=driver, daemon=True)
    t.start()
    turns = pipeline.run(max_turns=2, warmup=False, wake_word_mode=True)
    t.join(timeout=3.0)

    assert len(turns) == 2
    assert turns[0].barge_in_triggered is True
    assert turns[0].interrupted is True
    # Only ONE wake event occurred; barge-in did NOT spawn a duplicate wake session.
    assert ww.detections == 1


# ---------------------------------------------------------------------------
# Cases 15, 18, 19: API/WebSocket State Sync, Wake-Word Disable & Settings Persistence
# ---------------------------------------------------------------------------


def test_api_wakeword_sync_disable_and_settings_persistence(tmp_path: Path) -> None:
    """Cases 15, 18, 19: Wake events stream over WS, settings persist, disable stops mic."""
    cfg_path, cfg = _phase6_config(tmp_path, use_real_wake_models=False)
    capture = LiveQueueCapture()
    player = StreamRecordingPlayer(chunk_delay_s=0.005)
    vad = _fake_vad(cfg)
    stt = _mock_stt("hello", language="en")
    llm = _mock_llm("Hi there.")
    tts_router, _, _ = _mock_tts_engines(cfg)
    ww = StubWakeWordDetector(cfg, trigger_amp=0.80)

    pipeline = VoicePipeline(
        config=cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=tts_router,
        wakeword_detector=ww,
        wake_word_mode=True,
    )
    runtime = CompanionRuntime(config=cfg, config_path=cfg_path, pipeline=pipeline)
    get_config.cache_clear()
    app = create_app(runtime=runtime)

    with TestClient(app) as client:
        # System Monitor includes openWakeWord engine card.
        sys_data = client.get("/api/system").json()
        assert any("openWakeWord" in e["name"] for e in sys_data["engines"])

        # Settings endpoint returns real wake-word fields & persists cleanly to YAML.
        cur = client.get("/api/settings").json()
        assert cur["settings"]["wakeword_enabled"] is True
        assert cur["settings"]["wake_phrase"] == "Hey Jarvis"
        assert cur["wakeword_engine"]["model_name"] == "hey_jarvis_v0.1"

        updated = {
            **cur["settings"],
            "wakeword_sensitivity": "high",
            "wakeword_conversation_timeout_s": 25.0,
            "wakeword_activation_grace_ms": 200,
        }
        put_res = client.put("/api/settings", json=updated)
        assert put_res.status_code == 200
        assert put_res.json()["settings"]["wakeword_sensitivity"] == "high"
        assert put_res.json()["settings"]["wakeword_conversation_timeout_s"] == 25.0

        saved_yaml = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert saved_yaml["wakeword"]["sensitivity"] == "high"
        assert saved_yaml["wakeword"]["conversation_timeout_s"] == 25.0

        # Start hands-free wakeword standby and confirm WS snapshot & wake_detected broadcast.
        with client.websocket_connect("/ws/events") as ws:
            snap = ws.receive_json()
            assert snap["type"] == "snapshot"
            assert snap["wake_phrase"] == "Hey Jarvis"

            start_res = client.post("/api/voice/start?mode=wakeword")
            assert start_res.status_code == 200
            assert start_res.json()["mic_active"] is True

            # Wait briefly for pipeline thread to enter SLEEPING standby.
            deadline = time.monotonic() + 2.0
            while (
                time.monotonic() < deadline
                and pipeline.state_machine.state != PipelineState.SLEEPING
            ):
                time.sleep(0.02)
            assert pipeline.state_machine.state == PipelineState.SLEEPING

            # Disabling wakeword monitoring immediately stops microphone capture cleanly.
            dis = client.post("/api/wakeword/disable")
            assert dis.status_code == 200
            assert dis.json()["mic_active"] is False
            assert capture.stopped is True

            state_after = client.get("/api/state").json()
            assert state_after["wakeword_enabled"] is False
            assert state_after["mic_active"] is False


def test_personalized_voice_3_sample_enrollment_and_matching(tmp_path: Path) -> None:
    """3-sample personalized voice enrollment extracts local embeddings, saves zero WAVs, wakes."""
    cfg_path, cfg = _phase6_config(tmp_path, use_real_wake_models=True)
    real_kokoro_cfg = get_config()
    if not (
        check_wakeword_models(cfg.wakeword)
        and real_kokoro_cfg.tts.kokoro.model_path.exists()
        and real_kokoro_cfg.tts.kokoro.voices_path.exists()
    ):
        pytest.skip("Local openWakeWord or Kokoro ONNX models not present.")

    tts = KokoroTtsEngine(real_kokoro_cfg)

    def synth_16k(text: str, *, voice: str = "am_michael", speed: float = 1.0) -> np.ndarray:
        wav, sr, _ = tts.synthesize_array(text, voice=voice, speed=speed)
        return resample_mono(wav, sr, 16000)

    samples = [
        synth_16k("Hey Jarvis.", speed=0.93),
        synth_16k("Hey Jarvis!", speed=1.00),
        synth_16k("Hey Jarvis", speed=1.07),
    ]
    sample_iter = iter(samples)

    capture = LiveQueueCapture()
    player = StreamRecordingPlayer(chunk_delay_s=0.005)
    vad = _fake_vad(cfg)
    stt = _mock_stt("hello", language="en")
    llm = _mock_llm("Hi there.")
    tts_router, _, _ = _mock_tts_engines(cfg)
    engine = OpenWakeWordEngine(cfg)
    engine.load()

    pipeline = VoicePipeline(
        config=cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=tts_router,
        wakeword_detector=engine,
        wake_word_mode=True,
    )
    # Deterministic shared-mic utterance feed for the 3 enrollment calls.
    pipeline.capture_enrollment_utterance = lambda *, timeout_s=6.5: next(sample_iter)  # type: ignore[method-assign]

    runtime = CompanionRuntime(config=cfg, config_path=cfg_path, pipeline=pipeline)
    get_config.cache_clear()
    app = create_app(runtime=runtime)

    with TestClient(app) as client:
        prof0 = client.get("/api/wakeword/profile").json()
        assert prof0["enrolled"] is False
        assert prof0["samples_recorded"] == 0

        start_res = client.post("/api/wakeword/enroll/start")
        assert start_res.status_code == 200
        assert start_res.json()["in_progress_samples"] == 0

        r1 = client.post("/api/wakeword/enroll/record").json()
        assert r1["sample_index"] == 1
        assert r1["completed"] is False
        assert r1["profile"]["in_progress_samples"] == 1
        assert r1["profile"]["enrolled"] is False

        r2 = client.post("/api/wakeword/enroll/record").json()
        assert r2["sample_index"] == 2
        assert r2["completed"] is False
        assert r2["consistency_with_previous"] is not None
        assert r2["consistency_with_previous"] >= 0.80

        r3 = client.post("/api/wakeword/enroll/record").json()
        assert r3["sample_index"] == 3
        assert r3["completed"] is True
        assert r3["profile"]["enrolled"] is True
        assert r3["profile"]["samples_recorded"] == 3
        assert r3["profile"]["consistency_score"] >= 0.80

        # Privacy check: mathematical embeddings persisted (.npz), zero raw WAV audio saved.
        assert cfg.wakeword.personal_profile_path.exists()
        assert list(tmp_path.rglob("*.wav")) == []

        # Enable personal-only mode via Settings API and verify persistence.
        cur = client.get("/api/settings").json()["settings"]
        put_res = client.put(
            "/api/settings",
            json={**cur, "wakeword_personal_only": True, "wakeword_personal_enabled": True},
        )
        assert put_res.status_code == 200
        assert put_res.json()["settings"]["wakeword_personal_only"] is True
        assert put_res.json()["voice_profile"]["enrolled"] is True

        # Reload into a fresh engine to verify disk persistence and streaming recognition.
        reloaded_cfg = AppConfig.model_validate(
            yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        )
        fresh_engine = OpenWakeWordEngine(reloaded_cfg)
        fresh_engine.load()
        assert fresh_engine.info().personalized_enrolled is True
        assert fresh_engine.info().personalized_samples == 3
        assert fresh_engine.info().personal_only is True

        pos = np.concatenate(
            [
                np.zeros(8000, dtype=np.float32),
                synth_16k("Hey Jarvis!", speed=0.97),
                np.zeros(16000, dtype=np.float32),
            ]
        )
        pos_events = [
            ev
            for idx in range(0, pos.size, 512)
            if (ev := fresh_engine.process_chunk(pos[idx : idx + 512], 16000)) is not None
        ]
        assert len(pos_events) == 1
        assert pos_events[0].matched_by == "personalized_voice"
        assert pos_events[0].personal_confidence is not None
        assert pos_events[0].personal_confidence >= 0.72

        fresh_engine.reset(clear_cooldown=True)
        neg = np.concatenate(
            [
                np.zeros(8000, dtype=np.float32),
                synth_16k("Hey Siri, what projects am I working on?", speed=1.0),
                np.zeros(16000, dtype=np.float32),
            ]
        )
        neg_events = [
            ev
            for idx in range(0, neg.size, 512)
            if (ev := fresh_engine.process_chunk(neg[idx : idx + 512], 16000)) is not None
        ]
        assert neg_events == []

        # Deleting voice profile removes file cleanly and resets status.
        del_res = client.delete("/api/wakeword/profile")
        assert del_res.status_code == 200
        assert del_res.json()["enrolled"] is False
        assert not cfg.wakeword.personal_profile_path.exists()


def test_auto_start_wakeword_standby_on_app_startup(tmp_path: Path) -> None:
    """Hands-free wake-word standby starts automatically on boot even if Ollama is offline."""
    cfg_path, cfg = _phase6_config(tmp_path, use_real_wake_models=False)
    capture = LiveQueueCapture()
    player = StreamRecordingPlayer(chunk_delay_s=0.005)
    vad = _fake_vad(cfg)
    stt = _mock_stt("hello", language="en")
    llm = _mock_llm("Hi there.")
    llm.warmup.side_effect = OllamaError("Connection refused")
    tts_router, _, _ = _mock_tts_engines(cfg)
    ww = StubWakeWordDetector(cfg, trigger_amp=0.80)

    pipeline = VoicePipeline(
        config=cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=tts_router,
        wakeword_detector=ww,
        wake_word_mode=True,
    )
    runtime = CompanionRuntime(config=cfg, config_path=cfg_path, pipeline=pipeline)
    get_config.cache_clear()
    app = create_app(runtime=runtime, auto_start_wakeword=True)

    with TestClient(app) as client:
        deadline = time.monotonic() + 2.0
        while (
            time.monotonic() < deadline
            and pipeline.state_machine.state != PipelineState.SLEEPING
        ):
            time.sleep(0.02)
        snap = client.get("/api/state").json()
        assert snap["mic_active"] is True
        assert snap["wakeword_active"] is True
        assert snap["pipeline_state"] == PipelineState.SLEEPING.value
