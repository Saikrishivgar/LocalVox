"""Phase 3 tests for barge-in, streaming TTS, echo suppression, and state transitions."""

import json
import queue
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import yaml

from backend.config import AppConfig
from backend.llm.ollama_client import LlmResult, OllamaClient, OllamaInferenceError
from backend.pipeline.audio_io import SpeakerAudioPlayer, SpeechEndpointDetector
from backend.pipeline.echo import EchoSuppressor, PlaybackReferenceBuffer
from backend.pipeline.state import (
    InvalidStateTransitionError,
    PipelineState,
    PipelineStateMachine,
)
from backend.pipeline.voice_pipeline import VoicePipeline
from backend.stt.vad import SileroVadDetector
from backend.stt.whisper_stt import MlxWhisperStt, SttResult
from backend.tts.kokoro_tts import KokoroTtsEngine
from backend.tts.piper_tts import PiperTtsEngine
from backend.tts.router import TtsRouter
from backend.tts.streaming import AudioChunk, SentenceBuffer
from backend.tts.types import TtsResult, UnsupportedLanguageError


def _phase3_config(tmp_path: Path) -> AppConfig:
    """Load real Phase 3 config.yaml isolated inside tmp_path with fast test timings."""
    raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    raw["logging"]["app_log_file"] = str(tmp_path / "app.log")
    raw["logging"]["latency_log_file"] = str(tmp_path / "latency.jsonl")
    raw["paths"]["app_log_file"] = str(tmp_path / "app.log")
    raw["paths"]["latency_log_file"] = str(tmp_path / "latency.jsonl")
    raw["paths"]["memory_db_file"] = str(tmp_path / "memory.db")
    raw["paths"]["conversations_db_file"] = str(tmp_path / "conversations.db")
    raw["memory"]["db_path"] = str(tmp_path / "memory.db")
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
    return AppConfig.model_validate(raw)


def _tone_chunk(freq_hz: float, amp: float, size: int = 512, sr: int = 16000) -> np.ndarray:
    t = np.arange(size, dtype=np.float32) / float(sr)
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _make_tone_frames(
    count: int,
    *,
    freq_hz: float = 440.0,
    amp: float = 0.25,
    chunk_size: int = 512,
) -> list[np.ndarray]:
    if amp == 0.0:
        return [np.zeros(chunk_size, dtype=np.float32) for _ in range(count)]
    t = np.arange(count * chunk_size, dtype=np.float32) / 16000.0
    wave = (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    return [wave[i * chunk_size : (i + 1) * chunk_size].copy() for i in range(count)]


def _fake_vad(cfg: AppConfig) -> SileroVadDetector:
    vad = SileroVadDetector(cfg)
    vad._model = object()
    vad.load = MagicMock(return_value=5.0)
    vad.reset_states = MagicMock()
    vad.predict_chunk = MagicMock(
        side_effect=lambda chunk, sr=16000: 0.9 if float(np.max(np.abs(chunk))) > 0.04 else 0.05
    )
    return vad


def _mock_tts_engines(cfg: AppConfig) -> tuple[TtsRouter, MagicMock, MagicMock]:
    kokoro = MagicMock(spec=KokoroTtsEngine)

    def synth_en(text: str, **_kwargs: object) -> tuple[np.ndarray, int, TtsResult]:
        wave = np.full(1200, 0.05, dtype=np.float32)
        return (
            wave,
            24000,
            TtsResult(
                backend="kokoro",
                model_name="kokoro",
                voice="af_heart",
                language="en",
                sample_rate=24000,
                audio_duration_s=0.05,
                load_time_ms=0.0,
                processing_time_ms=12.0,
                rtf=0.24,
            ),
        )

    kokoro.synthesize_array.side_effect = synth_en
    piper = MagicMock(spec=PiperTtsEngine)

    def synth_ta(text: str, **_kwargs: object) -> tuple[np.ndarray, int, TtsResult]:
        wave = np.full(1102, 0.05, dtype=np.float32)
        return (
            wave,
            22050,
            TtsResult(
                backend="piper",
                model_name="ta_IN-rasa_female-medium",
                voice="ta_IN-rasa_female-medium",
                language="ta",
                sample_rate=22050,
                audio_duration_s=0.05,
                load_time_ms=0.0,
                processing_time_ms=8.0,
                rtf=0.16,
            ),
        )

    piper.synthesize_array.side_effect = synth_ta
    return TtsRouter(cfg, kokoro=kokoro, piper=piper), kokoro, piper


class LiveQueueCapture:
    """Thread-safe asynchronous AudioCaptureSource for realistic Phase 3 concurrency tests."""

    def __init__(self) -> None:
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._accepting = threading.Event()
        self.started = False
        self.stopped = False

    @property
    def is_paused(self) -> bool:
        return not self._accepting.is_set()

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        self._accepting.clear()

    def pause(self) -> None:
        self._accepting.clear()
        self.clear_buffer()

    def resume(self) -> None:
        self._accepting.set()

    def clear_buffer(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def push_chunks(self, chunks: list[np.ndarray]) -> None:
        for ch in chunks:
            self._queue.put(ch)

    def read_chunk(self, timeout_s: float = 0.2) -> np.ndarray | None:
        if not self._accepting.is_set():
            return None
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None


class StreamRecordingPlayer:
    """In-memory AudioPlaybackSink supporting play_stream, reference buffer, and instant cancel."""

    def __init__(
        self,
        *,
        chunk_delay_s: float = 0.03,
        on_playing_chunk: object = None,
    ) -> None:
        self.reference_buffer = PlaybackReferenceBuffer(target_sr=16000)
        self.played_chunks: list[AudioChunk] = []
        self.cancelled = False
        self.reset_calls = 0
        self.last_cancel_latency_ms: float | None = None
        self._cancel_event = threading.Event()
        self._cancel_requested_at: float | None = None
        self._chunk_delay_s = chunk_delay_s
        self._on_playing_chunk = on_playing_chunk

    def cancel(self) -> None:
        self.cancelled = True
        if not self._cancel_event.is_set():
            self._cancel_requested_at = time.perf_counter()
            self._cancel_event.set()

    def reset_cancel(self) -> None:
        self.cancelled = False
        self.reset_calls += 1
        self._cancel_event.clear()
        self._cancel_requested_at = None
        self.last_cancel_latency_ms = None

    def play(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        stop_event: threading.Event | None = None,
    ) -> tuple[float, bool]:
        self.reference_buffer.push(waveform, sample_rate)
        interrupted = self._cancel_event.is_set() or (
            stop_event is not None and stop_event.is_set()
        )
        return 5.0, interrupted

    def play_stream(
        self,
        chunk_queue: "queue.Queue[AudioChunk | None]",
        *,
        stop_event: threading.Event | None = None,
        on_chunk_start: object = None,
    ) -> tuple[float, bool]:
        start = time.perf_counter()
        interrupted = False
        while True:
            if self._cancel_event.is_set() or (stop_event is not None and stop_event.is_set()):
                interrupted = True
                if self._cancel_requested_at is not None:
                    self.last_cancel_latency_ms = round(
                        (time.perf_counter() - self._cancel_requested_at) * 1000.0,
                        2,
                    )
                break
            try:
                item = chunk_queue.get(timeout=0.02)
            except queue.Empty:
                continue
            if item is None:
                break
            if on_chunk_start is not None:
                on_chunk_start(item)
            self.played_chunks.append(item)
            self.reference_buffer.push(item.waveform, item.sample_rate)
            if self._on_playing_chunk is not None:
                self._on_playing_chunk(item)
            steps = max(1, int(self._chunk_delay_s / 0.005))
            for _ in range(steps):
                if self._cancel_event.is_set() or (
                    stop_event is not None and stop_event.is_set()
                ):
                    interrupted = True
                    if self._cancel_requested_at is not None:
                        self.last_cancel_latency_ms = round(
                            (time.perf_counter() - self._cancel_requested_at) * 1000.0,
                            2,
                        )
                    break
                time.sleep(0.005)
            if interrupted:
                break
        if interrupted:
            self.reference_buffer.clear()
        return round((time.perf_counter() - start) * 1000.0, 2), interrupted


def _assert_no_worker_threads_alive() -> None:
    lingering = [
        t.name
        for t in threading.enumerate()
        if t.name.startswith("localvox-") and t.is_alive()
    ]
    assert lingering == []


def test_streaming_sentence_buffer_boundaries_and_ordering() -> None:
    """Case 10: SentenceBuffer preserves decimals/abbreviations, orders chunks, and flushes tail."""
    buf = SentenceBuffer(min_sentence_chars=16, max_chunk_chars=60)
    tokens = [
        "Dr. Smith measured ",
        "pi at 3.14 today. ",
        "No! ",
        "Force equals mass times acceleration. ",
        "And here is a trailing clause without punctuation",
    ]
    emitted: list[str] = []
    for tok in tokens:
        emitted.extend(buf.push(tok))
    emitted.extend(buf.flush())

    assert emitted == [
        "Dr. Smith measured pi at 3.14 today.",
        "No! Force equals mass times acceleration.",
        "And here is a trailing clause without punctuation",
    ]

    # Soft split on long punctuation-free runs and skip punctuation-only noise.
    long_buf = SentenceBuffer(min_sentence_chars=10, max_chunk_chars=35)
    long_chunks = long_buf.push("Alpha beta gamma, delta epsilon zeta eta")
    long_chunks.extend(long_buf.flush())
    assert len(long_chunks) == 2
    assert all(len(c) <= 35 for c in long_chunks)

    noise_buf = SentenceBuffer(min_sentence_chars=5, max_chunk_chars=40)
    assert noise_buf.push("...   \n   !!!") == []
    assert noise_buf.flush() == []


def test_state_machine_transitions_and_generation_invalidation() -> None:
    """Case 8: State machine validates edges, tracks INTERRUPTED/STOPPING, invalidates old turns."""
    seen: list[tuple[PipelineState, PipelineState, str]] = []
    sm = PipelineStateMachine(on_change=lambda p, n, r: seen.append((p, n, r)))

    assert sm.state == PipelineState.IDLE
    gen1, cancel1 = sm.begin_generation()
    assert sm.is_generation_current(gen1) is True

    sm.transition(PipelineState.LISTENING, "start")
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(PipelineState.SPEAKING, "cannot_skip_processing")

    sm.transition(PipelineState.PROCESSING, "stt")
    sm.transition(PipelineState.GENERATING, "llm")
    sm.transition(PipelineState.SPEAKING, "tts")
    sm.request_barge_in()
    assert cancel1.is_set() is True
    assert sm.is_generation_current(gen1) is False

    sm.transition(PipelineState.INTERRUPTED, "user_barge_in")
    gen2, cancel2 = sm.begin_generation()
    assert gen2 > gen1
    assert cancel2.is_set() is False
    assert sm.is_generation_current(gen2) is True

    sm.request_stop()
    assert sm.state == PipelineState.STOPPING
    assert cancel2.is_set() is True
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(PipelineState.LISTENING, "cannot_reenter_from_stopping")
    sm.transition(PipelineState.IDLE, "done")
    assert any(curr == PipelineState.INTERRUPTED for _, curr, _ in seen)
    assert any(curr == PipelineState.STOPPING for _, curr, _ in seen)


def test_audio_playback_queue_and_stream_player_mocked(tmp_path: Path) -> None:
    """Case 9: SpeakerAudioPlayer.play_stream plays chunks in order and aborts on cancel."""
    cfg = _phase3_config(tmp_path)
    player = SpeakerAudioPlayer(cfg)
    written_arrays: list[np.ndarray] = []

    class FakeOutputStream:
        def __init__(self, **_kwargs: object) -> None:
            self.aborted = False
            self.closed = False

        def start(self) -> None:
            pass

        def write(self, block: np.ndarray) -> None:
            written_arrays.append(block.copy())

        def abort(self) -> None:
            self.aborted = True

        def stop(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    fake_stream = FakeOutputStream()
    q: queue.Queue[AudioChunk | None] = queue.Queue()
    w1 = np.full(1200, 0.1, dtype=np.float32)
    w2 = np.full(1200, 0.2, dtype=np.float32)
    q.put(AudioChunk(seq_id=0, text="First.", waveform=w1, sample_rate=24000))
    q.put(AudioChunk(seq_id=1, text="Second.", waveform=w2, sample_rate=24000))
    q.put(None)

    started_order: list[int] = []
    with (
        patch("backend.pipeline.audio_io.resolve_audio_device", return_value=0),
        patch(
            "backend.pipeline.audio_io.sd.query_devices",
            return_value={"default_samplerate": 48000},
        ),
        patch("backend.pipeline.audio_io.sd.check_output_settings"),
        patch("backend.pipeline.audio_io.sd.OutputStream", return_value=fake_stream),
    ):
        elapsed_ms, interrupted = player.play_stream(
            q,
            on_chunk_start=lambda c: started_order.append(c.seq_id),
        )
    assert interrupted is False
    assert elapsed_ms >= 0.0
    assert started_order == [0, 1]
    assert len(written_arrays) > 0

    # Verify instant abort inside play_stream when cancel() fires after chunk 0 starts.
    q2: queue.Queue[AudioChunk | None] = queue.Queue()
    wl = np.full(4800, 0.2, dtype=np.float32)
    q2.put(AudioChunk(seq_id=0, text="Long chunk.", waveform=wl, sample_rate=24000))
    q2.put(AudioChunk(seq_id=1, text="Skipped chunk.", waveform=wl, sample_rate=24000))
    q2.put(None)

    cancel_stream = FakeOutputStream()
    seen_seqs: list[int] = []
    player.reset_cancel()

    def on_start_and_cancel(chunk: AudioChunk) -> None:
        seen_seqs.append(chunk.seq_id)
        player.cancel()

    with (
        patch("backend.pipeline.audio_io.resolve_audio_device", return_value=0),
        patch(
            "backend.pipeline.audio_io.sd.query_devices",
            return_value={"default_samplerate": 48000},
        ),
        patch("backend.pipeline.audio_io.sd.check_output_settings"),
        patch("backend.pipeline.audio_io.sd.OutputStream", return_value=cancel_stream),
    ):
        _, interrupted2 = player.play_stream(q2, on_chunk_start=on_start_and_cancel)

    assert interrupted2 is True
    assert cancel_stream.aborted is True
    assert seen_seqs == [0]
    assert player.last_cancel_latency_ms is not None


def test_echo_suppressor_and_sensitivity_profiles(tmp_path: Path) -> None:
    """Case 3 DSP: Reject correlated playback echo while keeping double-talk user speech."""
    cfg = _phase3_config(tmp_path)
    low_cfg = cfg.model_copy(deep=True)
    low_cfg.pipeline.barge_in_sensitivity = "low"
    high_cfg = cfg.model_copy(deep=True)
    high_cfg.pipeline.barge_in_sensitivity = "high"

    p_low = low_cfg.pipeline.effective_barge_in_params(32.0)
    p_med = cfg.pipeline.effective_barge_in_params(32.0)
    p_high = high_cfg.pipeline.effective_barge_in_params(32.0)
    assert p_high[0] < p_med[0] < p_low[0]  # VAD threshold increases from high -> medium -> low
    assert p_high[1] <= p_med[1] <= p_low[1]  # required frames increases
    assert p_high[2] < p_med[2] < p_low[2]  # RMS gate increases

    ref_buf = PlaybackReferenceBuffer(target_sr=16000)
    suppressor = EchoSuppressor(ref_buf, cfg)

    # Push uncorrelated wideband audio into ref_buf and derive delayed mic echo at 16 kHz.
    rng = np.random.default_rng(7)
    ref_16k = (0.20 * rng.standard_normal(512 * 6)).astype(np.float32)
    ref_buf.push(ref_16k, 16000)

    # Pure delayed speaker echo (offset 700 samples): suppressed and classified as echo.
    delayed_echo = 0.55 * ref_16k[-(700 + 512) : -700].copy()
    echo_res = suppressor.process_frame(delayed_echo)
    assert echo_res.correlation > 0.85
    assert echo_res.residual_rms < 0.01
    assert echo_res.is_likely_pure_echo is True

    # Double-talk: delayed speaker echo plus louder uncorrelated user interruption tone.
    user_voice = _tone_chunk(290.0, 0.25, size=512)
    double_talk = delayed_echo + user_voice
    dt_res = suppressor.process_frame(double_talk)
    assert dt_res.residual_rms >= 0.10
    assert dt_res.is_likely_pure_echo is False


def test_normal_and_multiturn_streaming_tts_without_interruption(tmp_path: Path) -> None:
    """Cases 1, 2, 11: Streaming TTS overlaps with LLM, keeps ordering & completes normally."""
    cfg = _phase3_config(tmp_path)
    vad = _fake_vad(cfg)
    capture = LiveQueueCapture()
    player = StreamRecordingPlayer(chunk_delay_s=0.01)
    tts_router, kokoro, piper = _mock_tts_engines(cfg)

    stt = MagicMock(spec=MlxWhisperStt)
    stt.load.return_value = 5.0
    stt.transcribe_array.side_effect = [
        SttResult(
            text="Explain osmosis.",
            language="en",
            audio_duration_s=0.6,
            load_time_ms=0.0,
            processing_time_ms=30.0,
            rtf=0.05,
            model_path="mock",
            segments=[],
        ),
        SttResult(
            text="Give an example in Tamil.",
            language="ta",
            audio_duration_s=0.6,
            load_time_ms=0.0,
            processing_time_ms=25.0,
            rtf=0.04,
            model_path="mock",
            segments=[],
        ),
    ]

    def streaming_llm_side_effect(
        messages: list[dict[str, str]],
        *,
        on_token: object = None,
        cancel_event: threading.Event | None = None,
    ) -> LlmResult:
        user_msg = messages[-1]["content"]
        if "Tamil" in user_msg:
            chunks = ["சரி, இப்போது பார்க்கலாம். ", "தாவரங்கள் நீர் உறிஞ்சுவது சவ்வூடுபரவல்."]
        else:
            chunks = ["Osmosis moves water across membranes. ", "Cells depend on this every day."]
        for ch in chunks:
            if on_token is not None:
                on_token(ch)
            time.sleep(0.01)
        return LlmResult(
            model_name="qwen3:8b",
            response_text="".join(chunks),
            thinking_text="",
            ttft_ms=15.0,
            total_time_ms=40.0,
            load_duration_ms=None,
            prompt_tokens=20,
            completion_tokens=16,
            tokens_per_sec=30.0,
        )

    llm = MagicMock(spec=OllamaClient)
    llm.warmup.return_value = None
    llm.chat.side_effect = streaming_llm_side_effect

    states_seen: list[PipelineState] = []
    listen_cond = threading.Condition()
    listening_entries = 0

    def on_state(_p: PipelineState, curr: PipelineState, _r: str) -> None:
        nonlocal listening_entries
        states_seen.append(curr)
        if curr == PipelineState.LISTENING:
            with listen_cond:
                listening_entries += 1
                listen_cond.notify_all()

    sm = PipelineStateMachine(on_change=on_state)
    pipeline = VoicePipeline(
        cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=tts_router,
        state_machine=sm,
    )

    turn1_audio = (
        _make_tone_frames(1, amp=0.0)
        + _make_tone_frames(4, amp=0.25)
        + _make_tone_frames(5, amp=0.0)
    )
    turn2_audio = (
        _make_tone_frames(1, amp=0.0)
        + _make_tone_frames(4, amp=0.25)
        + _make_tone_frames(5, amp=0.0)
    )

    def feed_two_turns() -> None:
        with listen_cond:
            assert listen_cond.wait_for(lambda: listening_entries >= 1, timeout=2.0)
        capture.push_chunks(turn1_audio)
        with listen_cond:
            assert listen_cond.wait_for(lambda: listening_entries >= 2, timeout=2.0)
        capture.push_chunks(turn2_audio)

    feeder = threading.Thread(target=feed_two_turns, daemon=True)
    feeder.start()
    metrics = pipeline.run(max_turns=2, warmup=True)
    feeder.join(timeout=2.0)

    assert len(metrics) == 2
    assert metrics[0].interrupted is False
    assert metrics[0].barge_in_triggered is False
    assert metrics[0].tts_chunks == 2
    assert metrics[0].tts_ttfa_ms == 12.0
    assert metrics[0].stt_ms == 30.0
    assert metrics[0].first_audio_latency_ms > 0.0
    assert metrics[0].speech_end_to_stt_ms >= 0.0
    assert metrics[1].tts_backend == "piper"
    assert metrics[1].tts_language == "ta"
    assert [c.seq_id for c in player.played_chunks[:2]] == [0, 1]
    assert [c.seq_id for c in player.played_chunks[2:4]] == [0, 1]
    assert pipeline.conversation.turn_count == 2
    assert sm.state == PipelineState.IDLE
    _assert_no_worker_threads_alive()


def test_barge_in_during_playback_interrupts_and_preserves_followup_speech(tmp_path: Path) -> None:
    """Case 3 & 7: User speech during playback cancels audio/LLM immediately and seeds next turn."""
    cfg = _phase3_config(tmp_path)
    vad = _fake_vad(cfg)
    capture = LiveQueueCapture()
    tts_router, _, _ = _mock_tts_engines(cfg)

    stt = MagicMock(spec=MlxWhisperStt)
    stt.load.return_value = 5.0
    stt.transcribe_array.side_effect = [
        SttResult(
            text="Tell me a very long story about physics.",
            language="en",
            audio_duration_s=0.8,
            load_time_ms=0.0,
            processing_time_ms=25.0,
            rtf=0.03,
            model_path="mock",
            segments=[],
        ),
        SttResult(
            text="Wait, summarize in one sentence.",
            language="en",
            audio_duration_s=0.9,
            load_time_ms=0.0,
            processing_time_ms=22.0,
            rtf=0.025,
            model_path="mock",
            segments=[],
        ),
    ]

    llm_cancelled_in_turn1 = threading.Event()
    turn_idx = 0

    def llm_side_effect(
        messages: list[dict[str, str]],
        *,
        on_token: object = None,
        cancel_event: threading.Event | None = None,
    ) -> LlmResult:
        nonlocal turn_idx
        turn_idx += 1
        if turn_idx == 1:
            sentences = [
                "First sentence about Newtonian mechanics starts here. ",
                "Second long sentence continues for quite a while. ",
                "Third sentence should never be spoken after interruption. ",
            ]
            spoken_buf: list[str] = []
            for sent in sentences:
                if cancel_event is not None and cancel_event.is_set():
                    llm_cancelled_in_turn1.set()
                    return LlmResult(
                        model_name="qwen3:8b",
                        response_text="".join(spoken_buf),
                        thinking_text="",
                        ttft_ms=10.0,
                        total_time_ms=30.0,
                        load_duration_ms=None,
                        prompt_tokens=20,
                        completion_tokens=10,
                        tokens_per_sec=20.0,
                        cancelled=True,
                    )
                if on_token is not None:
                    on_token(sent)
                spoken_buf.append(sent)
                time.sleep(0.06)
            return LlmResult(
                model_name="qwen3:8b",
                response_text="".join(spoken_buf),
                thinking_text="",
                ttft_ms=10.0,
                total_time_ms=180.0,
                load_duration_ms=None,
                prompt_tokens=20,
                completion_tokens=30,
                tokens_per_sec=20.0,
            )

        assert any("[interrupted]" in m["content"] for m in messages if m["role"] == "assistant")
        if on_token is not None:
            on_token("Physics studies matter, motion, and energy.")
        return LlmResult(
            model_name="qwen3:8b",
            response_text="Physics studies matter, motion, and energy.",
            thinking_text="",
            ttft_ms=12.0,
            total_time_ms=35.0,
            load_duration_ms=None,
            prompt_tokens=30,
            completion_tokens=9,
            tokens_per_sec=25.0,
        )

    llm = MagicMock(spec=OllamaClient)
    llm.warmup.return_value = None
    llm.chat.side_effect = llm_side_effect

    interrupt_sent = False

    def on_playing_chunk(chunk: AudioChunk) -> None:
        nonlocal interrupt_sent
        if chunk.seq_id == 0 and not interrupt_sent:
            interrupt_sent = True
            # Push 5 barge-in frames (160ms > 96ms threshold), continuation, and silence.
            capture.push_chunks(
                _make_tone_frames(5, freq_hz=310.0, amp=0.35)
                + _make_tone_frames(2, freq_hz=310.0, amp=0.35)
                + _make_tone_frames(6, amp=0.0)
            )

    player = StreamRecordingPlayer(chunk_delay_s=0.12, on_playing_chunk=on_playing_chunk)
    states_seen: list[PipelineState] = []
    turn1_ready = threading.Event()

    def on_state(_p: PipelineState, c: PipelineState, _r: str) -> None:
        states_seen.append(c)
        if c == PipelineState.LISTENING:
            turn1_ready.set()

    sm = PipelineStateMachine(on_change=on_state)

    pipeline = VoicePipeline(
        cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=tts_router,
        state_machine=sm,
    )

    def feed_turn1() -> None:
        assert turn1_ready.wait(timeout=2.0)
        capture.push_chunks(
            _make_tone_frames(1, amp=0.0)
            + _make_tone_frames(4, amp=0.25)
            + _make_tone_frames(5, amp=0.0)
        )

    feeder = threading.Thread(target=feed_turn1, daemon=True)
    feeder.start()
    metrics = pipeline.run(max_turns=2, warmup=True)
    feeder.join(timeout=2.0)

    assert len(metrics) == 2
    assert metrics[0].interrupted is True
    assert metrics[0].barge_in_triggered is True
    assert metrics[0].interruption_detect_ms is not None
    assert metrics[0].playback_cancel_ms is not None
    assert player.reset_calls == 2  # Reset cleanly before Turn 2 playback
    assert PipelineState.INTERRUPTED in states_seen
    assert llm_cancelled_in_turn1.is_set() is True
    assert metrics[1].interrupted is False
    assert metrics[1].transcript == "Wait, summarize in one sentence."
    # Verify Turn 2 STT received seeded prefix + continuation (> 5 speech chunks).
    turn2_waveform = stt.transcribe_array.call_args_list[1].args[0]
    assert turn2_waveform.size >= 7 * 512
    _assert_no_worker_threads_alive()


def test_cancellation_during_tts_and_llm_generation(tmp_path: Path) -> None:
    """Cases 4 & 5: Cancel cleanly while TTS is synthesizing and while LLM is streaming pre-TTS."""
    cfg = _phase3_config(tmp_path)
    vad = _fake_vad(cfg)
    capture = LiveQueueCapture()
    capture.resume()
    player = StreamRecordingPlayer(chunk_delay_s=0.02)

    # --- Case 4: Cancellation during TTS synthesis ---
    stt = MagicMock(spec=MlxWhisperStt)
    stt.transcribe_array.return_value = SttResult(
        text="Explain thermodynamics.",
        language="en",
        audio_duration_s=0.5,
        load_time_ms=0.0,
        processing_time_ms=20.0,
        rtf=0.04,
        model_path="mock",
        segments=[],
    )

    llm = MagicMock(spec=OllamaClient)
    llm.chat.return_value = LlmResult(
        model_name="qwen3:8b",
        response_text="Heat flows from hot objects to cold objects naturally.",
        thinking_text="",
        ttft_ms=10.0,
        total_time_ms=20.0,
        load_duration_ms=None,
        prompt_tokens=10,
        completion_tokens=10,
        tokens_per_sec=25.0,
    )

    in_tts = threading.Event()
    release_tts = threading.Event()
    kokoro = MagicMock(spec=KokoroTtsEngine)

    def slow_synth(_text: str, **_kw: object) -> tuple[np.ndarray, int, TtsResult]:
        in_tts.set()
        release_tts.wait(timeout=1.0)
        return (
            np.full(1200, 0.1, dtype=np.float32),
            24000,
            TtsResult(
                backend="kokoro",
                model_name="kokoro",
                voice="af_heart",
                language="en",
                sample_rate=24000,
                audio_duration_s=0.05,
                load_time_ms=0.0,
                processing_time_ms=25.0,
                rtf=0.5,
            ),
        )

    kokoro.synthesize_array.side_effect = slow_synth
    pipeline = VoicePipeline(
        cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=TtsRouter(cfg, kokoro=kokoro),
    )

    def cancel_mid_tts() -> None:
        in_tts.wait(timeout=1.0)
        pipeline.cancel_playback()
        release_tts.set()

    t_cancel = threading.Thread(target=cancel_mid_tts, daemon=True)
    t_cancel.start()
    m_tts = pipeline.process_utterance(np.full(4000, 0.25, dtype=np.float32))
    t_cancel.join(timeout=1.0)

    assert m_tts is not None
    assert m_tts.interrupted is True
    assert player.played_chunks == []  # Cancelled synthesized chunk was discarded, never played!
    _assert_no_worker_threads_alive()

    # --- Case 5: Real OllamaClient HTTP stream cancellation during LLM generation ---
    cfg2 = _phase3_config(tmp_path)
    cfg2.pipeline.barge_in_enabled = True
    seen_lines = 0
    cancel_evt = threading.Event()

    class HangingStream(httpx.SyncByteStream):
        def __iter__(self):
            nonlocal seen_lines
            line1 = json.dumps({"message": {"role": "assistant", "content": "Part one "}})
            line2 = json.dumps({"message": {"role": "assistant", "content": "Part two"}})
            yield line1.encode() + b"\n"
            seen_lines += 1
            cancel_evt.set()
            yield line2.encode() + b"\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "size": 5_000_000}]})
        if request.url.path == "/api/chat":
            return httpx.Response(200, stream=HangingStream())
        return httpx.Response(404)

    real_llm = OllamaClient(
        cfg2,
        http_client=httpx.Client(
            base_url=cfg2.llm.base_url,
            transport=httpx.MockTransport(handler),
        ),
    )
    tokens_out: list[str] = []
    res = real_llm.chat(
        [{"role": "user", "content": "Hi"}],
        on_token=tokens_out.append,
        cancel_event=cancel_evt,
    )
    assert res.cancelled is True
    assert tokens_out == ["Part one"] or tokens_out == ["Part one "]
    assert seen_lines == 1
    _assert_no_worker_threads_alive()


def test_empty_transcription_and_silence_timeout_and_pause_tolerance(tmp_path: Path) -> None:
    """Case 6 & Turn-taking: Empty transcripts, mid-sentence pauses, and listen_timeout_s work."""
    cfg = _phase3_config(tmp_path)
    cfg.pipeline.vad_hangover_ms = 128  # 4 frames hangover
    vad = _fake_vad(cfg)
    detector = SpeechEndpointDetector(vad, cfg)

    # Empty capture times out gracefully without error.
    capture = LiveQueueCapture()
    capture.resume()
    assert detector.capture_utterance(capture, timeout_s=0.05) is None

    # Mid-sentence pause of 2 silent frames (< 4 hangover) must NOT split utterance early!
    phrase_frames = (
        _make_tone_frames(1, amp=0.0)
        + _make_tone_frames(3, amp=0.25)
        + _make_tone_frames(2, amp=0.0)
        + _make_tone_frames(3, amp=0.25)
        + _make_tone_frames(5, amp=0.0)
    )
    capture.push_chunks(phrase_frames)
    captured = detector.capture_utterance(capture, timeout_s=1.0)
    assert captured is not None
    waveform, _ = captured
    assert waveform.size >= 11 * 512

    # Empty transcription returns cleanly to LISTENING without invoking LLM or altering history.
    stt = MagicMock(spec=MlxWhisperStt)
    stt.transcribe_array.return_value = SttResult(
        text="   ",
        language="en",
        audio_duration_s=0.4,
        load_time_ms=0.0,
        processing_time_ms=10.0,
        rtf=0.025,
        model_path="mock",
        segments=[],
    )
    llm = MagicMock(spec=OllamaClient)
    tts_router, _, _ = _mock_tts_engines(cfg)
    sm = PipelineStateMachine()
    pipeline = VoicePipeline(
        cfg,
        capture=capture,
        player=StreamRecordingPlayer(),
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=tts_router,
        state_machine=sm,
    )
    assert pipeline.process_utterance(waveform) is None
    assert sm.state == PipelineState.LISTENING
    assert llm.chat.call_count == 0
    assert pipeline.conversation.turn_count == 0
    _assert_no_worker_threads_alive()


def test_streaming_worker_exception_recovery_and_no_zombies(tmp_path: Path) -> None:
    """Case 12 & 7: Exceptions in LLM worker, TTS worker, and playback sink recover to LISTENING."""
    cfg = _phase3_config(tmp_path)
    vad = _fake_vad(cfg)
    capture = LiveQueueCapture()
    capture.resume()
    player = StreamRecordingPlayer()
    tts_router, kokoro, _ = _mock_tts_engines(cfg)
    stt = MagicMock(spec=MlxWhisperStt)
    stt.transcribe_array.return_value = SttResult(
        text="Test recovery",
        language="en",
        audio_duration_s=0.4,
        load_time_ms=0.0,
        processing_time_ms=15.0,
        rtf=0.04,
        model_path="mock",
        segments=[],
    )
    llm = MagicMock(spec=OllamaClient)
    errors: list[Exception] = []
    sm = PipelineStateMachine()

    pipeline = VoicePipeline(
        cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=tts_router,
        state_machine=sm,
        on_error=errors.append,
    )
    utt = np.full(4000, 0.25, dtype=np.float32)

    # 1. LLM streaming error inside worker thread
    llm.chat.side_effect = OllamaInferenceError("Local Ollama crashed")
    assert pipeline.process_utterance(utt) is None
    assert sm.state == PipelineState.LISTENING
    _assert_no_worker_threads_alive()

    # 2. TTS worker error mid-stream
    llm.chat.side_effect = None
    llm.chat.return_value = LlmResult(
        model_name="qwen3:8b",
        response_text="First valid sentence here.",
        thinking_text="",
        ttft_ms=10.0,
        total_time_ms=20.0,
        load_duration_ms=None,
        prompt_tokens=5,
        completion_tokens=5,
        tokens_per_sec=20.0,
    )
    kokoro.synthesize_array.side_effect = UnsupportedLanguageError("Broken voice")
    assert pipeline.process_utterance(utt) is None
    assert sm.state == PipelineState.LISTENING
    assert len(errors) == 2
    _assert_no_worker_threads_alive()

    # 3. Clean recovery turn immediately succeeds afterward
    _, kokoro_ok, piper_ok = _mock_tts_engines(cfg)
    pipeline_ok = VoicePipeline(
        cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=TtsRouter(cfg, kokoro=kokoro_ok, piper=piper_ok),
        state_machine=sm,
        conversation=pipeline.conversation,
    )
    ok_metrics = pipeline_ok.process_utterance(utt)
    assert ok_metrics is not None
    assert ok_metrics.interrupted is False
    assert sm.state == PipelineState.LISTENING
    _assert_no_worker_threads_alive()

