"""Hardware-independent integration tests for the Phase 2 & 2.1 voice pipeline."""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import yaml

from backend.config import AppConfig
from backend.llm.ollama_client import OllamaClient, OllamaUnavailableError
from backend.pipeline.audio_io import (
    AudioDeviceError,
    AudioDeviceInfo,
    MicrophoneAudioCapture,
    SpeakerAudioPlayer,
    SpeechEndpointDetector,
    resolve_audio_device,
)
from backend.pipeline.conversation import ConversationManager
from backend.pipeline.state import PipelineState, PipelineStateMachine
from backend.pipeline.voice_pipeline import VoicePipeline
from backend.stt.vad import SileroVadDetector
from backend.stt.whisper_stt import MlxWhisperStt, SttResult
from backend.tts.kokoro_tts import KokoroTtsEngine
from backend.tts.piper_tts import PiperTtsEngine
from backend.tts.router import TtsRouter, clean_text_for_speech, detect_speech_language
from backend.tts.types import TtsResult, UnsupportedLanguageError


def _isolated_config(tmp_path: Path) -> AppConfig:
    """Build a validated AppConfig isolated inside tmp_path."""
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
    raw["pipeline"]["post_speech_pad_ms"] = 160
    raw["pipeline"]["vad_hangover_ms"] = 128
    raw["pipeline"]["min_utterance_ms"] = 96
    raw["pipeline"]["barge_in_enabled"] = False
    raw["vad"]["min_speech_duration_ms"] = 64
    return AppConfig.model_validate(raw)


class FakeCapture:
    """Deterministic gateable AudioCaptureSource supporting scripted multi-turn feeds."""

    def __init__(
        self,
        chunks: list[np.ndarray] | None = None,
        *,
        turn_scripts: list[list[np.ndarray]] | None = None,
    ) -> None:
        self._buffer = list(chunks or [])
        self._turn_scripts = list(turn_scripts or [])
        self.started = False
        self.stopped = False
        self._paused = False
        self.pause_calls = 0
        self.resume_calls = 0
        self.clear_calls = 0
        self.dropped_while_paused = 0

    @property
    def is_paused(self) -> bool:
        return self._paused

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        self._paused = True

    def pause(self) -> None:
        self._paused = True
        self.pause_calls += 1

    def resume(self) -> None:
        self._paused = False
        self.resume_calls += 1
        if not self._buffer and self._turn_scripts:
            self._buffer.extend(self._turn_scripts.pop(0))

    def clear_buffer(self) -> None:
        self.clear_calls += 1
        if self._turn_scripts:
            # Clear only transient buffered chunks (e.g. stale pre-turn audio or echo).
            self._buffer.clear()

    def push_mic_audio(self, chunk: np.ndarray) -> None:
        """Simulate asynchronous microphone audio arriving (e.g., speaker echo)."""
        if self._paused:
            self.dropped_while_paused += 1
            return
        self._buffer.append(chunk)

    def read_chunk(self, timeout_s: float = 0.2) -> np.ndarray | None:
        if self._paused or not self._buffer:
            return None
        return self._buffer.pop(0)


class FakePlayer:
    """In-memory AudioPlaybackSink recording played buffers and simulating speaker echo."""

    def __init__(
        self,
        *,
        trigger_barge_in_on_call: int | None = None,
        echo_target: FakeCapture | None = None,
    ) -> None:
        self.played: list[tuple[np.ndarray, int]] = []
        self.trigger_barge_in_on_call = trigger_barge_in_on_call
        self.echo_target = echo_target
        self.calls = 0
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def reset_cancel(self) -> None:
        self.cancelled = False

    def play(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        stop_event: threading.Event | None = None,
    ) -> tuple[float, bool]:
        self.calls += 1
        self.played.append((waveform.copy(), sample_rate))
        if self.echo_target is not None:
            for _ in range(6):
                self.echo_target.push_mic_audio(np.full(512, 0.35, dtype=np.float32))
        if (
            self.trigger_barge_in_on_call is not None
            and self.calls == self.trigger_barge_in_on_call
            and stop_event is not None
        ):
            stop_event.set()
        interrupted = bool(
            self.cancelled or (stop_event is not None and stop_event.is_set())
        )
        return 12.5, interrupted


def _make_chunks(spec: list[tuple[str, int, float]], chunk_size: int = 512) -> list[np.ndarray]:
    """Generate tagged audio chunks where amplitude encodes speech vs silence for fake VAD."""
    chunks: list[np.ndarray] = []
    for kind, count, amp in spec:
        val = amp if kind == "speech" else 0.0
        for _ in range(count):
            chunks.append(np.full(chunk_size, val, dtype=np.float32))
    return chunks


def _fake_vad_detector(cfg: AppConfig) -> SileroVadDetector:
    vad = SileroVadDetector(cfg)
    vad._model = object()
    vad.load = MagicMock(return_value=10.0)
    vad.reset_states = MagicMock()
    vad.predict_chunk = MagicMock(
        side_effect=lambda chunk, sr=16000: 0.9 if float(np.max(np.abs(chunk))) > 0.05 else 0.05
    )
    return vad


def test_audio_capture_and_vad_endpointing(tmp_path: Path) -> None:
    """Audio capture: verify pre-padding, short-pause tolerance, and device validation."""
    cfg = _isolated_config(tmp_path)
    vad = _fake_vad_detector(cfg)
    detector = SpeechEndpointDetector(vad, cfg)

    # 3 silent frames -> 3 speech frames -> 1 pause frame (<4 hangover) -> 3 speech -> 5 silence
    chunks = _make_chunks(
        [
            ("silence", 3, 0.0),
            ("speech", 3, 0.2),
            ("silence", 1, 0.0),
            ("speech", 3, 0.2),
            ("silence", 5, 0.0),
        ]
    )
    capture = FakeCapture(chunks)
    stop_event = threading.Event()
    onset_called = False

    def on_onset() -> None:
        nonlocal onset_called
        onset_called = True

    captured = detector.capture_utterance(
        capture,
        stop_event=stop_event,
        on_speech_start=on_onset,
        timeout_s=2.0,
    )
    assert onset_called is True
    assert captured is not None
    waveform, vad_ms = captured
    assert vad_ms >= 0.0
    # 2 pre-pad + 3 speech + 1 pause + 3 speech + 4 hangover = 13 frames * 512 samples
    assert waveform.size == 13 * 512

    fake_devices = [
        AudioDeviceInfo(
            index=0,
            name="Built-in Mic",
            max_input_channels=1,
            max_output_channels=0,
            default_samplerate=48000.0,
            is_default_input=True,
            is_default_output=False,
        ),
        AudioDeviceInfo(
            index=1,
            name="Built-in Speakers",
            max_input_channels=0,
            max_output_channels=2,
            default_samplerate=48000.0,
            is_default_input=False,
            is_default_output=True,
        ),
    ]
    with patch("backend.pipeline.audio_io.list_audio_devices", return_value=fake_devices):
        assert resolve_audio_device("Built-in Mic", "input") == 0
        assert resolve_audio_device(1, "output") == 1
        with pytest.raises(AudioDeviceError):
            resolve_audio_device("Nonexistent Mic", "input")


def test_microphone_capture_gating_and_buffer_clearing(tmp_path: Path) -> None:
    """Verify MicrophoneAudioCapture drops audio when paused and clears stale buffers."""
    cfg = _isolated_config(tmp_path)
    mic = MicrophoneAudioCapture(cfg)

    # Initially paused before start/resume: callback drops incoming frames.
    mic._callback(np.full((512, 1), 0.3, dtype=np.float32), 512, None, 0)
    assert mic.is_paused is True
    assert mic.read_chunk(timeout_s=0.01) is None

    # Resume accepts chunks cleanly.
    mic.resume()
    assert mic.is_paused is False
    mic._callback(np.full((512, 1), 0.25, dtype=np.float32), 512, None, 0)
    chunk = mic.read_chunk(timeout_s=0.05)
    assert chunk is not None and chunk.size == 512

    # Push stale audio, then pause -> must flush queue & pending and ignore subsequent frames.
    mic._callback(np.full((512, 1), 0.4, dtype=np.float32), 512, None, 0)
    mic.pause()
    assert mic.is_paused is True
    mic._callback(np.full((512, 1), 0.5, dtype=np.float32), 512, None, 0)
    assert mic.read_chunk(timeout_s=0.01) is None

    mic.resume()
    assert mic.read_chunk(timeout_s=0.01) is None


def test_vad_to_stt_integration(tmp_path: Path) -> None:
    """VAD -> STT: verify captured speech array transcribes via MLX Whisper wrapper."""
    cfg = _isolated_config(tmp_path)
    model_dir = tmp_path / "whisper"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "weights.safetensors").write_bytes(b"weights")
    cfg.stt.local_model_path = model_dir

    vad = _fake_vad_detector(cfg)
    detector = SpeechEndpointDetector(vad, cfg)
    capture = FakeCapture(
        _make_chunks([("silence", 2, 0.0), ("speech", 4, 0.25), ("silence", 5, 0.0)])
    )
    result = detector.capture_utterance(capture, timeout_s=2.0)
    assert result is not None
    waveform, _ = result

    stt = MlxWhisperStt(cfg)
    fake_holder = SimpleNamespace(model=object(), model_path=str(model_dir))
    fake_mlx_whisper = SimpleNamespace(
        transcribe=MagicMock(
            return_value={
                "text": "Newton second law எப்படி work ஆகுது",
                "language": "ta",
                "segments": [],
            }
        )
    )
    fake_modules = {
        "mlx": SimpleNamespace(core=SimpleNamespace(float16="float16")),
        "mlx.core": SimpleNamespace(float16="float16"),
        "mlx_whisper": fake_mlx_whisper,
        "mlx_whisper.transcribe": SimpleNamespace(ModelHolder=fake_holder),
    }
    with patch.dict(sys.modules, fake_modules):
        stt_res = stt.transcribe_array(waveform, 16000)

    assert stt_res.text == "Newton second law எப்படி work ஆகுது"
    assert stt_res.language == "ta"
    assert stt_res.audio_duration_s > 0.0


def test_stt_to_llm_integration(tmp_path: Path) -> None:
    """STT -> LLM: verify transcript passes with history to streaming Ollama client."""
    cfg = _isolated_config(tmp_path)
    seen_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "size": 5_000_000}]})
        if request.url.path == "/api/chat":
            body = json.loads(request.content.decode("utf-8"))
            if not body.get("messages"):
                return httpx.Response(200, json={"done": True})
            seen_payloads.append(body)
            lines = [
                json.dumps({"message": {"role": "assistant", "content": "Force equals "}}),
                json.dumps(
                    {
                        "message": {"role": "assistant", "content": "mass times acceleration."},
                        "done": True,
                        "prompt_eval_count": 32,
                        "eval_count": 8,
                        "eval_duration": 200_000_000,
                        "load_duration": 10_000_000,
                    }
                ),
            ]
            return httpx.Response(200, content="\n".join(lines).encode("utf-8"))
        return httpx.Response(404)

    llm = OllamaClient(
        cfg,
        http_client=httpx.Client(
            base_url=cfg.llm.base_url,
            transport=httpx.MockTransport(handler),
        ),
    )
    conv = ConversationManager(cfg.pipeline.voice_system_prompt, max_turns=4)
    conv.add_turn("Hi!", "Hello! What topic shall we study?")
    streamed: list[str] = []

    llm.warmup()
    res = llm.chat(conv.build_messages("Explain Newton's second law."), on_token=streamed.append)
    assert res.response_text == "Force equals mass times acceleration."
    assert "".join(streamed) == "Force equals mass times acceleration."
    assert len(seen_payloads[0]["messages"]) == 4


def test_llm_to_tts_routing(tmp_path: Path) -> None:
    """LLM -> TTS: verify language routing, text cleaning, and Kokoro Tamil guard."""
    cfg = _isolated_config(tmp_path)
    assert clean_text_for_speech("**Hello** `world`! 🌟") == "Hello world !"
    assert detect_speech_language("Photosynthesis uses sunlight.") == "en"
    assert detect_speech_language("ஒளிச்சேர்க்கை சூரிய ஒளியைப் பயன்படுத்துகிறது.") == "ta"

    kokoro = KokoroTtsEngine(cfg)
    kokoro.synthesize_array = MagicMock(
        return_value=(
            np.zeros(2400, dtype=np.float32),
            24000,
            TtsResult(
                backend="kokoro",
                model_name="kokoro-v1.0.onnx",
                voice="af_heart",
                language="en",
                sample_rate=24000,
                audio_duration_s=0.1,
                load_time_ms=10.0,
                processing_time_ms=20.0,
                rtf=0.2,
            ),
        )
    )
    piper = PiperTtsEngine(cfg)
    piper.synthesize_array = MagicMock(
        return_value=(
            np.zeros(2205, dtype=np.float32),
            22050,
            TtsResult(
                backend="piper",
                model_name="ta_IN-rasa_female-medium",
                voice="ta_IN-rasa_female-medium",
                language="ta",
                sample_rate=22050,
                audio_duration_s=0.1,
                load_time_ms=10.0,
                processing_time_ms=15.0,
                rtf=0.15,
            ),
        )
    )

    router = TtsRouter(cfg, kokoro=kokoro, piper=piper)
    _, sr_en, res_en = router.synthesize("**Force** equals mass times acceleration.")
    assert sr_en == 24000
    assert res_en.backend == "kokoro"

    _, sr_ta, res_ta = router.synthesize("வணக்கம்! இன்று நாம் அறிவியல் படிப்போம்.")
    assert sr_ta == 22050
    assert res_ta.backend == "piper"
    assert res_ta.language == "ta"

    real_kokoro = KokoroTtsEngine(cfg)
    strict_router = TtsRouter(cfg, kokoro=real_kokoro, piper=piper)
    with pytest.raises(UnsupportedLanguageError):
        strict_router.synthesize("வணக்கம்", backend_override="kokoro")


def test_complete_pipeline_multiturn_and_feedback_prevention(tmp_path: Path) -> None:
    """Verify playback echo is dropped, buffers cleared, VAD reset, and listening resumes."""
    cfg = _isolated_config(tmp_path)
    vad = _fake_vad_detector(cfg)

    turn1 = _make_chunks([("silence", 1, 0.0), ("speech", 4, 0.25), ("silence", 5, 0.0)])
    turn2 = _make_chunks([("silence", 1, 0.0), ("speech", 4, 0.25), ("silence", 5, 0.0)])
    turn3 = _make_chunks([("silence", 1, 0.0), ("speech", 4, 0.25), ("silence", 5, 0.0)])
    capture = FakeCapture(turn_scripts=[turn1, turn2, turn3])
    player = FakePlayer(echo_target=capture)

    stt = MagicMock(spec=MlxWhisperStt)
    stt.load.return_value = 50.0
    stt.transcribe_array.side_effect = [
        SttResult(
            text="What is osmosis?",
            language="en",
            audio_duration_s=1.0,
            load_time_ms=50.0,
            processing_time_ms=120.0,
            rtf=0.12,
            model_path="mock",
            segments=[],
        ),
        SttResult(
            text="clear conversation",
            language="en",
            audio_duration_s=1.0,
            load_time_ms=0.0,
            processing_time_ms=80.0,
            rtf=0.08,
            model_path="mock",
            segments=[],
        ),
        SttResult(
            text="நியூட்டன் விதி என்ன?",
            language="ta",
            audio_duration_s=1.0,
            load_time_ms=0.0,
            processing_time_ms=140.0,
            rtf=0.14,
            model_path="mock",
            segments=[],
        ),
    ]

    llm = MagicMock(spec=OllamaClient)
    llm.warmup.return_value = None
    llm.chat.side_effect = [
        SimpleNamespace(
            response_text="Osmosis is water moving across a semipermeable membrane.",
            ttft_ms=150.0,
            total_time_ms=420.0,
            tokens_per_sec=24.0,
        ),
        SimpleNamespace(
            response_text="நியூட்டனின் விதி பொருளின் இயக்கத்தை விளக்குகிறது.",
            ttft_ms=180.0,
            total_time_ms=460.0,
            tokens_per_sec=23.5,
        ),
    ]

    kokoro = MagicMock(spec=KokoroTtsEngine)
    kokoro.synthesize_array.return_value = (
        np.zeros(2400, dtype=np.float32),
        24000,
        TtsResult(
            backend="kokoro",
            model_name="kokoro",
            voice="af_heart",
            language="en",
            sample_rate=24000,
            audio_duration_s=0.1,
            load_time_ms=20.0,
            processing_time_ms=60.0,
            rtf=0.6,
        ),
    )
    piper = MagicMock(spec=PiperTtsEngine)
    piper.synthesize_array.return_value = (
        np.zeros(2205, dtype=np.float32),
        22050,
        TtsResult(
            backend="piper",
            model_name="ta_IN-rasa_female-medium",
            voice="ta_IN-rasa_female-medium",
            language="ta",
            sample_rate=22050,
            audio_duration_s=0.1,
            load_time_ms=20.0,
            processing_time_ms=40.0,
            rtf=0.4,
        ),
    )

    states_seen: list[PipelineState] = []
    sm = PipelineStateMachine(on_change=lambda _prev, curr, _r: states_seen.append(curr))
    pipeline = VoicePipeline(
        cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=TtsRouter(cfg, kokoro=kokoro, piper=piper),
        state_machine=sm,
    )

    metrics = pipeline.run(max_turns=3, warmup=True)
    assert len(metrics) == 3
    # Every playback call attempted to echo 6 speech chunks; all 18 must be dropped.
    assert capture.dropped_while_paused == 18
    assert capture.clear_calls >= 6
    assert capture.resume_calls == 3
    assert vad.reset_states.call_count >= 6
    assert metrics[0].tts_backend == "kokoro"
    assert metrics[1].transcript == "clear conversation"
    assert metrics[2].tts_backend == "piper"
    assert metrics[2].tts_language == "ta"
    assert pipeline.conversation.turn_count == 1
    assert capture.started is True and capture.stopped is True
    assert sm.state == PipelineState.IDLE
    assert PipelineState.LISTENING in states_seen
    assert PipelineState.PROCESSING in states_seen
    assert PipelineState.GENERATING in states_seen
    assert PipelineState.SPEAKING in states_seen
    assert (tmp_path / "latency.jsonl").exists()


def test_pipeline_recovers_from_stt_llm_and_tts_errors(tmp_path: Path) -> None:
    """Verify ERROR -> LISTENING recovery across STT, LLM, and TTS failures."""
    cfg = _isolated_config(tmp_path)
    vad = _fake_vad_detector(cfg)
    capture = FakeCapture([])
    player = FakePlayer()

    stt = MagicMock(spec=MlxWhisperStt)
    llm = MagicMock(spec=OllamaClient)
    kokoro = MagicMock(spec=KokoroTtsEngine)
    piper = MagicMock(spec=PiperTtsEngine)
    errors: list[Exception] = []
    states: list[PipelineState] = []
    sm = PipelineStateMachine(on_change=lambda _prev, curr, _r: states.append(curr))

    pipeline = VoicePipeline(
        cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=TtsRouter(cfg, kokoro=kokoro, piper=piper),
        state_machine=sm,
        on_error=errors.append,
    )
    sample = np.full(8000, 0.2, dtype=np.float32)

    # 1. STT error recovery
    stt.transcribe_array.side_effect = ValueError("Corrupted audio buffer")
    assert pipeline.process_utterance(sample) is None
    assert sm.state == PipelineState.LISTENING

    # 2. LLM error recovery
    stt.transcribe_array.side_effect = None
    stt.transcribe_array.return_value = SttResult(
        text="Hello",
        language="en",
        audio_duration_s=0.5,
        load_time_ms=10.0,
        processing_time_ms=40.0,
        rtf=0.08,
        model_path="mock",
        segments=[],
    )
    llm.chat.side_effect = OllamaUnavailableError("Ollama stopped")
    assert pipeline.process_utterance(sample) is None
    assert sm.state == PipelineState.LISTENING

    # 3. TTS error recovery
    llm.chat.side_effect = None
    llm.chat.return_value = SimpleNamespace(
        response_text="Response text",
        ttft_ms=100.0,
        total_time_ms=250.0,
        tokens_per_sec=20.0,
    )
    kokoro.synthesize_array.side_effect = UnsupportedLanguageError("Missing voice")
    assert pipeline.process_utterance(sample) is None
    assert sm.state == PipelineState.LISTENING

    assert len(errors) == 3
    assert states.count(PipelineState.ERROR) == 3
    assert capture.clear_calls == 3


def test_playback_cancellation_and_barge_in_hook(tmp_path: Path) -> None:
    """Verify playback cancellation stops output cleanly and preserves state recovery."""
    cfg = _isolated_config(tmp_path)
    sm = PipelineStateMachine()
    vad = _fake_vad_detector(cfg)
    capture = FakeCapture([])

    class CancellablePlayer(FakePlayer):
        def play(
            self,
            waveform: np.ndarray,
            sample_rate: int,
            *,
            stop_event: threading.Event | None = None,
        ) -> tuple[float, bool]:
            assert sm.state == PipelineState.SPEAKING
            pipeline.cancel_playback()
            return super().play(waveform, sample_rate, stop_event=stop_event)

    stt = MagicMock(spec=MlxWhisperStt)
    stt.transcribe_array.return_value = SttResult(
        text="Tell me a long story",
        language="en",
        audio_duration_s=1.0,
        load_time_ms=10.0,
        processing_time_ms=30.0,
        rtf=0.03,
        model_path="mock",
        segments=[],
    )
    llm = MagicMock(spec=OllamaClient)
    llm.chat.return_value = SimpleNamespace(
        response_text="Long story text.",
        ttft_ms=80.0,
        total_time_ms=200.0,
        tokens_per_sec=25.0,
    )
    kokoro = MagicMock(spec=KokoroTtsEngine)
    kokoro.synthesize_array.return_value = (
        np.zeros(2400, dtype=np.float32),
        24000,
        TtsResult(
            backend="kokoro",
            model_name="kokoro",
            voice="af_heart",
            language="en",
            sample_rate=24000,
            audio_duration_s=0.1,
            load_time_ms=10.0,
            processing_time_ms=20.0,
            rtf=0.2,
        ),
    )

    player = CancellablePlayer()
    pipeline = VoicePipeline(
        cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=TtsRouter(cfg, kokoro=kokoro),
        state_machine=sm,
    )

    metrics = pipeline.process_utterance(np.full(8000, 0.2, dtype=np.float32))
    assert metrics is not None
    assert metrics.interrupted is True
    assert player.cancelled is True
    assert sm.state == PipelineState.LISTENING

    # Also test SpeakerAudioPlayer cancellation flag without requiring audio hardware.
    real_player = SpeakerAudioPlayer(cfg)
    real_player.cancel()
    assert real_player._cancel_event.is_set() is True
    real_player.reset_cancel()
    assert real_player._cancel_event.is_set() is False


def test_ten_consecutive_turns_without_reloads_or_echo(tmp_path: Path) -> None:
    """Verify 10 consecutive turns run continuously without model reloads or echo loops."""
    cfg = _isolated_config(tmp_path)
    cfg.pipeline.max_history_turns = 4
    vad = _fake_vad_detector(cfg)

    turn_scripts = [
        _make_chunks([("silence", 1, 0.0), ("speech", 4, 0.25), ("silence", 5, 0.0)])
        for _ in range(10)
    ]
    capture = FakeCapture(turn_scripts=turn_scripts)
    player = FakePlayer(echo_target=capture)

    stt = MagicMock(spec=MlxWhisperStt)
    stt.load.return_value = 50.0
    stt.transcribe_array.side_effect = [
        SttResult(
            text=f"Question {i + 1}",
            language="ta" if i % 3 == 2 else "en",
            audio_duration_s=0.8,
            load_time_ms=0.0,
            processing_time_ms=50.0,
            rtf=0.06,
            model_path="mock",
            segments=[],
        )
        for i in range(10)
    ]

    llm = MagicMock(spec=OllamaClient)
    llm.warmup.return_value = None
    llm.chat.side_effect = [
        SimpleNamespace(
            response_text=("பதில் " if i % 3 == 2 else "Answer ") + str(i + 1),
            ttft_ms=100.0,
            total_time_ms=220.0,
            tokens_per_sec=24.0,
        )
        for i in range(10)
    ]

    kokoro = MagicMock(spec=KokoroTtsEngine)
    kokoro.synthesize_array.return_value = (
        np.zeros(2400, dtype=np.float32),
        24000,
        TtsResult(
            backend="kokoro",
            model_name="kokoro",
            voice="af_heart",
            language="en",
            sample_rate=24000,
            audio_duration_s=0.1,
            load_time_ms=10.0,
            processing_time_ms=20.0,
            rtf=0.2,
        ),
    )
    piper = MagicMock(spec=PiperTtsEngine)
    piper.synthesize_array.return_value = (
        np.zeros(2205, dtype=np.float32),
        22050,
        TtsResult(
            backend="piper",
            model_name="ta_IN-rasa_female-medium",
            voice="ta_IN-rasa_female-medium",
            language="ta",
            sample_rate=22050,
            audio_duration_s=0.1,
            load_time_ms=10.0,
            processing_time_ms=15.0,
            rtf=0.15,
        ),
    )

    pipeline = VoicePipeline(
        cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=TtsRouter(cfg, kokoro=kokoro, piper=piper),
    )

    metrics = pipeline.run(max_turns=10, warmup=True)
    assert len(metrics) == 10
    assert [m.transcript for m in metrics] == [f"Question {i + 1}" for i in range(10)]
    assert vad.load.call_count == 1
    assert stt.load.call_count == 1
    assert llm.warmup.call_count == 1
    assert capture.dropped_while_paused == 60
    assert pipeline.conversation.turn_count == 4
    assert pipeline.conversation.turns[-1].user == "Question 10"


def test_conversation_manager_bounds_and_clear() -> None:
    """Conversation history: verify sliding window bounds, context, and clear."""
    conv = ConversationManager("System prompt", max_turns=2, max_chars=80)
    conv.add_turn("Question 1", "Answer 1")
    conv.add_turn("Question 2", "Answer 2")
    conv.add_turn("Question 3", "Answer 3")

    assert conv.turn_count == 2
    assert conv.turns[0].user == "Question 2"
    assert conv.turns[1].user == "Question 3"

    msgs = conv.build_messages("Question 4", extra_context="Remember F=ma.")
    assert msgs[0]["role"] == "system"
    assert "Remember F=ma." in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "Question 4"}

    conv.clear()
    assert conv.turn_count == 0
    assert conv.build_messages() == [{"role": "system", "content": "System prompt"}]
