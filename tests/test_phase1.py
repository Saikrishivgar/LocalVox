"""Automated tests for Phase 1 configuration, benchmarks, and component guards."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import soundfile as sf
import yaml
from pydantic import ValidationError

from backend.config import AppConfig, load_config
from backend.eval.benchmark import (
    BenchmarkRecord,
    compute_rtf,
    format_markdown_table,
    get_disk_size_mb,
    get_process_memory_mb,
    load_benchmark_json,
    save_benchmark_json,
)
from backend.llm.ollama_client import (
    OllamaClient,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from backend.stt.vad import SileroVadDetector
from backend.stt.whisper_stt import MlxWhisperStt
from backend.tts.kokoro_tts import KokoroTtsEngine
from backend.tts.piper_tts import PiperTtsEngine
from backend.tts.types import UnsupportedLanguageError


def _make_isolated_config(tmp_path: Path) -> AppConfig:
    """Create a validated AppConfig whose paths point inside tmp_path."""
    raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    raw["logging"]["app_log_file"] = str(tmp_path / "app.log")
    raw["logging"]["latency_log_file"] = str(tmp_path / "latency.jsonl")
    raw["paths"]["app_log_file"] = str(tmp_path / "app.log")
    raw["paths"]["latency_log_file"] = str(tmp_path / "latency.jsonl")
    raw["stt"]["local_model_path"] = str(tmp_path / "missing_whisper")
    raw["tts"]["output_dir"] = str(tmp_path / "tts_out")
    raw["tts"]["kokoro"]["model_path"] = str(tmp_path / "kokoro.onnx")
    raw["tts"]["kokoro"]["voices_path"] = str(tmp_path / "voices.bin")
    raw["tts"]["piper"]["en_model_path"] = str(tmp_path / "en.onnx")
    raw["tts"]["piper"]["en_config_path"] = str(tmp_path / "en.onnx.json")
    raw["tts"]["piper"]["ta_model_path"] = str(tmp_path / "ta.onnx")
    raw["tts"]["piper"]["ta_config_path"] = str(tmp_path / "ta.onnx.json")
    return AppConfig.model_validate(raw)


def test_phase1_config_sections() -> None:
    """Verify VAD, STT, LLM, TTS, benchmark, and logging sections load cleanly."""
    cfg = load_config()
    assert cfg.vad.backend in ("onnx", "torch")
    assert cfg.vad.sample_rate in (8000, 16000)
    assert cfg.stt.backend == "mlx"
    assert "ta" in cfg.stt.supported_languages
    assert cfg.llm.backend == "ollama"
    assert cfg.llm.model_name == "qwen3:8b"
    assert "ta" not in cfg.tts.kokoro.supported_languages
    assert "ta" in cfg.tts.piper.supported_languages
    assert cfg.benchmark.benchmark_runs >= 1
    assert cfg.logging.level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def test_invalid_config_handling(tmp_path: Path) -> None:
    """Verify Pydantic validation rejects invalid fields, bounds, and missing files."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")

    base_raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    bad_extra = copy.deepcopy(base_raw)
    bad_extra["vad"]["unknown_flag"] = True
    with pytest.raises(ValidationError):
        AppConfig.model_validate(bad_extra)

    bad_threshold = copy.deepcopy(base_raw)
    bad_threshold["vad"]["threshold"] = 1.5
    with pytest.raises(ValidationError):
        AppConfig.model_validate(bad_threshold)

    bad_sr = copy.deepcopy(base_raw)
    bad_sr["vad"]["sample_rate"] = 44100
    with pytest.raises(ValidationError):
        AppConfig.model_validate(bad_sr)

    bad_llm = copy.deepcopy(base_raw)
    bad_llm["llm"]["max_tokens"] = 0
    with pytest.raises(ValidationError):
        AppConfig.model_validate(bad_llm)


def test_benchmark_utilities(tmp_path: Path) -> None:
    """Verify benchmark metrics, disk/memory helpers, JSON I/O, and Markdown rendering."""
    assert compute_rtf(250.0, 1.0) == 0.25
    assert compute_rtf(100.0, 0.0) is None
    assert get_process_memory_mb() > 0.0

    sample_file = tmp_path / "weights.bin"
    sample_file.write_bytes(b"0" * 1024 * 512)
    assert get_disk_size_mb(sample_file) == 0.5
    assert get_disk_size_mb(tmp_path) == 0.5
    assert get_disk_size_mb(tmp_path / "missing") is None

    rec = BenchmarkRecord(
        component="VAD",
        model_name="silero-vad",
        language="en",
        measurement_type="measured",
        model_size_mb=2.22,
        load_time_ms=15.0,
        inference_time_ms=4.5,
        memory_mb=120.0,
        audio_duration_s=2.0,
        rtf=compute_rtf(4.5, 2.0),
        hardware="Apple M5",
        notes="Local ONNX check",
    )
    json_path = tmp_path / "bench.json"
    save_benchmark_json([rec], json_path)
    loaded = load_benchmark_json(json_path)
    assert len(loaded) == 1
    assert loaded[0].model_name == "silero-vad"
    assert loaded[0].rtf == 0.0022

    md = format_markdown_table(loaded)
    assert "| VAD | `silero-vad` | measured |" in md


def test_vad_validation_and_mocked_detection(tmp_path: Path) -> None:
    """Verify VAD rejects invalid inputs and formats mocked Silero timestamps."""
    cfg = _make_isolated_config(tmp_path)
    detector = SileroVadDetector(cfg)

    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), np.zeros(16000, dtype=np.float32), 16000)

    with pytest.raises(FileNotFoundError):
        detector.detect(tmp_path / "missing.wav")

    empty_wav = tmp_path / "empty.wav"
    empty_wav.write_bytes(b"")
    with pytest.raises(ValueError):
        detector.detect(empty_wav)

    with pytest.raises(ValueError):
        detector.detect(wav_path, sample_rate=44100)

    with pytest.raises(ValueError):
        detector.detect(wav_path, threshold=1.2)

    with (
        patch("backend.stt.vad.load_silero_vad", return_value=object()),
        patch(
            "backend.stt.vad.get_speech_timestamps",
            return_value=[{"start": 0.25, "end": 0.85}],
        ),
    ):
        result = detector.detect(wav_path)

    assert result.speech_detected is True
    assert len(result.segments) == 1
    assert result.segments[0].start_s == 0.25
    assert result.segments[0].end_s == 0.85
    assert result.audio_duration_s == 1.0
    assert (tmp_path / "latency.jsonl").exists()


def test_stt_validation_without_downloading_models(tmp_path: Path) -> None:
    """Verify STT checks audio format and local model files without hub downloads."""
    cfg = _make_isolated_config(tmp_path)
    stt = MlxWhisperStt(cfg)

    bad_ext = tmp_path / "notes.txt"
    bad_ext.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        stt.transcribe(bad_ext)

    valid_wav = tmp_path / "sample.wav"
    sf.write(str(valid_wav), np.zeros(8000, dtype=np.float32), 16000)

    with pytest.raises(FileNotFoundError):
        stt.transcribe(valid_wav)


def test_ollama_client_mocked(tmp_path: Path) -> None:
    """Verify Ollama health checks, missing model handling, and streaming stats."""
    cfg = _make_isolated_config(tmp_path)

    def offline_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    offline_client = OllamaClient(
        cfg,
        http_client=httpx.Client(
            base_url=cfg.llm.base_url,
            transport=httpx.MockTransport(offline_handler),
        ),
    )
    with pytest.raises(OllamaUnavailableError):
        offline_client.list_models()

    def missing_model_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3:latest", "size": 1024}]})

    missing_client = OllamaClient(
        cfg,
        http_client=httpx.Client(
            base_url=cfg.llm.base_url,
            transport=httpx.MockTransport(missing_model_handler),
        ),
    )
    with pytest.raises(OllamaModelNotFoundError):
        missing_client.verify_model_exists("qwen3:8b")

    def chat_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen3:8b", "size": 5_200_000_000}]},
            )
        if request.url.path == "/api/chat":
            lines = [
                json.dumps(
                    {"message": {"role": "assistant", "content": "Vanakkam! "}, "done": False}
                ),
                json.dumps(
                    {
                        "message": {"role": "assistant", "content": "Ready to study."},
                        "done": True,
                        "prompt_eval_count": 18,
                        "eval_count": 6,
                        "eval_duration": 100_000_000,
                        "load_duration": 5_000_000,
                    }
                ),
            ]
            return httpx.Response(200, content="\n".join(lines).encode("utf-8"))
        return httpx.Response(404)

    ok_client = OllamaClient(
        cfg,
        http_client=httpx.Client(
            base_url=cfg.llm.base_url,
            transport=httpx.MockTransport(chat_handler),
        ),
    )
    res = ok_client.generate("Hello!")
    assert res.response_text == "Vanakkam! Ready to study."
    assert res.prompt_tokens == 18
    assert res.completion_tokens == 6
    assert res.tokens_per_sec == 60.0
    assert res.load_duration_ms == 5.0


def test_tts_guards_and_language_checks(tmp_path: Path) -> None:
    """Verify Kokoro rejects Tamil and both TTS engines require local model files."""
    cfg = _make_isolated_config(tmp_path)
    kokoro = KokoroTtsEngine(cfg)
    piper = PiperTtsEngine(cfg)

    with pytest.raises(UnsupportedLanguageError):
        kokoro.synthesize("வணக்கம்", tmp_path / "ta.wav", language="ta")

    with pytest.raises(FileNotFoundError):
        kokoro.synthesize("Hello", tmp_path / "en.wav", language="en")

    with pytest.raises(UnsupportedLanguageError):
        piper.synthesize("Bonjour", tmp_path / "fr.wav", language="fr")

    with pytest.raises(FileNotFoundError):
        piper.synthesize("வணக்கம்", tmp_path / "ta.wav", language="ta")

    assert piper.model_size_mb("en") is None
    assert piper.model_size_mb("ta") is None


def test_stt_mocked_transcription(tmp_path: Path) -> None:
    """Verify MlxWhisperStt transcription metrics using mocked MLX modules."""
    model_dir = tmp_path / "whisper_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "weights.safetensors").write_bytes(b"weights")

    wav_path = tmp_path / "speech.wav"
    sf.write(str(wav_path), np.zeros(16000, dtype=np.float32), 16000)

    cfg = _make_isolated_config(tmp_path)
    cfg.stt.local_model_path = model_dir
    stt = MlxWhisperStt(cfg)

    fake_holder = SimpleNamespace(model=object(), model_path=str(model_dir))
    fake_mlx_whisper = SimpleNamespace(
        transcribe=MagicMock(
            return_value={
                "text": " Hello LocalVox ",
                "language": "en",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello LocalVox"}],
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
        result = stt.transcribe(wav_path, language="en")

    assert result.text == "Hello LocalVox"
    assert result.language == "en"
    assert result.audio_duration_s == 1.0
    assert result.rtf is not None
    assert (tmp_path / "latency.jsonl").exists()


def test_tts_mocked_synthesis(tmp_path: Path) -> None:
    """Verify Kokoro and Piper synthesis wrappers produce WAV metadata with mocks."""
    cfg = _make_isolated_config(tmp_path)
    cfg.tts.kokoro.model_path.write_bytes(b"kokoro")
    cfg.tts.kokoro.voices_path.write_bytes(b"voices")
    cfg.tts.piper.en_model_path.write_bytes(b"piper_en")
    cfg.tts.piper.en_config_path.write_text("{}", encoding="utf-8")

    fake_kokoro_instance = MagicMock()
    fake_kokoro_instance.get_voices.return_value = ["af_heart"]
    fake_kokoro_instance.create.return_value = (np.zeros(24000, dtype=np.float32), 24000)

    with patch("backend.tts.kokoro_tts.Kokoro", return_value=fake_kokoro_instance):
        kokoro = KokoroTtsEngine(cfg)
        k_res = kokoro.synthesize("Hello!", tmp_path / "kokoro.wav", language="en")
        assert k_res.audio_duration_s == 1.0
        assert k_res.sample_rate == 24000
        assert k_res.output_path.exists()
        with pytest.raises(ValueError):
            kokoro.synthesize("   ", tmp_path / "empty.wav", language="en")
        with pytest.raises(ValueError):
            kokoro.synthesize("Hello", tmp_path / "bad_voice.wav", voice="missing")

    def fake_synth_wav(text: str, wav_file: object, syn_config: object = None) -> None:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 22050)

    fake_piper_voice = MagicMock()
    fake_piper_voice.synthesize_wav.side_effect = fake_synth_wav

    with patch("backend.tts.piper_tts.PiperVoice.load", return_value=fake_piper_voice):
        piper = PiperTtsEngine(cfg)
        p_res = piper.synthesize("Hello!", tmp_path / "piper.wav", language="en")
        assert p_res.audio_duration_s == 1.0
        assert p_res.sample_rate == 22050
        assert p_res.output_path.exists()

