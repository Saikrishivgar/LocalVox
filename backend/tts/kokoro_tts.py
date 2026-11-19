"""Kokoro ONNX Text-to-Speech wrapper for LocalVox."""

from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

from backend.config import AppConfig, get_config
from backend.eval.benchmark import compute_rtf, get_disk_size_mb
from backend.logger import get_logger, stage_timer
from backend.tts.types import TtsResult, UnsupportedLanguageError

_KOKORO_LANG_MAP: dict[str, str] = {
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
}


class KokoroTtsEngine:
    """Offline Kokoro ONNX TTS engine wrapper (English supported; Tamil unsupported)."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config if config is not None else get_config()
        self._tts_cfg = self._config.tts
        self._kokoro_cfg = self._tts_cfg.kokoro
        self._logger = get_logger("localvox.tts.kokoro", self._config)
        self._engine: Kokoro | None = None
        self._load_time_ms: float = 0.0

    def model_size_mb(self) -> float | None:
        """Return combined disk size in MiB for the Kokoro model and voices bundle."""
        model_mb = get_disk_size_mb(self._kokoro_cfg.model_path)
        voices_mb = get_disk_size_mb(self._kokoro_cfg.voices_path)
        if model_mb is None or voices_mb is None:
            return None
        return round(model_mb + voices_mb, 2)

    def load(self) -> float:
        """Load local Kokoro ONNX weights and voice embeddings."""
        if self._engine is not None:
            return self._load_time_ms

        model_path = self._kokoro_cfg.model_path
        voices_path = self._kokoro_cfg.voices_path
        if not model_path.exists():
            raise FileNotFoundError(f"Kokoro ONNX model not found at {model_path}")
        if not voices_path.exists():
            raise FileNotFoundError(f"Kokoro voices file not found at {voices_path}")

        with stage_timer("tts_kokoro_load", log_path=self._config.logging.latency_log_file) as rec:
            self._engine = Kokoro(str(model_path), str(voices_path))
            rec["model"] = self._kokoro_cfg.model_name

        self._load_time_ms = float(rec["latency_ms"])
        self._logger.info("Kokoro ONNX loaded (load_ms=%.2f)", self._load_time_ms)
        return self._load_time_ms

    def available_voices(self) -> list[str]:
        """Return voice identifiers packaged in voices-v1.0.bin."""
        self.load()
        assert self._engine is not None
        return list(self._engine.get_voices())

    def synthesize_array(
        self,
        text: str,
        *,
        language: str = "en",
        voice: str | None = None,
        speed: float | None = None,
    ) -> tuple[np.ndarray, int, TtsResult]:
        """Synthesize speech in memory and return (waveform, sample_rate, TtsResult)."""
        if not text.strip():
            raise ValueError("Input text for TTS synthesis must not be empty.")

        norm_lang = language.lower().strip()
        supported = self._kokoro_cfg.supported_languages
        if norm_lang not in supported or norm_lang not in _KOKORO_LANG_MAP:
            raise UnsupportedLanguageError(
                f"Kokoro v1.0 does not support language '{language}'. "
                f"Supported LocalVox languages: {self._kokoro_cfg.supported_languages}. "
                "Tamil (ta) is NOT supported by Kokoro v1.0."
            )

        load_ms = self.load()
        assert self._engine is not None

        chosen_voice = voice if voice is not None else self._kokoro_cfg.default_voice
        voices = self._engine.get_voices()
        if chosen_voice not in voices:
            raise ValueError(f"Unknown Kokoro voice '{chosen_voice}'. Sample voices: {voices[:10]}")

        chosen_speed = speed if speed is not None else self._kokoro_cfg.speed
        kokoro_lang = _KOKORO_LANG_MAP[norm_lang]

        with stage_timer("tts_kokoro", log_path=self._config.logging.latency_log_file) as rec:
            audio, sample_rate = self._engine.create(
                text,
                voice=chosen_voice,
                speed=chosen_speed,
                lang=kokoro_lang,
            )
            rec["model"] = self._kokoro_cfg.model_name
            rec["voice"] = chosen_voice
            rec["language"] = norm_lang

        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise RuntimeError("Kokoro generated empty audio.")

        processing_ms = float(rec["latency_ms"])
        audio_duration_s = float(waveform.size / sample_rate)
        rtf = compute_rtf(processing_ms, audio_duration_s)
        self._logger.info(
            "Kokoro synthesis done (voice=%s, dur_s=%.2f, latency_ms=%.2f, rtf=%s)",
            chosen_voice,
            audio_duration_s,
            processing_ms,
            rtf,
        )
        result = TtsResult(
            backend="kokoro",
            model_name=self._kokoro_cfg.model_name,
            voice=chosen_voice,
            language=norm_lang,
            output_path=None,
            sample_rate=sample_rate,
            audio_duration_s=round(audio_duration_s, 3),
            load_time_ms=round(load_ms, 2),
            processing_time_ms=round(processing_ms, 2),
            rtf=rtf,
        )
        return waveform, sample_rate, result

    def synthesize(
        self,
        text: str,
        output_path: Path | str,
        *,
        language: str = "en",
        voice: str | None = None,
        speed: float | None = None,
    ) -> TtsResult:
        """Synthesize speech to a WAV file using local Kokoro ONNX."""
        waveform, sample_rate, result = self.synthesize_array(
            text,
            language=language,
            voice=voice,
            speed=speed,
        )
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_file), waveform, sample_rate)
        info = sf.info(str(out_file))
        if info.frames <= 0:
            raise RuntimeError(f"Kokoro generated an empty WAV file: {out_file}")
        return result.model_copy(update={"output_path": out_file})
