"""Piper ONNX Text-to-Speech wrapper for LocalVox."""

import wave
from pathlib import Path

import soundfile as sf
from piper import PiperVoice, SynthesisConfig

from backend.config import AppConfig, get_config
from backend.eval.benchmark import compute_rtf, get_disk_size_mb
from backend.logger import get_logger, stage_timer
from backend.tts.types import TtsResult, UnsupportedLanguageError


class PiperTtsEngine:
    """Offline Piper ONNX TTS wrapper supporting English and community Tamil voices."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config if config is not None else get_config()
        self._piper_cfg = self._config.tts.piper
        self._logger = get_logger("localvox.tts.piper", self._config)
        self._voices: dict[str, PiperVoice] = {}
        self._load_times_ms: dict[str, float] = {}

    def _resolve_voice_paths(self, language: str) -> tuple[str, Path, Path]:
        norm_lang = language.lower().strip()
        if norm_lang not in self._piper_cfg.supported_languages:
            raise UnsupportedLanguageError(
                f"Piper TTS is not configured for language '{language}'. "
                f"Supported: {self._piper_cfg.supported_languages}."
            )
        if norm_lang == "en":
            return (
                self._piper_cfg.en_model_name,
                self._piper_cfg.en_model_path,
                self._piper_cfg.en_config_path,
            )
        if norm_lang == "ta":
            return (
                self._piper_cfg.ta_model_name,
                self._piper_cfg.ta_model_path,
                self._piper_cfg.ta_config_path,
            )
        raise UnsupportedLanguageError(f"Unsupported Piper language '{language}'.")

    def model_size_mb(self, language: str = "en") -> float | None:
        """Return total disk size in MiB for the selected Piper voice model and JSON config."""
        _, model_path, config_path = self._resolve_voice_paths(language)
        model_mb = get_disk_size_mb(model_path)
        cfg_mb = get_disk_size_mb(config_path)
        if model_mb is None or cfg_mb is None:
            return None
        return round(model_mb + cfg_mb, 2)

    def load(self, language: str = "en") -> float:
        """Load local Piper ONNX voice files for the given language."""
        norm_lang = language.lower().strip()
        if norm_lang in self._voices:
            return self._load_times_ms[norm_lang]

        model_name, model_path, config_path = self._resolve_voice_paths(norm_lang)
        if not model_path.exists():
            raise FileNotFoundError(f"Piper ONNX model not found at {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Piper voice config not found at {config_path}")

        with stage_timer("tts_piper_load", log_path=self._config.logging.latency_log_file) as rec:
            voice = PiperVoice.load(model_path, config_path=config_path)
            rec["model"] = model_name
            rec["language"] = norm_lang

        load_ms = float(rec["latency_ms"])
        self._voices[norm_lang] = voice
        self._load_times_ms[norm_lang] = load_ms
        self._logger.info("Piper voice loaded (model=%s, load_ms=%.2f)", model_name, load_ms)
        return load_ms

    def synthesize(
        self,
        text: str,
        output_path: Path | str,
        *,
        language: str = "en",
        length_scale: float | None = None,
    ) -> TtsResult:
        """Synthesize speech to a WAV file using local Piper ONNX."""
        if not text.strip():
            raise ValueError("Input text for TTS synthesis must not be empty.")

        norm_lang = language.lower().strip()
        load_ms = self.load(norm_lang)
        model_name, _, _ = self._resolve_voice_paths(norm_lang)
        voice = self._voices[norm_lang]
        scale = length_scale if length_scale is not None else self._piper_cfg.length_scale

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        syn_cfg = SynthesisConfig(length_scale=scale)

        with stage_timer("tts_piper", log_path=self._config.logging.latency_log_file) as rec:
            with wave.open(str(out_file), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file, syn_config=syn_cfg)
            rec["model"] = model_name
            rec["language"] = norm_lang

        processing_ms = float(rec["latency_ms"])
        info = sf.info(str(out_file))
        if info.frames <= 0:
            raise RuntimeError(f"Piper generated an empty WAV file: {out_file}")

        audio_duration_s = float(info.duration)
        rtf = compute_rtf(processing_ms, audio_duration_s)
        self._logger.info(
            "Piper synthesis done (model=%s, lang=%s, dur_s=%.2f, latency_ms=%.2f, rtf=%s)",
            model_name,
            norm_lang,
            audio_duration_s,
            processing_ms,
            rtf,
        )
        return TtsResult(
            backend="piper",
            model_name=model_name,
            voice=model_name,
            language=norm_lang,
            output_path=out_file,
            sample_rate=info.samplerate,
            audio_duration_s=round(audio_duration_s, 3),
            load_time_ms=round(load_ms, 2),
            processing_time_ms=round(processing_ms, 2),
            rtf=rtf,
        )
