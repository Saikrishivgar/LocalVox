"""Language-aware TTS routing across Kokoro (English) and Piper (English/Tamil)."""

import re

import numpy as np

from backend.config import AppConfig, get_config
from backend.logger import get_logger
from backend.tts.kokoro_tts import KokoroTtsEngine
from backend.tts.piper_tts import PiperTtsEngine
from backend.tts.types import TtsResult, UnsupportedLanguageError

_MARKDOWN_RE = re.compile(r"[*_`~#>]+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\u200D"
    "\uFE0F"
    "]+",
    flags=re.UNICODE,
)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text_for_speech(text: str) -> str:
    """Strip Markdown tokens, emojis, and extra whitespace for natural TTS synthesis."""
    cleaned = _EMOJI_RE.sub(" ", text)
    cleaned = _MARKDOWN_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def detect_speech_language(text: str, stt_language: str | None = None) -> str:
    """Infer whether synthesized response text is predominantly Tamil ('ta') or English ('en')."""
    tamil_chars = sum(1 for ch in text if "\u0B80" <= ch <= "\u0BFF")
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    if tamil_chars > 0 and (alpha_chars == 0 or (tamil_chars / alpha_chars) >= 0.15):
        return "ta"
    if stt_language and stt_language.lower().startswith("ta") and tamil_chars > 0:
        return "ta"
    return "en"


class TtsRouter:
    """Route local speech synthesis between Kokoro ONNX and Piper ONNX."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        kokoro: KokoroTtsEngine | None = None,
        piper: PiperTtsEngine | None = None,
    ) -> None:
        self._config = config if config is not None else get_config()
        self._logger = get_logger("localvox.tts.router", self._config)
        self._kokoro = kokoro
        self._piper = piper

    @property
    def kokoro(self) -> KokoroTtsEngine:
        """Return lazy or injected Kokoro TTS engine."""
        if self._kokoro is None:
            self._kokoro = KokoroTtsEngine(self._config)
        return self._kokoro

    @property
    def piper(self) -> PiperTtsEngine:
        """Return lazy or injected Piper TTS engine."""
        if self._piper is None:
            self._piper = PiperTtsEngine(self._config)
        return self._piper

    def preload(self, routing: str | None = None) -> None:
        """Warm configured local TTS engine(s) when model files exist on disk."""
        mode = routing if routing is not None else self._config.pipeline.tts_routing
        if mode in ("auto", "kokoro") and self._config.tts.kokoro.model_path.exists():
            self.kokoro.load()
        if mode == "piper":
            if self._config.tts.piper.en_model_path.exists():
                self.piper.load("en")
        if mode in ("auto", "piper") and self._config.tts.piper.ta_model_path.exists():
            self.piper.load("ta")

    def synthesize(
        self,
        text: str,
        *,
        stt_language: str | None = None,
        language_override: str | None = None,
        backend_override: str | None = None,
        voice_override: str | None = None,
    ) -> tuple[np.ndarray, int, TtsResult]:
        """Synthesize text using the configured routing policy without cloud fallbacks."""
        spoken_text = clean_text_for_speech(text)
        if not spoken_text:
            raise ValueError("Cannot synthesize empty text after cleaning.")

        target_lang = (
            language_override.lower().strip()
            if language_override
            else detect_speech_language(spoken_text, stt_language=stt_language)
        )
        mode = backend_override or self._config.pipeline.tts_routing

        if mode == "kokoro":
            return self.kokoro.synthesize_array(
                spoken_text,
                language=target_lang,
                voice=voice_override,
            )

        if mode == "piper":
            return self.piper.synthesize_array(spoken_text, language=target_lang)

        # Auto routing: Kokoro for English, Piper for Tamil.
        if target_lang in ("en", "en-us", "en-gb"):
            if self._config.tts.kokoro.model_path.exists() or self._kokoro is not None:
                return self.kokoro.synthesize_array(
                    spoken_text,
                    language=target_lang,
                    voice=voice_override,
                )
            return self.piper.synthesize_array(spoken_text, language="en")

        if target_lang == "ta":
            return self.piper.synthesize_array(spoken_text, language="ta")

        raise UnsupportedLanguageError(
            f"Unsupported TTS language '{target_lang}' in offline LocalVox router."
        )

