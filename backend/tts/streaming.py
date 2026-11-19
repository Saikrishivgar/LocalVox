"""Incremental sentence boundary segmentation and queued streaming TTS types."""

import re
from dataclasses import dataclass

import numpy as np

from backend.tts.router import clean_text_for_speech
from backend.tts.types import TtsResult

_COMMON_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "eg",
        "ie",
        "e.g",
        "i.e",
        "fig",
        "eq",
        "no",
        "st",
        "am",
        "pm",
        "a.m",
        "p.m",
    }
)
_SENTENCE_END_CHARS = frozenset(".!?\n\u0964")
_CLAUSE_SPLIT_CHARS = frozenset(",;:\u2014")


@dataclass(frozen=True)
class AudioChunk:
    """Single synthesized speech chunk queued for sequential speaker playback."""

    seq_id: int
    text: str
    waveform: np.ndarray
    sample_rate: int
    tts_result: TtsResult | None = None


def is_speakable_text(text: str) -> bool:
    """Return True if cleaned text contains at least one alphanumeric/word character."""
    cleaned = clean_text_for_speech(text)
    return any(ch.isalnum() for ch in cleaned)


def _is_decimal_dot(buf: str, idx: int) -> bool:
    if buf[idx] != ".":
        return False
    prev_digit = idx > 0 and buf[idx - 1].isdigit()
    next_digit = (idx + 1) < len(buf) and buf[idx + 1].isdigit()
    return prev_digit and next_digit


def _is_abbreviation_dot(buf: str, idx: int) -> bool:
    if buf[idx] != ".":
        return False
    prefix = buf[:idx]
    match = re.search(r"([A-Za-z0-9.]+)$", prefix)
    if not match:
        return False
    token = match.group(1)
    lowered = token.lower()
    if lowered in _COMMON_ABBREVIATIONS:
        return True
    # Avoid breaking single-letter initials ("F. Scott") or enumerated list prefixes ("1. Force").
    return len(token) == 1 and (token.isupper() or token.isdigit())


class SentenceBuffer:
    """Buffer streaming LLM tokens into natural speakable chunks for incremental TTS."""

    def __init__(
        self,
        *,
        min_sentence_chars: int = 16,
        max_chunk_chars: int = 220,
    ) -> None:
        self._min_chars = max(1, int(min_sentence_chars))
        self._max_chars = max(self._min_chars + 10, int(max_chunk_chars))
        self._buffer = ""

    @property
    def pending_text(self) -> str:
        """Return currently buffered un-emitted text."""
        return self._buffer

    def push(self, token: str) -> list[str]:
        """Append token delta and return any complete ready-to-speak sentence chunks."""
        if token:
            self._buffer += token
        return self._extract_ready_chunks()

    def flush(self) -> list[str]:
        """Emit any remaining speakable buffered text at the end of an LLM response."""
        emitted = self._extract_ready_chunks()
        tail = self._buffer.strip()
        self._buffer = ""
        if tail and is_speakable_text(tail):
            emitted.append(tail)
        return emitted

    def clear(self) -> None:
        """Discard any pending partial text (e.g. on interruption or cancellation)."""
        self._buffer = ""

    def _find_sentence_boundary(self) -> int | None:
        buf = self._buffer
        n = len(buf)
        for idx, ch in enumerate(buf):
            if ch not in _SENTENCE_END_CHARS:
                continue
            if ch == "\n":
                candidate = buf[: idx + 1].strip()
                if len(candidate) >= self._min_chars and is_speakable_text(candidate):
                    return idx + 1
                continue
            # Wait for at least one lookahead char so decimals ("3.14") or ellipses ("...") resolve.
            if idx + 1 >= n:
                return None
            next_ch = buf[idx + 1]
            if ch == "." and next_ch == ".":
                continue
            if not (next_ch.isspace() or next_ch in "\"')]\n"):
                continue
            if ch == "." and (_is_decimal_dot(buf, idx) or _is_abbreviation_dot(buf, idx)):
                continue
            candidate = buf[: idx + 1].strip()
            if len(candidate) >= self._min_chars and is_speakable_text(candidate):
                return idx + 1
        return None

    def _find_soft_split_boundary(self) -> int | None:
        buf = self._buffer
        if len(buf) < self._max_chars:
            return None
        window = buf[: self._max_chars]
        for idx in range(len(window) - 1, self._min_chars - 1, -1):
            if window[idx] in _CLAUSE_SPLIT_CHARS and is_speakable_text(window[: idx + 1]):
                return idx + 1
        for idx in range(len(window) - 1, self._min_chars - 1, -1):
            if window[idx].isspace() and is_speakable_text(window[:idx]):
                return idx + 1
        return None

    def _extract_ready_chunks(self) -> list[str]:
        chunks: list[str] = []
        while self._buffer:
            cut = self._find_sentence_boundary()
            if cut is None:
                cut = self._find_soft_split_boundary()
            if cut is None:
                break
            chunk = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:].lstrip()
            if chunk and is_speakable_text(chunk):
                chunks.append(chunk)
        return chunks

