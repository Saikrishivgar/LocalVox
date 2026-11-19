"""Local 3-sample personalized voice enrollment and embedding matcher (zero raw audio saved)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from backend.memory.models import utc_now_iso

REQUIRED_ENROLLMENT_SAMPLES = 3
MIN_SAMPLE_DURATION_S = 0.20
MAX_SAMPLE_DURATION_S = 4.5
MIN_SAMPLE_RMS = 0.0025

EMBEDDING_FRAMES = 16
EMBEDDING_DIM = 96
TEMPLATE_DIM = EMBEDDING_FRAMES * EMBEDDING_DIM  # 1536
MEL_BINS = 32


class VoiceProfileSummary(BaseModel):
    """Public summary of the local personalized 3-sample wake-word voice profile."""

    model_config = ConfigDict(extra="forbid")

    enrolled: bool = False
    samples_recorded: int = 0
    required_samples: int = REQUIRED_ENROLLMENT_SAMPLES
    in_progress_samples: int = 0
    wake_phrase: str = "Hey Jarvis"
    profile_path: str = ""
    updated_at: str | None = None
    personal_enabled: bool = True
    personal_only: bool = False
    personal_threshold: float = 0.68
    consistency_score: float | None = None


class EnrollmentSampleResult(BaseModel):
    """Feedback returned after recording one personalized voice enrollment sample (1..3)."""

    model_config = ConfigDict(extra="forbid")

    sample_index: int = Field(ge=1, le=REQUIRED_ENROLLMENT_SAMPLES)
    required_samples: int = REQUIRED_ENROLLMENT_SAMPLES
    completed: bool = False
    duration_s: float = Field(ge=0.0)
    audio_rms: float = Field(ge=0.0)
    generic_score: float = Field(ge=0.0, le=1.0)
    consistency_with_previous: float | None = None
    message: str
    profile: VoiceProfileSummary


def extract_template_from_trace(
    centered_embs: np.ndarray,
    step_mels: np.ndarray,
    step_rms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract normalized (1536,) centered embedding trajectory and (32,) speaker timbre vector."""
    if centered_embs.ndim != 2 or centered_embs.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected centered_embs shape (T, {EMBEDDING_DIM}), got {centered_embs.shape}."
        )
    if step_mels.ndim != 2 or step_mels.shape[1] != MEL_BINS:
        raise ValueError(f"Expected step_mels shape (T, {MEL_BINS}), got {step_mels.shape}.")
    if step_rms.size == 0 or centered_embs.shape[0] == 0:
        raise ValueError("Empty audio trace for voice enrollment.")

    peak_rms = float(np.max(step_rms))
    if peak_rms < MIN_SAMPLE_RMS:
        raise ValueError("Voice sample was too quiet. Please speak clearly toward the microphone.")

    active_thresh = max(MIN_SAMPLE_RMS, peak_rms * 0.15)
    active_idx = np.where(step_rms >= active_thresh)[0]
    if active_idx.size == 0:
        raise ValueError("No clear speech detected in enrollment sample.")

    # embedding_model uses a 760 ms receptive window, so speech features peak ~4-6 steps
    # (~320-480 ms) after raw speech RMS ends. Anchor the 16-step window cleanly over speech+tail.
    end_idx = min(int(centered_embs.shape[0]), int(active_idx[-1]) + 6)
    start_idx = max(0, end_idx - EMBEDDING_FRAMES)
    window = centered_embs[start_idx:end_idx]
    if window.shape[0] < EMBEDDING_FRAMES:
        pad = np.zeros((EMBEDDING_FRAMES - window.shape[0], EMBEDDING_DIM), dtype=np.float32)
        window = np.vstack([pad, window])

    flat = window.astype(np.float32, copy=False).reshape(TEMPLATE_DIM)
    flat_norm = float(np.linalg.norm(flat))
    if flat_norm < 1.0:
        raise ValueError("Speech features were too weak to enroll. Please say the wake phrase.")
    unit_template = (flat / flat_norm).astype(np.float32)

    active_mels = step_mels[active_idx]
    mel_mean = np.mean(active_mels, axis=0, dtype=np.float64)
    mel_centered = mel_mean - float(np.mean(mel_mean))
    mel_norm = float(np.linalg.norm(mel_centered)) + 1e-8
    unit_timbre = (mel_centered / mel_norm).astype(np.float32)
    return unit_template, unit_timbre


def compute_pairwise_consistency(templates: list[np.ndarray]) -> float:
    """Compute mean pairwise cosine similarity across enrolled unit templates in [0, 1]."""
    if len(templates) < 2:
        return 1.0
    sims: list[float] = []
    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            sims.append(float(np.clip(np.dot(templates[i], templates[j]), 0.0, 1.0)))
    return round(float(np.mean(sims)), 4) if sims else 1.0


@dataclass(frozen=True)
class PersonalizedVoiceProfile:
    """In-memory and disk-persisted mathematical speaker+phrase embedding profile (no raw audio)."""

    templates: np.ndarray  # shape (N, 1536), unit-normalized float32
    centroid: np.ndarray  # shape (1536,), unit-normalized float32
    timbre_centroid: np.ndarray  # shape (32,), unit-normalized float32
    wake_phrase: str
    updated_at: str
    consistency_score: float

    @property
    def sample_count(self) -> int:
        """Number of enrolled voice templates."""
        return int(self.templates.shape[0])

    @classmethod
    def from_samples(
        cls,
        templates: list[np.ndarray],
        timbres: list[np.ndarray],
        *,
        wake_phrase: str,
    ) -> PersonalizedVoiceProfile:
        """Build a normalized voice profile from 3 extracted embedding/timbre sample pairs."""
        if not templates or len(templates) != len(timbres):
            raise ValueError("Templates and timbre vectors must be non-empty and equal length.")

        norm_templates: list[np.ndarray] = []
        for tpl in templates:
            arr = np.asarray(tpl, dtype=np.float32).reshape(TEMPLATE_DIM)
            norm = float(np.linalg.norm(arr)) + 1e-8
            norm_templates.append((arr / norm).astype(np.float32))

        norm_timbres: list[np.ndarray] = []
        for tim in timbres:
            arr = np.asarray(tim, dtype=np.float32).reshape(MEL_BINS)
            norm = float(np.linalg.norm(arr)) + 1e-8
            norm_timbres.append((arr / norm).astype(np.float32))

        tpl_stack = np.stack(norm_templates, axis=0)
        centroid = np.mean(tpl_stack, axis=0, dtype=np.float64)
        centroid = (centroid / (float(np.linalg.norm(centroid)) + 1e-8)).astype(np.float32)

        tim_stack = np.stack(norm_timbres, axis=0)
        timbre_centroid = np.mean(tim_stack, axis=0, dtype=np.float64)
        timbre_centroid = (
            timbre_centroid / (float(np.linalg.norm(timbre_centroid)) + 1e-8)
        ).astype(np.float32)

        return cls(
            templates=tpl_stack,
            centroid=centroid,
            timbre_centroid=timbre_centroid,
            wake_phrase=wake_phrase.strip() or "Hey Jarvis",
            updated_at=utc_now_iso(),
            consistency_score=compute_pairwise_consistency(norm_templates),
        )

    def score_window(
        self,
        centered_emb_window: np.ndarray,
        mel_steps: np.ndarray,
        step_rms_window: np.ndarray,
    ) -> float:
        """Score a live (16, 96) centered embedding window against the enrolled user profile."""
        if step_rms_window.size == 0:
            return 0.0
        peak_rms = float(np.max(step_rms_window))
        if peak_rms < MIN_SAMPLE_RMS:
            return 0.0

        flat = np.asarray(centered_emb_window, dtype=np.float32).reshape(-1)
        if flat.size != TEMPLATE_DIM:
            return 0.0
        win_norm = float(np.linalg.norm(flat))
        if win_norm < 1.0:
            return 0.0
        unit_win = flat / win_norm

        cos_centroid = float(np.dot(unit_win, self.centroid))
        cos_templates = self.templates @ unit_win
        cos_best_tpl = float(np.max(cos_templates)) if cos_templates.size else cos_centroid
        emb_sim = max(0.0, 0.5 * cos_centroid + 0.5 * cos_best_tpl)

        timbre_sim = 0.0
        if mel_steps.ndim == 2 and mel_steps.shape[1] == MEL_BINS and mel_steps.shape[0] > 0:
            active_mask = step_rms_window >= max(MIN_SAMPLE_RMS, peak_rms * 0.2)
            if active_mask.shape[0] == mel_steps.shape[0] and np.any(active_mask):
                mel_mean = np.mean(mel_steps[active_mask], axis=0, dtype=np.float64)
            else:
                mel_mean = np.mean(mel_steps, axis=0, dtype=np.float64)
            mel_centered = mel_mean - float(np.mean(mel_mean))
            mel_norm = float(np.linalg.norm(mel_centered))
            if mel_norm > 1e-6:
                unit_mel = (mel_centered / mel_norm).astype(np.float32)
                timbre_sim = max(0.0, float(np.dot(unit_mel, self.timbre_centroid)))

        combined = emb_sim * (0.75 + 0.25 * timbre_sim)
        return float(np.clip(combined, 0.0, 1.0))

    def save(self, path: Path) -> Path:
        """Persist mathematical voice embeddings atomically to local .npz (never raw audio)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.stem}.tmp.npz")
        np.savez(
            tmp_path,
            templates=self.templates.astype(np.float32),
            centroid=self.centroid.astype(np.float32),
            timbre_centroid=self.timbre_centroid.astype(np.float32),
            wake_phrase=np.array(self.wake_phrase),
            updated_at=np.array(self.updated_at),
            consistency_score=np.array(float(self.consistency_score), dtype=np.float32),
        )
        os.replace(tmp_path, path)
        return path

    @classmethod
    def load(cls, path: Path) -> PersonalizedVoiceProfile | None:
        """Load persisted local .npz voice embeddings if present and valid."""
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                templates = np.asarray(data["templates"], dtype=np.float32)
                centroid = np.asarray(data["centroid"], dtype=np.float32).reshape(TEMPLATE_DIM)
                timbre_centroid = np.asarray(
                    data["timbre_centroid"],
                    dtype=np.float32,
                ).reshape(MEL_BINS)
                if templates.ndim != 2 or templates.shape[1] != TEMPLATE_DIM:
                    return None
                wake_phrase = str(data["wake_phrase"].item())
                updated_at = str(data["updated_at"].item())
                consistency_score = float(data["consistency_score"].item())
            return cls(
                templates=templates,
                centroid=centroid,
                timbre_centroid=timbre_centroid,
                wake_phrase=wake_phrase,
                updated_at=updated_at,
                consistency_score=round(consistency_score, 4),
            )
        except Exception:
            return None

    @staticmethod
    def delete(path: Path) -> bool:
        """Delete local voice profile file cleanly if it exists."""
        if path.exists():
            path.unlink()
            return True
        return False
