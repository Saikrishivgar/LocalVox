"""Local openWakeWord ONNX model artifact management and verification."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from backend.config import WakeWordConfig
from backend.eval.benchmark import get_disk_size_mb
from backend.wakeword.events import WakeWordModelMissingError

OPENWAKEWORD_RELEASE_BASE = (
    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
)

OPENWAKEWORD_ONNX_ARTIFACTS: dict[str, str] = {
    "melspectrogram.onnx": f"{OPENWAKEWORD_RELEASE_BASE}/melspectrogram.onnx",
    "embedding_model.onnx": f"{OPENWAKEWORD_RELEASE_BASE}/embedding_model.onnx",
    "hey_jarvis_v0.1.onnx": f"{OPENWAKEWORD_RELEASE_BASE}/hey_jarvis_v0.1.onnx",
}


def required_model_paths(cfg: WakeWordConfig) -> tuple[Path, Path, Path]:
    """Return (melspectrogram_path, embedding_path, classifier_path) from config."""
    return (
        cfg.melspectrogram_model,
        cfg.embedding_model,
        cfg.classifier_model,
    )


def missing_model_paths(cfg: WakeWordConfig) -> list[Path]:
    """Return list of required ONNX files that are not yet present on disk."""
    return [path for path in required_model_paths(cfg) if not path.exists()]


def check_wakeword_models(cfg: WakeWordConfig) -> bool:
    """Return True only when all three required openWakeWord ONNX files exist locally."""
    return len(missing_model_paths(cfg)) == 0


def ensure_wakeword_models_exist(cfg: WakeWordConfig) -> tuple[Path, Path, Path]:
    """Validate all required local ONNX model artifacts exist; never download silently."""
    missing = missing_model_paths(cfg)
    if missing:
        joined = ", ".join(str(p) for p in missing)
        raise WakeWordModelMissingError(
            f"Missing local openWakeWord ONNX model file(s): {joined}. "
            "Run `make download-models` once while online to fetch openWakeWord ONNX artifacts."
        )
    return required_model_paths(cfg)


def wakeword_disk_size_mb(cfg: WakeWordConfig) -> float | None:
    """Return combined disk size in MiB across the three local ONNX models, or None."""
    sizes: list[float] = []
    for path in required_model_paths(cfg):
        size_mb = get_disk_size_mb(path)
        if size_mb is None:
            return None
        sizes.append(size_mb)
    return round(sum(sizes), 2)


def download_wakeword_models(target_dir: Path) -> list[Path]:
    """Download official openWakeWord v0.5.1 ONNX models into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for filename, url in OPENWAKEWORD_ONNX_ARTIFACTS.items():
        dst = target_dir / filename
        if not dst.exists():
            urllib.request.urlretrieve(url, dst)
        saved.append(dst)
    return saved
