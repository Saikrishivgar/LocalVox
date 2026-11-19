"""Download required local Phase 1 STT and TTS model files into configured paths."""

import argparse
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from backend.config import get_config
from backend.wakeword.models import download_wakeword_models

_KOKORO_RELEASE_BASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
_PIPER_EN_REPO = "rhasspy/piper-voices"
_PIPER_EN_FILES = (
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
)
_PIPER_TA_REPO = "tinisoft/piper-ta_IN-rasa_female-medium"
_STT_REPOS: dict[str, str] = {
    "whisper-small-mlx": "mlx-community/whisper-small-mlx",
    "whisper-large-v3-turbo-8bit": "mlx-community/whisper-large-v3-turbo-8bit",
    "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI options for local model downloads."""
    parser = argparse.ArgumentParser(description="Download LocalVox Phase 1 offline model weights.")
    parser.add_argument(
        "--all-stt",
        action="store_true",
        help="Download all candidate MLX Whisper models instead of only the configured default.",
    )
    return parser.parse_args()


def main() -> int:
    """Download STT (MLX Whisper) and TTS (Kokoro + Piper) weights locally."""
    args = parse_args()
    cfg = get_config()

    stt_targets = cfg.stt.candidate_models if args.all_stt else [cfg.stt.local_model_path]
    for target in stt_targets:
        repo_id = _STT_REPOS.get(target.name)
        if repo_id is None:
            print(f"Skipping unknown STT repo mapping for {target}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {target}")
        snapshot_download(repo_id=repo_id, local_dir=str(target))

    for dst in (cfg.tts.kokoro.model_path, cfg.tts.kokoro.voices_path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            url = f"{_KOKORO_RELEASE_BASE}/{dst.name}"
            print(f"Downloading {url} -> {dst}")
            urllib.request.urlretrieve(url, dst)

    piper_en_root = Path(cfg.paths.tts_models_dir) / "piper" / cfg.tts.piper.en_model_name
    piper_en_root.mkdir(parents=True, exist_ok=True)
    for rel_file in _PIPER_EN_FILES:
        print(f"Downloading {_PIPER_EN_REPO}/{rel_file}")
        hf_hub_download(repo_id=_PIPER_EN_REPO, filename=rel_file, local_dir=str(piper_en_root))

    piper_ta_dir = cfg.tts.piper.ta_model_path.parent
    piper_ta_dir.mkdir(parents=True, exist_ok=True)
    for file_path in (cfg.tts.piper.ta_model_path, cfg.tts.piper.ta_config_path):
        print(f"Downloading {_PIPER_TA_REPO}/{file_path.name}")
        hf_hub_download(
            repo_id=_PIPER_TA_REPO,
            filename=file_path.name,
            local_dir=str(piper_ta_dir),
        )

    print(f"Downloading openWakeWord ONNX models -> {cfg.paths.wakeword_models_dir}")
    download_wakeword_models(cfg.paths.wakeword_models_dir)

    print("Local model downloads complete (STT, TTS, and openWakeWord ONNX).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
