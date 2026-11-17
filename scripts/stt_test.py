"""Independent CLI verification script for MLX Whisper STT (Component B)."""

import argparse
import json
import sys
from pathlib import Path

from backend.config import get_config
from backend.eval.benchmark import get_process_memory_mb
from backend.stt.whisper_stt import MlxWhisperStt


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for MLX Whisper verification."""
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Test MLX Whisper STT locally.")
    parser.add_argument("audio", type=Path, help="Path to local audio file.")
    parser.add_argument(
        "--language",
        type=str,
        default=cfg.stt.default_language,
        help="Language code ('en', 'ta') or 'auto'/omitted for detection.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=cfg.stt.local_model_path,
        help="Local directory containing MLX Whisper weights.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser.parse_args()


def main() -> int:
    """Transcribe a local audio file and display text, language, latency, and RTF."""
    args = parse_args()
    cfg = get_config()
    stt = MlxWhisperStt(cfg)

    try:
        result = stt.transcribe(args.audio, language=args.language, model_path=args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"STT error: {exc}", file=sys.stderr)
        return 1

    payload = {
        "text": result.text,
        "language": result.language,
        "audio_duration_s": result.audio_duration_s,
        "load_time_ms": result.load_time_ms,
        "processing_time_ms": result.processing_time_ms,
        "rtf": result.rtf,
        "model_path": result.model_path,
        "model_size_mb": stt.model_size_mb(args.model),
        "memory_mb": get_process_memory_mb(),
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"Model:           {args.model} ({payload['model_size_mb']} MiB)")
    print(f"Audio File:      {args.audio} ({result.audio_duration_s} s)")
    print(f"Language:        {result.language}")
    print(f"Recognized Text: {result.text}")
    print(f"Load Time:       {result.load_time_ms} ms")
    print(f"Processing Time: {result.processing_time_ms} ms")
    print(f"RTF:             {result.rtf}")
    print(f"Process RSS:     {payload['memory_mb']} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
