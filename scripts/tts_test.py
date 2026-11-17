"""Independent CLI verification script for Kokoro and Piper TTS (Component D)."""

import argparse
import json
import sys
from pathlib import Path

from backend.config import get_config
from backend.eval.benchmark import get_process_memory_mb
from backend.tts.kokoro_tts import KokoroTtsEngine
from backend.tts.piper_tts import PiperTtsEngine
from backend.tts.types import UnsupportedLanguageError


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local TTS evaluation."""
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Test Kokoro and Piper TTS locally.")
    parser.add_argument("--backend", choices=["kokoro", "piper"], default=cfg.tts.default_backend)
    parser.add_argument("--text", type=str, default="Hello! Let us study physics together.")
    parser.add_argument("--language", type=str, default=cfg.tts.default_language)
    parser.add_argument("--voice", type=str, default=None, help="Kokoro voice ID override.")
    parser.add_argument("--output-dir", type=Path, default=cfg.tts.output_dir)
    parser.add_argument("--filename", type=str, default=None, help="Output WAV filename.")
    parser.add_argument("--list-voices", action="store_true", help="List available Kokoro voices.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser.parse_args()


def main() -> int:
    """Synthesize speech locally using Kokoro or Piper and print latency/RTF."""
    args = parse_args()
    cfg = get_config()

    if args.list_voices:
        if args.backend == "kokoro":
            engine = KokoroTtsEngine(cfg)
            print("\n".join(engine.available_voices()))
            return 0
        print(
            f"Piper voices configured: en={cfg.tts.piper.en_model_name}, "
            f"ta={cfg.tts.piper.ta_model_name}"
        )
        return 0

    out_name = args.filename or f"{args.backend}_{args.language}.wav"
    out_path = args.output_dir / out_name

    try:
        if args.backend == "kokoro":
            kokoro = KokoroTtsEngine(cfg)
            result = kokoro.synthesize(
                args.text,
                out_path,
                language=args.language,
                voice=args.voice,
            )
            model_size_mb = kokoro.model_size_mb()
        else:
            piper = PiperTtsEngine(cfg)
            result = piper.synthesize(args.text, out_path, language=args.language)
            model_size_mb = piper.model_size_mb(args.language)
    except (FileNotFoundError, UnsupportedLanguageError, ValueError, RuntimeError) as exc:
        print(f"TTS error: {exc}", file=sys.stderr)
        return 1

    payload = {
        **result.model_dump(mode="json"),
        "model_size_mb": model_size_mb,
        "memory_mb": get_process_memory_mb(),
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"Backend:         {result.backend}")
    print(f"Model / Voice:   {result.model_name} / {result.voice} ({model_size_mb} MiB)")
    print(f"Language:        {result.language}")
    print(f"Output File:     {result.output_path}")
    print(f"Audio Duration:  {result.audio_duration_s} s @ {result.sample_rate} Hz")
    print(f"Load Time:       {result.load_time_ms} ms")
    print(f"Generation Time: {result.processing_time_ms} ms")
    print(f"RTF:             {result.rtf}")
    print(f"Process RSS:     {payload['memory_mb']} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
