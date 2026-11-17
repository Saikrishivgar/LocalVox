"""Independent CLI verification script for Silero VAD (Component A)."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.config import get_config
from backend.eval.benchmark import get_process_memory_mb
from backend.stt.vad import SileroVadDetector


def create_sample_audio(output_path: Path, sample_rate: int) -> Path:
    """Create a short local WAV sample containing silence and spoken speech."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(0.6 * sample_rate), dtype=np.float32)

    say_bin = shutil.which("say")
    ffmpeg_bin = shutil.which("ffmpeg")
    if say_bin and ffmpeg_bin:
        raw_aiff = output_path.with_suffix(".aiff")
        speech_wav = output_path.with_name(f"{output_path.stem}_speech.wav")
        try:
            subprocess.run(
                [
                    say_bin,
                    "-v",
                    "Samantha",
                    "-o",
                    str(raw_aiff),
                    "Hello LocalVox, testing voice activity detection.",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    str(raw_aiff),
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "1",
                    str(speech_wav),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            speech_data, _ = sf.read(str(speech_wav), dtype="float32")
            speech = np.asarray(speech_data, dtype=np.float32)
            if speech.size > 0:
                sf.write(str(output_path), np.concatenate([silence, speech, silence]), sample_rate)
                return output_path
        except subprocess.SubprocessError:
            pass
        finally:
            raw_aiff.unlink(missing_ok=True)
            speech_wav.unlink(missing_ok=True)

    cfg = get_config()
    if cfg.tts.piper.en_model_path.exists() and cfg.tts.piper.en_config_path.exists():
        from backend.tts.piper_tts import PiperTtsEngine

        tmp_wav = output_path.with_name(f"{output_path.stem}_piper.wav")
        try:
            PiperTtsEngine(cfg).synthesize(
                "Hello LocalVox, testing voice activity detection.",
                tmp_wav,
                language="en",
            )
            from backend.stt.vad import load_audio_mono

            speech, _ = load_audio_mono(tmp_wav, sample_rate)
            sf.write(str(output_path), np.concatenate([silence, speech, silence]), sample_rate)
            return output_path
        finally:
            tmp_wav.unlink(missing_ok=True)

    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    tone = 0.25 * np.sin(2.0 * np.pi * 220.0 * t)
    sf.write(str(output_path), np.concatenate([silence, tone, silence]), sample_rate)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Silero VAD verification."""
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Test Silero VAD locally.")
    parser.add_argument("--audio", type=Path, default=None, help="Path to local audio file.")
    parser.add_argument("--sample-rate", type=int, default=cfg.vad.sample_rate)
    parser.add_argument("--threshold", type=float, default=cfg.vad.threshold)
    parser.add_argument("--min-speech-ms", type=int, default=cfg.vad.min_speech_duration_ms)
    parser.add_argument("--min-silence-ms", type=int, default=cfg.vad.min_silence_duration_ms)
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser.parse_args()


def main() -> int:
    """Run local Silero VAD verification and print timing/timestamp metrics."""
    args = parse_args()
    cfg = get_config()
    audio_path = args.audio
    if audio_path is None:
        sample_wav = cfg.paths.test_audio_dir / "vad_sample.wav"
        audio_path = create_sample_audio(sample_wav, args.sample_rate)

    detector = SileroVadDetector(cfg)
    try:
        result = detector.detect(
            audio_path,
            sample_rate=args.sample_rate,
            threshold=args.threshold,
            min_speech_duration_ms=args.min_speech_ms,
            min_silence_duration_ms=args.min_silence_ms,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"VAD error: {exc}", file=sys.stderr)
        return 1

    payload = {
        **result.model_dump(),
        "audio_file": str(audio_path),
        "model_size_mb": detector.model_size_mb(),
        "memory_mb": get_process_memory_mb(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Model:           {result.model_name} ({payload['model_size_mb']} MiB)")
    print(f"Audio File:      {audio_path} ({result.audio_duration_s} s @ {result.sample_rate} Hz)")
    print(f"Speech Detected: {result.speech_detected}")
    print(f"Speech Regions:  {[seg.model_dump() for seg in result.segments]}")
    print(f"Load Time:       {result.load_time_ms} ms")
    print(f"Processing Time: {result.processing_time_ms} ms")
    print(f"RTF:             {result.rtf}")
    print(f"Process RSS:     {payload['memory_mb']} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
