"""Manual hardware test script for microphone capture, Silero VAD, and speaker playback."""

import argparse
import sys
import threading

import numpy as np

from backend.config import get_config
from backend.pipeline.audio_io import (
    AudioDeviceError,
    MicrophoneAudioCapture,
    SpeakerAudioPlayer,
    SpeechEndpointDetector,
    list_audio_devices,
)
from backend.stt.vad import SileroVadDetector


def _parse_device(raw: str | None) -> int | str | None:
    if raw is None:
        return None
    return int(raw) if raw.isdigit() else raw


def parse_args() -> argparse.Namespace:
    """Parse CLI options for manual microphone and speaker verification."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify physical microphone capture, Silero VAD endpointing, and speaker output."
        )
    )
    parser.add_argument("--input-device", type=str, default=None, help="Mic index or substring.")
    parser.add_argument("--output-device", type=str, default=None, help="Speaker index/substring.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Max wait seconds for speech.")
    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help="Skip mic capture and only play a short speaker test chime.",
    )
    return parser.parse_args()


def main() -> int:
    """Test local microphone capture + Silero VAD and echo captured speech to speakers."""
    args = parse_args()
    cfg = get_config()

    try:
        devices = list_audio_devices()
    except AudioDeviceError as exc:
        print(f"Device listing failed: {exc}", file=sys.stderr)
        return 1

    print("Detected audio devices:")
    for d in devices:
        if d.max_input_channels or d.max_output_channels:
            print(
                f"  [{d.index}] {d.name} "
                f"(in={d.max_input_channels}, out={d.max_output_channels}, "
                f"default_in={d.is_default_input}, default_out={d.is_default_output})"
            )

    player = SpeakerAudioPlayer(cfg, device=_parse_device(args.output_device))
    if args.skip_capture:
        sr = cfg.audio.sample_rate
        t = np.linspace(0.0, 0.4, int(0.4 * sr), endpoint=False, dtype=np.float32)
        chime = 0.15 * np.sin(2.0 * np.pi * 523.25 * t)
        try:
            elapsed_ms, interrupted = player.play(chime, sr)
        except AudioDeviceError as exc:
            print(f"Speaker playback failed: {exc}", file=sys.stderr)
            return 1
        print(f"Played speaker chime: elapsed={elapsed_ms} ms, interrupted={interrupted}")
        return 0

    vad = SileroVadDetector(cfg)
    vad.load()
    detector = SpeechEndpointDetector(vad, cfg)
    capture = MicrophoneAudioCapture(cfg, device=_parse_device(args.input_device))
    stop_event = threading.Event()

    print(f"\nListening for up to {args.timeout:.1f}s. Speak a short sentence...", flush=True)
    try:
        capture.start()
        result = detector.capture_utterance(
            capture,
            stop_event=stop_event,
            on_speech_start=lambda: print("Speech onset detected...", flush=True),
            timeout_s=args.timeout,
        )
    except AudioDeviceError as exc:
        print(f"Microphone capture failed: {exc}", file=sys.stderr)
        return 1
    finally:
        capture.stop()

    if result is None:
        print("No speech detected before timeout.", file=sys.stderr)
        return 1

    waveform, vad_ms = result
    duration_s = round(waveform.size / cfg.audio.sample_rate, 3)
    rms = round(float(np.sqrt(np.mean(np.square(waveform)))), 5)
    print(
        f"Captured utterance: {duration_s} s ({waveform.size} samples), "
        f"rms={rms}, vad_time={vad_ms} ms"
    )
    print("Playing captured audio back through speakers...", flush=True)
    play_ms, interrupted = player.play(waveform, cfg.audio.sample_rate)
    print(f"Playback finished: {play_ms} ms, interrupted={interrupted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
