"""Main CLI entrypoint to launch the LocalVox Phase 2 offline voice assistant."""

import argparse
import sys
from pathlib import Path

from backend.config import AppConfig, get_config
from backend.memory.manager import MemoryManager
from backend.pipeline.audio_io import (
    AudioDeviceError,
    MicrophoneAudioCapture,
    SpeakerAudioPlayer,
    list_audio_devices,
)
from backend.pipeline.barge_in import BargeInCapture
from backend.pipeline.state import PipelineState, PipelineStateMachine, TurnMetrics
from backend.pipeline.voice_pipeline import VoicePipeline
from backend.wakeword.events import WakeWordDetectionEvent


def _parse_device_arg(raw: str | None) -> int | str | None:
    if raw is None:
        return None
    return int(raw) if raw.isdigit() else raw


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for LocalVox voice assistant."""
    parser = argparse.ArgumentParser(description="LocalVox offline Tamil+English voice assistant.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--input-device", type=str, default=None, help="Mic device index or name.")
    parser.add_argument(
        "--output-device",
        type=str,
        default=None,
        help="Speaker device index or name.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional STT language override ('en', 'ta', or omit for auto-detect).",
    )
    parser.add_argument(
        "--tts-backend",
        choices=["auto", "kokoro", "piper"],
        default=None,
        help="TTS routing policy override.",
    )
    parser.add_argument(
        "--barge-in-sensitivity",
        choices=["low", "medium", "high"],
        default=None,
        help="Override configured barge-in interruption sensitivity.",
    )
    parser.add_argument(
        "--no-barge-in",
        action="store_true",
        help="Disable full-duplex barge-in and fall back to Phase 2.1 half-duplex mic gating.",
    )
    parser.add_argument(
        "--no-streaming-tts",
        action="store_true",
        help="Wait for full LLM completion before synthesizing speech.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable persistent Phase 4 SQLite memory retrieval and extraction.",
    )
    parser.add_argument(
        "--memory-db",
        type=str,
        default=None,
        help="Override SQLite memory database file path.",
    )
    parser.add_argument(
        "--list-memories",
        action="store_true",
        help="Print stored persistent memories from local SQLite DB and exit.",
    )
    parser.add_argument(
        "--wake-word",
        action="store_true",
        help="Run in hands-free 'Hey Jarvis' standby wake-word mode.",
    )
    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="Disable wake-word standby and listen continuously immediately.",
    )
    parser.add_argument(
        "--wake-sensitivity",
        choices=["low", "medium", "high"],
        default=None,
        help="Override wake-word detection sensitivity preset.",
    )
    parser.add_argument("--single-turn", action="store_true", help="Exit after one voice turn.")
    return parser.parse_args()


def _print_devices() -> int:
    try:
        devices = list_audio_devices()
    except AudioDeviceError as exc:
        print(f"Audio device query failed: {exc}", file=sys.stderr)
        return 1

    print("Available Audio Devices:")
    for dev in devices:
        tags = []
        if dev.is_default_input:
            tags.append("default-in")
        if dev.is_default_output:
            tags.append("default-out")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        print(
            f"  [{dev.index}] {dev.name} "
            f"(in={dev.max_input_channels}, out={dev.max_output_channels}, "
            f"sr={int(dev.default_samplerate)} Hz){tag_str}"
        )
    return 0


def _build_pipeline(cfg: AppConfig, args: argparse.Namespace) -> VoicePipeline:
    in_dev = _parse_device_arg(args.input_device)
    out_dev = _parse_device_arg(args.output_device)
    capture = MicrophoneAudioCapture(cfg, device=in_dev)
    player = SpeakerAudioPlayer(cfg, device=out_dev)

    def on_state(prev: PipelineState, curr: PipelineState, reason: str) -> None:
        if curr == PipelineState.SLEEPING:
            print(
                f"\n[SLEEPING] Standby active — say '{cfg.wakeword.wake_phrase}' (Ctrl+C to exit)",
                flush=True,
            )
        elif curr == PipelineState.WAKE_DETECTED:
            print(f"\n[WAKE_DETECTED] '{cfg.wakeword.wake_phrase}' detected!", flush=True)
        elif curr == PipelineState.RETURNING_TO_STANDBY:
            print("\n[STANDBY] Inactivity timeout reached; returning to standby...", flush=True)
        elif curr == PipelineState.LISTENING:
            if reason == "barge_in_follow_up":
                print("\n[LISTENING] Capturing interrupting speech...", flush=True)
            else:
                print("\n[LISTENING] Speak into your microphone (Ctrl+C to exit)...", flush=True)
        elif curr == PipelineState.PROCESSING:
            print("[PROCESSING] Transcribing speech...", flush=True)
        elif curr == PipelineState.GENERATING:
            print("[GENERATING] Assistant: ", end="", flush=True)
        elif curr == PipelineState.SPEAKING:
            print("\n[SPEAKING] Playing response...", flush=True)
        elif curr == PipelineState.INTERRUPTED:
            print("\n[INTERRUPTED] Barge-in detected; cancelling playback...", flush=True)
        elif curr == PipelineState.STOPPING:
            print("\n[STOPPING] Shutting down LocalVox cleanly...", flush=True)
        elif curr == PipelineState.ERROR:
            print(f"\n[ERROR] {reason}", file=sys.stderr, flush=True)

    def on_transcript(text: str, lang: str | None) -> None:
        print(f"\nYou ({lang or 'auto'}): {text}", flush=True)

    def on_token(token: str) -> None:
        print(token, end="", flush=True)

    def on_barge_in(info: BargeInCapture) -> None:
        if info.detect_latency_ms is not None:
            print(
                f"\n[BARGE-IN] Speech confirmed in {info.detect_latency_ms:.1f}ms",
                flush=True,
            )

    def on_turn_complete(m: TurnMetrics) -> None:
        interrupt_info = ""
        if m.interrupted:
            interrupt_info = (
                f" interrupted=True detect={m.interruption_detect_ms or 0.0:.1f}ms"
                f" cancel={m.playback_cancel_ms or 0.0:.1f}ms"
            )
        print(
            f"[METRICS turn={m.turn_id}] "
            f"vad={m.vad_ms:.1f}ms stt={m.stt_ms:.1f}ms "
            f"ttft={m.llm_ttft_ms:.1f}ms llm={m.llm_total_ms:.1f}ms "
            f"({m.llm_tokens_per_sec:.1f} tok/s) "
            f"tts_ttfa={m.tts_ttfa_ms:.1f}ms tts={m.tts_ms:.1f}ms(x{m.tts_chunks}) "
            f"first_audio={m.first_audio_latency_ms:.1f}ms "
            f"play={m.playback_ms:.1f}ms "
            f"mem={m.memories_retrieved}({m.memory_retrieval_ms:.1f}ms) "
            f"rss={m.memory_mb:.1f}MiB{interrupt_info}",
            flush=True,
        )

    def on_wake_word(event: WakeWordDetectionEvent) -> None:
        print(
            f"\n[WAKE] '{event.wake_phrase}' confidence={event.confidence:.3f} "
            f"(threshold={event.threshold:.2f}, infer={event.inference_ms:.2f}ms)",
            flush=True,
        )

    return VoicePipeline(
        cfg,
        capture=capture,
        player=player,
        state_machine=PipelineStateMachine(on_change=on_state),
        wake_word_mode=cfg.wakeword.enabled,
        language=args.language,
        tts_backend=args.tts_backend,
        on_transcript=on_transcript,
        on_token=on_token,
        on_turn_complete=on_turn_complete,
        on_barge_in=on_barge_in,
        on_wake_word=on_wake_word,
    )


def main() -> int:
    """Launch the offline LocalVox voice assistant."""
    args = parse_args()
    if args.list_devices:
        return _print_devices()

    cfg = get_config().model_copy(deep=True)
    if args.memory_db is not None:
        cfg.memory.db_path = Path(args.memory_db)
        cfg.paths.memory_db_file = Path(args.memory_db)
    if args.no_memory:
        cfg.memory.enabled = False
    if args.list_memories:
        mem = MemoryManager(cfg)
        try:
            records = mem.list_memories(limit=50)
            print(f"Stored Memories ({len(records)}) at {cfg.memory.db_path}:")
            for rec in records:
                conf = "user-confirmed" if rec.confirmed_by_user else "inferred"
                print(
                    f"  [{rec.id[:8]}] ({rec.category.value} | {conf} | "
                    f"imp={rec.importance:.2f}) {rec.subject}: {rec.content}"
                )
        finally:
            mem.shutdown(wait=False)
        return 0
    if args.barge_in_sensitivity is not None:
        cfg.pipeline.barge_in_sensitivity = args.barge_in_sensitivity
    if args.no_barge_in:
        cfg.pipeline.barge_in_enabled = False
    if args.no_streaming_tts:
        cfg.pipeline.streaming_tts_enabled = False
    if args.wake_word:
        cfg.wakeword.enabled = True
    if args.no_wake_word:
        cfg.wakeword.enabled = False
    if args.wake_sensitivity is not None:
        cfg.wakeword.sensitivity = args.wake_sensitivity
    pipeline = _build_pipeline(cfg, args)

    print("Warming up local models (Silero VAD, MLX Whisper, Ollama, TTS)...", flush=True)
    try:
        pipeline.warmup()
        pipeline.run(
            max_turns=1 if args.single_turn else None,
            warmup=False,
            wake_word_mode=cfg.wakeword.enabled,
        )
    except Exception as exc:
        print(f"Startup or runtime error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

