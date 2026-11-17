"""Reproducible isolated Phase 1 benchmark runner for VAD, STT, LLM, and TTS."""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from backend.config import AppConfig, get_config
from backend.eval.benchmark import (
    BenchmarkRecord,
    compute_rtf,
    detect_hardware_summary,
    format_markdown_table,
    get_process_memory_mb,
    save_benchmark_json,
)


def _prepare_test_audio(cfg: AppConfig) -> dict[str, Path]:
    """Create local English, silence, and synthetic Tamil test WAVs for Phase 1 checks."""
    audio_dir = cfg.paths.test_audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    from backend.tts.kokoro_tts import KokoroTtsEngine
    from backend.tts.piper_tts import PiperTtsEngine

    en_speech = audio_dir / "bench_en_speech.wav"
    if not en_speech.exists():
        KokoroTtsEngine(cfg).synthesize(
            "LocalVox is an offline voice study assistant running on Apple Silicon.",
            en_speech,
            language="en",
        )
    paths["en"] = en_speech

    en_vad = audio_dir / "bench_vad_speech.wav"
    if not en_vad.exists():
        from backend.stt.vad import load_audio_mono

        speech_16k, _ = load_audio_mono(en_speech, cfg.vad.sample_rate)
        silence = np.zeros(int(0.6 * cfg.vad.sample_rate), dtype=np.float32)
        sf.write(str(en_vad), np.concatenate([silence, speech_16k, silence]), cfg.vad.sample_rate)
    paths["vad_speech"] = en_vad

    silence_wav = audio_dir / "bench_vad_silence.wav"
    if not silence_wav.exists():
        sf.write(
            str(silence_wav),
            np.zeros(int(2.0 * cfg.vad.sample_rate), dtype=np.float32),
            cfg.vad.sample_rate,
        )
    paths["vad_silence"] = silence_wav

    ta_piper = audio_dir / "bench_ta_piper_synth.wav"
    if not ta_piper.exists():
        PiperTtsEngine(cfg).synthesize(
            "வணக்கம்! இன்று நாம் அறிவியல் மற்றும் கணிதம் படிப்போம்.",
            ta_piper,
            language="ta",
        )
    paths["ta_piper"] = ta_piper

    ta_vani = audio_dir / "bench_ta_vani_synth.wav"
    say_bin = shutil.which("say")
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ta_vani.exists() and say_bin and ffmpeg_bin:
        raw_aiff = audio_dir / "bench_ta_vani_raw.aiff"
        try:
            subprocess.run(
                [say_bin, "-v", "Vani", "-o", str(raw_aiff), "வணக்கம், இன்று நாம் அறிவியல் படிப்போம்."],
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
                    "16000",
                    "-ac",
                    "1",
                    str(ta_vani),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            data, _ = sf.read(str(ta_vani), dtype="float32")
            if data.size > 0 and float(np.max(np.abs(data))) > 1e-4:
                paths["ta_vani"] = ta_vani
            else:
                ta_vani.unlink(missing_ok=True)
        except Exception:
            ta_vani.unlink(missing_ok=True)
        finally:
            raw_aiff.unlink(missing_ok=True)
    elif ta_vani.exists():
        data, _ = sf.read(str(ta_vani), dtype="float32")
        if data.size > 0 and float(np.max(np.abs(data))) > 1e-4:
            paths["ta_vani"] = ta_vani
        else:
            ta_vani.unlink(missing_ok=True)

    return paths


def _run_vad_worker(cfg: AppConfig, audio_paths: dict[str, Path]) -> list[dict[str, Any]]:
    from backend.stt.vad import SileroVadDetector

    hardware = detect_hardware_summary()
    detector = SileroVadDetector(cfg)
    load_ms = detector.load()
    size_mb = detector.model_size_mb()

    # Warmup
    detector.detect(audio_paths["vad_speech"])

    times: list[float] = []
    last_res = None
    for _ in range(cfg.benchmark.benchmark_runs):
        last_res = detector.detect(audio_paths["vad_speech"])
        times.append(last_res.processing_time_ms)
    assert last_res is not None
    mean_ms = round(statistics.mean(times), 2)

    silence_res = detector.detect(audio_paths["vad_silence"])
    mem_mb = get_process_memory_mb()

    return [
        BenchmarkRecord(
            component="VAD",
            model_name=f"{cfg.vad.model_name} ({cfg.vad.backend})",
            language="en",
            measurement_type="measured",
            model_size_mb=size_mb,
            load_time_ms=round(load_ms, 2),
            inference_time_ms=mean_ms,
            memory_mb=mem_mb,
            audio_duration_s=last_res.audio_duration_s,
            rtf=compute_rtf(mean_ms, last_res.audio_duration_s),
            hardware=hardware,
            notes=(
                f"Speech detected={last_res.speech_detected} "
                f"({[s.model_dump() for s in last_res.segments]}); "
                f"silence false-positive={silence_res.speech_detected}"
            ),
        ).model_dump()
    ]


def _run_stt_worker(
    cfg: AppConfig,
    model_path: Path,
    audio_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    from backend.stt.whisper_stt import MlxWhisperStt

    hardware = detect_hardware_summary()
    stt = MlxWhisperStt(cfg)
    size_mb = stt.model_size_mb(model_path)
    load_ms = stt.load(model_path)

    # Warmup (compiles MLX Metal kernels)
    stt.transcribe(audio_paths["en"], language="en", model_path=model_path)

    en_times: list[float] = []
    en_res = None
    for _ in range(cfg.benchmark.benchmark_runs):
        en_res = stt.transcribe(audio_paths["en"], language="en", model_path=model_path)
        en_times.append(en_res.processing_time_ms)
    assert en_res is not None
    en_mean_ms = round(statistics.mean(en_times), 2)

    records = [
        BenchmarkRecord(
            component="STT",
            model_name=model_path.name,
            language="en",
            measurement_type="measured",
            model_size_mb=size_mb,
            load_time_ms=round(load_ms, 2),
            inference_time_ms=en_mean_ms,
            memory_mb=get_process_memory_mb(),
            audio_duration_s=en_res.audio_duration_s,
            rtf=compute_rtf(en_mean_ms, en_res.audio_duration_s),
            hardware=hardware,
            notes=f"Synthetic EN transcript: '{en_res.text}'",
        ).model_dump()
    ]

    ta_key = "ta_vani" if "ta_vani" in audio_paths else "ta_piper"
    ta_source = "macOS Vani TTS" if ta_key == "ta_vani" else "Piper ta_IN TTS"
    ta_times: list[float] = []
    ta_res = None
    for _ in range(cfg.benchmark.benchmark_runs):
        ta_res = stt.transcribe(audio_paths[ta_key], language="ta", model_path=model_path)
        ta_times.append(ta_res.processing_time_ms)
    assert ta_res is not None
    ta_mean_ms = round(statistics.mean(ta_times), 2)

    records.append(
        BenchmarkRecord(
            component="STT",
            model_name=model_path.name,
            language="ta (synthetic)",
            measurement_type="measured",
            model_size_mb=size_mb,
            load_time_ms=round(load_ms, 2),
            inference_time_ms=ta_mean_ms,
            memory_mb=get_process_memory_mb(),
            audio_duration_s=ta_res.audio_duration_s,
            rtf=compute_rtf(ta_mean_ms, ta_res.audio_duration_s),
            accuracy="untested (no human Tamil audio)",
            hardware=hardware,
            notes=f"Tested on {ta_source} only; output='{ta_res.text}'",
        ).model_dump()
    )
    return records


def _run_llm_worker(cfg: AppConfig) -> list[dict[str, Any]]:
    from backend.llm.ollama_client import OllamaClient

    hardware = detect_hardware_summary()
    client = OllamaClient(cfg)
    model_info = client.verify_model_exists(cfg.llm.model_name)
    size_bytes = model_info.get("size")
    size_mb = round(float(size_bytes) / (1024.0 * 1024.0), 2) if size_bytes else None

    # Warmup call so model is resident in memory before timed cases
    client.generate("Say ready in one word.", think=False, max_tokens=16)

    long_history = [
        {
            "role": "system",
            "content": cfg.llm.system_prompt,
        },
        {
            "role": "user",
            "content": (
                "We are revising cell biology: mitochondria produce ATP via oxidative "
                "phosphorylation, chloroplasts run the light-dependent reactions and Calvin cycle, "
                "ribosomes translate mRNA into proteins, and the Golgi apparatus packages "
                "vesicles. Keep these four organelle roles in context."
            ),
        },
        {
            "role": "assistant",
            "content": "Understood. I will reference those four organelles in our study session.",
        },
        {
            "role": "user",
            "content": (
                "Also remember our physics notes: Newton's first law covers inertia, the second "
                "law gives F equals m times a, and the third law describes equal and opposite "
                "reaction forces. Next, recall our chemistry notes on covalent vs ionic bonding."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Noted. We have cell organelles, Newton's laws, and chemical bonding loaded."
            ),
        },
        {
            "role": "user",
            "content": (
                "Using only our earlier biology and physics notes, state in two short sentences "
                "what mitochondria do and what Newton's second law states."
            ),
        },
    ]

    cases: list[tuple[str, bool, list[dict[str, str]]]] = [
        (
            "en",
            False,
            [
                {"role": "system", "content": cfg.llm.system_prompt},
                {"role": "user", "content": "Explain Newton's first law in 2 short sentences."},
            ],
        ),
        (
            "ta",
            False,
            [
                {"role": "system", "content": cfg.llm.system_prompt},
                {
                    "role": "user",
                    "content": "நியூட்டனின் முதல் விதியை எளிய தமிழில் 2 வாக்கியங்களில் விளக்கவும்.",
                },
            ],
        ),
        (
            "ta-en",
            False,
            [
                {"role": "system", "content": cfg.llm.system_prompt},
                {
                    "role": "user",
                    "content": "Binary search எப்படி work ஆகுது nu 2 short lines-ல explain பண்ணுங்க.",
                },
            ],
        ),
        ("en (long ctx)", False, long_history),
        (
            "en (think=True)",
            True,
            [
                {"role": "system", "content": cfg.llm.system_prompt},
                {"role": "user", "content": "Why is the sky blue? Explain briefly in 2 sentences."},
            ],
        ),
    ]
    records: list[dict[str, Any]] = []
    for lang, think_mode, messages in cases:
        res = client.chat(messages, think=think_mode)
        mem_mb = client.get_running_memory_mb(cfg.llm.model_name)
        records.append(
            BenchmarkRecord(
                component="LLM",
                model_name=f"{cfg.llm.model_name} (think={think_mode})",
                language=lang,
                measurement_type="measured",
                model_size_mb=size_mb,
                load_time_ms=res.load_duration_ms,
                inference_time_ms=res.total_time_ms,
                memory_mb=mem_mb,
                ttft_ms=res.ttft_ms,
                tokens_per_sec=res.tokens_per_sec,
                hardware=hardware,
                notes=f"in={res.prompt_tokens}, out={res.completion_tokens} tok",
            ).model_dump()
        )
    return records


def _run_kokoro_worker(cfg: AppConfig) -> list[dict[str, Any]]:
    from backend.tts.kokoro_tts import KokoroTtsEngine
    from backend.tts.types import UnsupportedLanguageError

    hardware = detect_hardware_summary()
    engine = KokoroTtsEngine(cfg)
    load_ms = engine.load()
    size_mb = engine.model_size_mb()
    out_wav = cfg.paths.test_audio_dir / "bench_kokoro_out.wav"
    text = "Photosynthesis converts light energy into chemical energy inside plant chloroplasts."

    # Warmup
    engine.synthesize(text, out_wav, language="en")

    times: list[float] = []
    res = None
    for _ in range(cfg.benchmark.benchmark_runs):
        res = engine.synthesize(text, out_wav, language="en")
        times.append(res.processing_time_ms)
    assert res is not None
    mean_ms = round(statistics.mean(times), 2)

    ta_rejected = False
    try:
        engine.synthesize("வணக்கம்", out_wav, language="ta")
    except UnsupportedLanguageError:
        ta_rejected = True

    return [
        BenchmarkRecord(
            component="TTS",
            model_name=f"kokoro ({cfg.tts.kokoro.default_voice})",
            language="en",
            measurement_type="measured",
            model_size_mb=size_mb,
            load_time_ms=round(load_ms, 2),
            inference_time_ms=mean_ms,
            memory_mb=get_process_memory_mb(),
            audio_duration_s=res.audio_duration_s,
            rtf=compute_rtf(mean_ms, res.audio_duration_s),
            hardware=hardware,
            notes=f"24 kHz WAV; Tamil rejected={ta_rejected} (unsupported in Kokoro v1.0)",
        ).model_dump()
    ]


def _run_piper_worker(cfg: AppConfig, language: str) -> list[dict[str, Any]]:
    from backend.tts.piper_tts import PiperTtsEngine

    hardware = detect_hardware_summary()
    engine = PiperTtsEngine(cfg)
    load_ms = engine.load(language)
    size_mb = engine.model_size_mb(language)
    out_wav = cfg.paths.test_audio_dir / f"bench_piper_{language}_out.wav"

    if language == "en":
        text = (
            "Photosynthesis converts light energy into chemical energy inside plant chloroplasts."
        )
        model_label = cfg.tts.piper.en_model_name
        note = "22.05 kHz WAV; official rhasspy/piper-voices"
    else:
        text = "ஒளிச்சேர்க்கை என்பது தாவரங்கள் சூரிய ஒளியைப் பயன்படுத்தி உணவு தயாரிக்கும் செயல்முறை."
        model_label = cfg.tts.piper.ta_model_name
        note = "22.05 kHz WAV; community tinisoft/piper-ta_IN-rasa_female-medium (CC-BY-4.0)"

    # Warmup
    engine.synthesize(text, out_wav, language=language)

    times: list[float] = []
    res = None
    for _ in range(cfg.benchmark.benchmark_runs):
        res = engine.synthesize(text, out_wav, language=language)
        times.append(res.processing_time_ms)
    assert res is not None
    mean_ms = round(statistics.mean(times), 2)

    return [
        BenchmarkRecord(
            component="TTS",
            model_name=f"piper ({model_label})",
            language=language,
            measurement_type="measured",
            model_size_mb=size_mb,
            load_time_ms=round(load_ms, 2),
            inference_time_ms=mean_ms,
            memory_mb=get_process_memory_mb(),
            audio_duration_s=res.audio_duration_s,
            rtf=compute_rtf(mean_ms, res.audio_duration_s),
            hardware=hardware,
            notes=note,
        ).model_dump()
    ]


def _invoke_worker(worker: str, extra: str = "") -> list[BenchmarkRecord]:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        result_file = Path(tf.name)
    try:
        cmd = [sys.executable, __file__, "--worker", worker, "--result-file", str(result_file)]
        if extra:
            cmd.extend(["--extra", extra])
        subprocess.run(cmd, check=True)
        raw = json.loads(result_file.read_text(encoding="utf-8"))
        return [BenchmarkRecord.model_validate(item) for item in raw]
    finally:
        result_file.unlink(missing_ok=True)


def main() -> int:
    """Run all isolated component benchmarks and save JSON results."""
    parser = argparse.ArgumentParser(description="Run LocalVox Phase 1 isolated benchmarks.")
    parser.add_argument("--worker", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--extra", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = get_config()
    if args.worker is not None:
        audio_paths = _prepare_test_audio(cfg)
        if args.worker == "vad":
            payload = _run_vad_worker(cfg, audio_paths)
        elif args.worker == "stt":
            payload = _run_stt_worker(cfg, Path(args.extra), audio_paths)
        elif args.worker == "llm":
            payload = _run_llm_worker(cfg)
        elif args.worker == "kokoro":
            payload = _run_kokoro_worker(cfg)
        elif args.worker == "piper":
            payload = _run_piper_worker(cfg, args.extra)
        else:
            raise ValueError(f"Unknown worker: {args.worker}")
        serialized = json.dumps(payload, ensure_ascii=False)
        if args.result_file is not None:
            args.result_file.write_text(serialized, encoding="utf-8")
        else:
            print(serialized)
        return 0

    _prepare_test_audio(cfg)
    all_records: list[BenchmarkRecord] = []
    all_records.extend(_invoke_worker("vad"))
    for stt_path in cfg.stt.candidate_models:
        if stt_path.exists():
            all_records.extend(_invoke_worker("stt", str(stt_path)))
    all_records.extend(_invoke_worker("llm"))
    all_records.extend(_invoke_worker("kokoro"))
    all_records.extend(_invoke_worker("piper", "en"))
    all_records.extend(_invoke_worker("piper", "ta"))

    out_json = save_benchmark_json(all_records, cfg.benchmark.output_json_file)
    print(f"Saved machine-readable benchmarks to {out_json}\n")
    print(format_markdown_table(all_records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
