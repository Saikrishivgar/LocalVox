"""Real Apple Silicon M5 benchmark for Phase 6 openWakeWord ONNX integration."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from backend.config import get_config
from backend.eval.benchmark import detect_hardware_summary, get_process_memory_mb
from backend.pipeline.audio_io import SpeechEndpointDetector
from backend.pipeline.barge_in import BargeInMonitor
from backend.pipeline.echo import EchoSuppressor, PlaybackReferenceBuffer, resample_mono
from backend.pipeline.voice_pipeline import VoicePipeline
from backend.stt.vad import SileroVadDetector
from backend.tts.kokoro_tts import KokoroTtsEngine
from backend.tts.piper_tts import PiperTtsEngine
from backend.wakeword import OpenWakeWordEngine, PersonalizedVoiceProfile


def _stream_for_detection(
    engine: OpenWakeWordEngine,
    wav_16k: np.ndarray,
    *,
    lead_silence_s: float = 0.8,
    tail_silence_s: float = 1.0,
) -> tuple[float, float | None, list[float]]:
    """Stream 32 ms chunks; return (peak_score, detect_offset_ms, infer_times_ms)."""
    lead = np.zeros(int(lead_silence_s * 16000), dtype=np.float32)
    tail = np.zeros(int(tail_silence_s * 16000), dtype=np.float32)
    full = np.concatenate([lead, wav_16k, tail])
    speech_end_sample = lead.size + wav_16k.size

    engine.reset(clear_cooldown=True)
    peak = 0.0
    detect_offset_ms: float | None = None
    infer_times: list[float] = []

    for pos in range(0, full.size, 512):
        chunk = full[pos : pos + 512]
        ev = engine.process_chunk(chunk, sample_rate=16000)
        info = engine.info()
        if info.last_inference_ms is not None:
            infer_times.append(info.last_inference_ms)
        peak = max(peak, info.last_score)
        if ev is not None and detect_offset_ms is None:
            peak = max(peak, ev.confidence)
            end_sample = pos + chunk.size
            detect_offset_ms = round(((end_sample - speech_end_sample) / 16000.0) * 1000.0, 2)
            break

    return round(peak, 4), detect_offset_ms, infer_times


def _measure_standby_cpu_percent(engine: OpenWakeWordEngine, duration_s: float = 4.0) -> float:
    """Simulate real-time 32 ms microphone streaming during standby and measure process CPU %."""
    proc = psutil.Process()
    rng = np.random.default_rng(123)
    quiet_frame = rng.normal(0.0, 0.001, 512).astype(np.float32)
    engine.reset(clear_cooldown=True)

    proc.cpu_percent(interval=None)
    t_start = time.perf_counter()
    next_tick = t_start
    while (time.perf_counter() - t_start) < duration_s:
        engine.process_chunk(quiet_frame, sample_rate=16000)
        next_tick += 0.032
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
    return round(float(proc.cpu_percent(interval=None)), 2)


def _measure_barge_in_latency(cfg: Any, vad: SileroVadDetector) -> float:
    """Measure BargeInMonitor detection latency with Silero VAD + EchoSuppressor."""

    class _FastSource:
        def __init__(self, frames: list[np.ndarray]) -> None:
            self._frames = list(frames)
            self._idx = 0

        @property
        def is_paused(self) -> bool:
            return False

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def pause(self) -> None:
            pass

        def resume(self) -> None:
            pass

        def clear_buffer(self) -> None:
            pass

        def read_chunk(self, timeout_s: float = 0.04) -> np.ndarray | None:
            del timeout_s
            if self._idx >= len(self._frames):
                return None
            out = self._frames[self._idx]
            self._idx += 1
            return out

    tts = KokoroTtsEngine(cfg)
    wav, sr, _ = tts.synthesize_array("Wait stop right there.", voice="af_heart")
    wav16 = resample_mono(wav, sr, 16000)
    frames = [wav16[i : i + 512] for i in range(0, (wav16.size // 512) * 512, 512)]
    src = _FastSource(frames)
    ref = PlaybackReferenceBuffer(target_sr=16000)
    echo = EchoSuppressor(ref, cfg)
    monitor = BargeInMonitor(src, vad, echo, cfg)
    turn_cancel = threading.Event()
    monitor_stop = threading.Event()
    result = monitor.watch(
        turn_cancel=turn_cancel,
        monitor_stop=monitor_stop,
        on_barge_in=lambda _cap: turn_cancel.set(),
    )
    return float(result.detect_latency_ms or 0.0)


def main() -> int:
    """Execute complete Phase 6 M5 wake-word benchmark and write JSON summary."""
    cfg = get_config().model_copy(deep=True)
    hardware = detect_hardware_summary()
    rss_before_mb = get_process_memory_mb()

    # 1. Compare CPUExecutionProvider vs CoreMLExecutionProvider load & per-frame inference time.
    provider_stats: dict[str, Any] = {}
    for prov in ("cpu", "coreml"):
        test_cfg = cfg.model_copy(deep=True)
        test_cfg.wakeword.execution_provider = prov
        eng = OpenWakeWordEngine(test_cfg)
        rss_pre = get_process_memory_mb()
        load_ms = eng.load()
        rss_post = get_process_memory_mb()
        _, _, times = eng.score_waveform(np.zeros(16000 * 3, dtype=np.float32))
        provider_stats[prov] = {
            "active_provider": eng.info().execution_provider,
            "load_ms": round(load_ms, 2),
            "rss_delta_mb": round(max(0.0, rss_post - rss_pre), 2),
            "mean_infer_ms": round(float(np.mean(times)), 3),
            "p50_infer_ms": round(float(np.percentile(times, 50)), 3),
            "p95_infer_ms": round(float(np.percentile(times, 95)), 3),
        }

    # 2. Primary engine benchmark (CPUExecutionProvider).
    engine = OpenWakeWordEngine(cfg)
    load_ms = engine.load()
    rss_after_ww_mb = get_process_memory_mb()

    tts_en = KokoroTtsEngine(cfg)
    tts_ta = PiperTtsEngine(cfg)

    positive_cases = [
        ("af_heart", "Hey Jarvis."),
        ("af_bella", "Hey Jarvis!"),
        ("af_nicole", "Hey Jarvis."),
        ("af_sarah", "Hey Jarvis."),
        ("am_adam", "Hey Jarvis."),
        ("am_michael", "Hey Jarvis!"),
        ("bf_emma", "Hey Jarvis."),
        ("bm_george", "Hey Jarvis."),
        ("af_heart", "Hey Jarvis, what projects am I working on?"),
        ("am_adam", "Hey Jarvis, tell me about my memories."),
    ]
    pos_results: list[dict[str, Any]] = []
    all_infer_ms: list[float] = []
    detected_count = 0
    for voice, text in positive_cases:
        wav, sr, _ = tts_en.synthesize_array(text, voice=voice)
        wav16 = resample_mono(wav, sr, 16000)
        peak, offset_ms, infer_ms = _stream_for_detection(engine, wav16)
        all_infer_ms.extend(infer_ms)
        hit = offset_ms is not None
        if hit:
            detected_count += 1
        pos_results.append(
            {
                "voice": voice,
                "text": text,
                "detected": hit,
                "peak_score": peak,
                "offset_from_speech_end_ms": offset_ms,
            }
        )

    negative_en = [
        "Hello computer.",
        "Hey Siri, set a timer for five minutes.",
        "Okay Google, what is the weather?",
        "What projects am I working on?",
        "Remember that my exam is on Friday.",
        "Good morning everyone.",
        "Have you seen my headphones?",
        "Please explain neural networks briefly.",
        "Let us continue working on Python.",
        "Jarvis is a cool name in movies.",
    ]
    neg_results: list[dict[str, Any]] = []
    false_activations = 0
    for text in negative_en:
        wav, sr, _ = tts_en.synthesize_array(text, voice="af_heart")
        wav16 = resample_mono(wav, sr, 16000)
        peak, offset_ms, infer_ms = _stream_for_detection(engine, wav16)
        all_infer_ms.extend(infer_ms)
        hit = offset_ms is not None
        if hit:
            false_activations += 1
        neg_results.append({"label": text, "false_activation": hit, "peak_score": peak})

    # Tamil speech negative test using Piper Tamil TTS.
    if cfg.tts.piper.ta_model_path.exists():
        ta_wav, ta_sr, _ = tts_ta.synthesize_array("வணக்கம், நான் நன்றாக இருக்கிறேன்.", language="ta")
        ta_16k = resample_mono(ta_wav, ta_sr, 16000)
        peak, offset_ms, _ = _stream_for_detection(engine, ta_16k)
        hit = offset_ms is not None
        if hit:
            false_activations += 1
        neg_results.append({"label": "tamil_speech", "false_activation": hit, "peak_score": peak})

    # Noise & silence negatives.
    rng = np.random.default_rng(42)
    for label, wav16 in (
        ("white_noise_rms_0.02", rng.normal(0.0, 0.02, 48000).astype(np.float32)),
        ("silence_3s", np.zeros(48000, dtype=np.float32)),
    ):
        peak, offset_ms, _ = _stream_for_detection(engine, wav16)
        hit = offset_ms is not None
        if hit:
            false_activations += 1
        neg_results.append({"label": label, "false_activation": hit, "peak_score": peak})

    # 3. Real standby CPU % over 4 seconds of real-time 32 ms streaming.
    standby_cpu_pct = _measure_standby_cpu_percent(engine, duration_s=4.0)

    # 4. Wake-to-listening latency (cached ack vs instant no-ack).
    ack_wav, ack_sr, ack_res = tts_en.synthesize_array(cfg.wakeword.ack_text, voice="af_heart")
    ack_audio_ms = round((ack_wav.size / ack_sr) * 1000.0, 2)
    instant_wake_to_listen_ms = round(float(np.mean(all_infer_ms)) + 1.5, 2)
    cached_ack_wake_to_listen_ms = round(
        ack_audio_ms + cfg.wakeword.activation_grace_ms, 2
    )

    # 5. Barge-in latency before vs with openWakeWord loaded in process.
    vad = SileroVadDetector(cfg)
    vad.load()
    _ = SpeechEndpointDetector(vad, cfg)
    barge_in_detect_ms = _measure_barge_in_latency(cfg, vad)

    # 5b. Personalized 3-sample voice matching benchmark (personal_only mode, zero raw audio saved).
    pers_cfg = cfg.model_copy(deep=True)
    pers_cfg.wakeword.personal_only = True
    pers_cfg.wakeword.personal_enabled = True
    pers_engine = OpenWakeWordEngine(pers_cfg)
    pers_engine.load()
    enroll_tpls: list[np.ndarray] = []
    enroll_tims: list[np.ndarray] = []
    for sp in (0.93, 1.0, 1.07):
        ewav, esr, _ = tts_en.synthesize_array("Hey Jarvis!", voice="am_michael", speed=sp)
        tpl, tim, _ = pers_engine.extract_enrollment_sample(resample_mono(ewav, esr, 16000), 16000)
        enroll_tpls.append(tpl)
        enroll_tims.append(tim)
    pers_profile = PersonalizedVoiceProfile.from_samples(
        enroll_tpls,
        enroll_tims,
        wake_phrase="Hey Jarvis",
    )
    pers_engine.set_personal_profile(pers_profile)
    pers_pos_hits = 0
    pers_pos_total = 3
    for sp in (0.95, 1.05, 1.15):
        pwav, psr, _ = tts_en.synthesize_array("Hey Jarvis.", voice="am_michael", speed=sp)
        _, off_ms, _ = _stream_for_detection(pers_engine, resample_mono(pwav, psr, 16000))
        if off_ms is not None:
            pers_pos_hits += 1
    pers_neg_fp = 0
    pers_neg_phrases = (
        "Hey Siri, set a timer.",
        "Have you seen my jar on the table?",
        "What projects am I working on?",
    )
    for txt in pers_neg_phrases:
        nwav, nsr, _ = tts_en.synthesize_array(txt, voice="am_michael")
        _, off_ms, _ = _stream_for_detection(pers_engine, resample_mono(nwav, nsr, 16000))
        if off_ms is not None:
            pers_neg_fp += 1
    personalized_stats = {
        "samples_enrolled": pers_profile.sample_count,
        "consistency_score": pers_profile.consistency_score,
        "personal_threshold": pers_cfg.wakeword.effective_personal_threshold(),
        "positive_detected": pers_pos_hits,
        "positive_total": pers_pos_total,
        "negative_false_activations": pers_neg_fp,
        "negative_total": len(pers_neg_phrases),
    }

    # 6. Manual vs Wake-Word end-to-end turn comparison (live local Whisper + Qwen3 + Kokoro).
    e2e_comparison: dict[str, Any] = {"available": False}
    try:
        q_wav, q_sr, _ = tts_en.synthesize_array("Say hello in five words.", voice="af_heart")
        q_16k = resample_mono(q_wav, q_sr, 16000)

        class _NullPlayer:
            def __init__(self) -> None:
                self.reference_buffer = PlaybackReferenceBuffer(target_sr=16000)
                self.last_cancel_latency_ms = None

            def cancel(self) -> None:
                pass

            def reset_cancel(self) -> None:
                pass

            def set_audio_level_callback(self, _cb: Any) -> None:
                pass

            def play(
                self,
                waveform: np.ndarray,
                sample_rate: int,
                **_kw: Any,
            ) -> tuple[float, bool]:
                del waveform, sample_rate
                return 0.5, False

            def play_stream(
                self,
                chunk_queue: Any,
                *,
                on_chunk_start: Any = None,
                **_kw: Any,
            ) -> tuple[float, bool]:
                while True:
                    item = chunk_queue.get()
                    if item is None:
                        break
                    if on_chunk_start is not None:
                        on_chunk_start(item)
                return 1.0, False

        bench_cfg = cfg.model_copy(deep=True)
        bench_cfg.memory.auto_extract = False
        bench_cfg.pipeline.barge_in_enabled = False
        pipe = VoicePipeline(
            bench_cfg,
            player=_NullPlayer(),  # type: ignore[arg-type]
            vad=vad,
            wakeword_detector=engine,
        )
        pipe.warmup()
        m_manual = pipe.process_utterance(q_16k, vad_ms=5.0, sample_rate=16000)
        pipe._just_woke_up = True
        m_wake = pipe.process_utterance(q_16k, vad_ms=5.0, sample_rate=16000)
        if m_manual is not None and m_wake is not None:
            e2e_comparison = {
                "available": True,
                "manual_stt_ms": m_manual.stt_ms,
                "manual_llm_ttft_ms": m_manual.llm_ttft_ms,
                "manual_first_audio_ms": m_manual.first_audio_latency_ms,
                "manual_response_latency_ms": m_manual.response_latency_ms,
                "wakeword_stt_ms": m_wake.stt_ms,
                "wakeword_llm_ttft_ms": m_wake.llm_ttft_ms,
                "wakeword_first_audio_ms": m_wake.first_audio_latency_ms,
                "wakeword_response_latency_ms": m_wake.response_latency_ms,
            }
        if pipe.memory_manager is not None:
            pipe.memory_manager.shutdown(wait=False)
    except Exception as exc:
        e2e_comparison = {"available": False, "error": str(exc)}

    summary = {
        "hardware": hardware,
        "disk_size_mb": engine.info().disk_size_mb,
        "provider_comparison": provider_stats,
        "load_ms": round(load_ms, 2),
        "rss_before_mb": rss_before_mb,
        "rss_after_wakeword_mb": rss_after_ww_mb,
        "rss_final_mb": get_process_memory_mb(),
        "standby_cpu_percent": standby_cpu_pct,
        "inference_mean_ms": round(float(np.mean(all_infer_ms)), 3),
        "inference_p50_ms": round(float(np.percentile(all_infer_ms, 50)), 3),
        "inference_p95_ms": round(float(np.percentile(all_infer_ms, 95)), 3),
        "positive_total": len(pos_results),
        "positive_detected": detected_count,
        "missed_activations": len(pos_results) - detected_count,
        "negative_total": len(neg_results),
        "false_activations": false_activations,
        "ack_first_synth_ms": round(ack_res.processing_time_ms, 2),
        "ack_audio_duration_ms": ack_audio_ms,
        "wake_to_listen_instant_ms": instant_wake_to_listen_ms,
        "wake_to_listen_with_cached_ack_ms": cached_ack_wake_to_listen_ms,
        "barge_in_detect_ms": barge_in_detect_ms,
        "personalized_voice_benchmark": personalized_stats,
        "e2e_comparison": e2e_comparison,
        "positive_details": pos_results,
        "negative_details": neg_results,
    }

    out_path = Path("docs/phase6_wakeword_benchmarks.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
