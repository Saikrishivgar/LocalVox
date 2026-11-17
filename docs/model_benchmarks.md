# LocalVox Phase 1 — Model Evaluation & Benchmarks

This document records the measured local performance, disk footprint, memory consumption, language compatibility, and limitations of the four standalone AI components evaluated in Phase 1.

> **Privacy & Offline Guarantee**: All measurements were executed locally on-device with zero cloud APIs. Offline execution was verified with `HF_HUB_OFFLINE=1`.

---

## 1. Test Hardware & Environment

- **Device**: Apple MacBook Pro (`Apple M5`)
- **Memory**: 24 GB Unified Memory (`25,769,803,776` bytes)
- **Storage**: 926 GiB APFS volume (`193 GiB` free before Phase 1 downloads)
- **OS**: macOS (`darwin` ARM64)
- **Python**: `Python 3.11.18` managed via `uv` (`0.13.8`)
- **System Tools**: `Ollama 0.19.0`, `ffmpeg 8.1.2_1`
- **Key Libraries**: `silero-vad==6.2.3`, `onnxruntime==1.32.0`, `mlx==0.32.4`, `mlx-whisper==0.4.3`, `torch==2.15.0`, `kokoro-onnx==0.6.1`, `espeakng-loader==0.2.4`, `piper-tts==1.8.0`, `httpx==0.28.1`, `psutil==7.2.2`

---

## 2. Measurement Methodology

- **Isolation**: Each component (`VAD`, each `STT` candidate, `LLM`, `Kokoro`, `Piper EN`, `Piper TA`) was benchmarked in a separate subprocess via `scripts/run_benchmarks.py` so process resident memory (`psutil.Process().memory_info().rss`) was not polluted by previously loaded engines.
- **Warmup & Averaging**: For VAD, STT, and TTS, each worker ran 1 warmup pass (excluded from steady-state mean, capturing cold shader/graph compilation separately where relevant) followed by `3` timed runs (`benchmark.benchmark_runs = 3`).
- **RTF Definition**: `RTF = processing_time_s / audio_duration_s`. Values `< 1.0` are faster than real time.
- **MLX Metal Memory**: Because `psutil` RSS reflects host-mapped pages and does not fully capture Apple Silicon unified GPU buffer allocations, MLX Whisper models were additionally profiled with `mlx.core.get_active_memory()` and `mlx.core.get_peak_memory()`.
- **Ollama Memory & Latency**: Disk size was queried from `/api/tags`, resident unified memory from `/api/ps`, Time-To-First-Token (`TTFT`) from the first non-empty streaming chunk, and generation throughput (`tok/s`) from Ollama's `eval_count / eval_duration`.
- **Machine-Readable Output**: Raw benchmark output is stored in [`docs/model_benchmarks.json`](model_benchmarks.json).

---

## 3. Models Evaluated & Storage Requirements

| Component | Engine / Runtime | Model Identifier | License | Local Location | Disk Size |
| --- | --- | --- | --- | --- | ---: |
| VAD | Silero VAD (ONNX) | `silero-vad` (`silero_vad.onnx`) | MIT | Bundled in `silero-vad` wheel | 2.22 MiB |
| STT (Small) | `mlx-whisper` (Metal) | `mlx-community/whisper-small-mlx` | MIT | `data/models/stt/whisper-small-mlx` | 459.01 MiB |
| STT (8-bit Turbo) | `mlx-whisper` (Metal) | `mlx-community/whisper-large-v3-turbo-8bit` | MIT | `data/models/stt/whisper-large-v3-turbo-8bit` | 824.43 MiB |
| STT (FP16 Turbo) | `mlx-whisper` (Metal) | `mlx-community/whisper-large-v3-turbo` | MIT | `data/models/stt/whisper-large-v3-turbo` | 1,539.21 MiB |
| LLM | Ollama (`0.19.0`) | `qwen3:8b` | Apache-2.0 | Local Ollama store | 4,983.32 MiB |
| TTS (Candidate 1) | `kokoro-onnx` | `kokoro-v1.0.onnx` + `voices-v1.0.bin` | Apache-2.0 | `data/models/tts/kokoro/` | 337.36 MiB |
| TTS (Candidate 2 EN) | `piper-tts` (ONNX) | `en_US-lessac-medium` (`rhasspy/piper-voices`) | MIT (code) / voice card | `data/models/tts/piper/en_US-lessac-medium/` | 60.27 MiB |
| TTS (Candidate 2 TA) | `piper-tts` (ONNX) | `ta_IN-rasa_female-medium` (`tinisoft`) | CC-BY-4.0 | `data/models/tts/piper/ta_IN-rasa_female-medium/` | 60.58 MiB |

---

## 4. Measured Results vs. Untested Capabilities

### A. Summary Table (Measured Values from `docs/model_benchmarks.json`)

| Component | Model | Type | Lang | Size (MiB) | Load (ms) | Inference (ms) | Memory (MiB) | RTF | Accuracy / Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| VAD | `silero-vad (onnx)` | measured | en | 2.22 | 12.9 | 13.4 | 313.2 | 0.0023 | Speech detected=True (`[0.6, 5.3]`); silence false-positive=False |
| STT | `whisper-small-mlx` | measured | en | 459.01 | 178.2 | 152.1 | 715.0 | 0.0327 | Synthetic EN: `'Local Vox is an offline voice study assistant running on Apple Silicon.'` |
| STT | `whisper-small-mlx` | measured | ta (synthetic) | 459.01 | 178.2 | 222.7 | 715.0 | 0.0792 | `untested` on human audio; Vani TTS output misread `அறிவியல்` as `அறிவில்` |
| STT | `whisper-large-v3-turbo-8bit` | measured | en | 824.43 | 141.0 | 383.7 | 1,070.6 | 0.0825 | Synthetic EN: `'LocalVox is an offline voice study assistant running on Apple Silicon.'` |
| STT | `whisper-large-v3-turbo-8bit` | measured | ta (synthetic) | 824.43 | 141.0 | 409.5 | 1,071.1 | 0.1457 | `untested` on human audio; Vani TTS exact output `'வணக்கம் இன்று நாம் அறிவியல் படிப்போம்'` |
| STT | `whisper-large-v3-turbo` | measured | en | 1,539.21 | 297.8 | 372.6 | 543.8* | 0.0801 | Synthetic EN: `'LocalVox is an offline voice study assistant running on Apple Silicon.'` |
| STT | `whisper-large-v3-turbo` | measured | ta (synthetic) | 1,539.21 | 297.8 | 436.1 | 544.7* | 0.1551 | `untested` on human audio; Vani TTS exact output `'வணக்கம் இன்று நாம் அறிவியல் படிப்போம்'` |
| LLM | `qwen3:8b (think=False)` | measured | en | 4,983.32 | 36.7 | 1,818.3 | 6,249.6 | n/a | 24.6 tok/s, TTFT 187.6 ms; `in=52, out=40 tok` |
| LLM | `qwen3:8b (think=False)` | measured | ta | 4,983.32 | 37.5 | 7,747.6 | 6,249.6 | n/a | 24.1 tok/s, TTFT 491.1 ms; `in=111, out=175 tok` |
| LLM | `qwen3:8b (think=False)` | measured | ta-en | 4,983.32 | 36.4 | 5,379.1 | 6,249.6 | n/a | 24.0 tok/s, TTFT 368.3 ms; `in=75, out=120 tok` |
| LLM | `qwen3:8b (think=False)` | measured | en (long ctx) | 4,983.32 | 37.1 | 1,886.8 | 6,249.6 | n/a | 24.9 tok/s, TTFT 959.0 ms; `in=224, out=23 tok` |
| LLM | `qwen3:8b (think=True)` | measured | en (`think=True`) | 4,983.32 | 36.1 | 9,676.8 | 6,249.6 | n/a | 23.8 tok/s, TTFT 269.9 ms; `in=47, out=225 tok` |
| TTS | `kokoro (af_heart)` | measured | en | 337.36 | 197.0 | 966.5 | 724.8 | 0.1703 | 24 kHz WAV (5.675 s); Tamil explicitly rejected (`UnsupportedLanguageError`) |
| TTS | `piper (en_US-lessac-medium)` | measured | en | 60.27 | 100.5 | 85.6 | 279.9 | 0.0168 | 22.05 kHz WAV (5.085 s); official `rhasspy/piper-voices` |
| TTS | `piper (ta_IN-rasa_female-medium)` | measured | ta | 60.58 | 144.2 | 103.0 | 298.0 | 0.0201 | 22.05 kHz WAV (5.132 s); community AI4Bharat Rasa voice |

*\*POSIX process RSS omits Metal-allocated unified GPU buffers; see MLX Metal memory measurements below.*

### B. MLX Whisper Metal Memory Profile (`mlx.core`)

Measured directly with `mx.get_active_memory()` and `mx.get_peak_memory()` during Tamil inference (`2.811 s` audio):

| Model | Disk Size (MiB) | MLX Active Memory (MiB) | MLX Peak Memory (MiB) | Cold Compile + First Run (ms) | Warm Inference Mean (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `whisper-small-mlx` | 459.01 | 461.6 | 1,166.8 | 2,258.5 | 152.1 (EN) / 222.7 (TA) |
| `whisper-large-v3-turbo-8bit` | 824.43 | 833.7 | 1,463.6 | 825.2 | 383.7 (EN) / 409.5 (TA) |
| `whisper-large-v3-turbo` | 1,539.21 | 1,543.5 | 2,372.9 | 791.5 | 372.6 (EN) / 436.1 (TA) |

> **Compatibility note**: `mlx-community/whisper-large-v3-turbo-8bit` distributes weights as `model.safetensors` rather than `weights.safetensors`, which stock `mlx_whisper.load_models.load_model()` does not check. [`backend/stt/whisper_stt.py`](../backend/stt/whisper_stt.py) supports `weights.safetensors`, `weights.npz`, and `model.safetensors`.

### C. Untested Capabilities (Explicitly Flagged)

| Capability | Status | Reason |
| --- | --- | --- |
| Human Tamil STT accuracy (WER/CER) | **Incomplete / Untested** | No genuine native Tamil human speech recordings exist in the project workspace; only synthetic TTS audio (`macOS Vani`, `Piper ta_IN`) was tested. |
| Tamil-English code-mixed STT accuracy | **Incomplete / Untested** | No real human code-mixed recordings were available during Phase 1. |
| Long-context LLM (>2K tokens) | **Partially Tested** | Tested multi-turn recall (`142` and `224` input tokens, `num_ctx=8192`); multi-thousand-token RAG context latency will be measured in Phase 2/3. |
| Kokoro Tamil TTS | **Unsupported** | Kokoro v1.0 supports EN, HI, JA, ZH, ES, FR, IT, PT — **not Tamil**. |

---

## 5. Language Compatibility Matrix

| Component | English (`en`) | Tamil (`ta`) | Tamil-English Mixed | Notes |
| --- | --- | --- | --- | --- |
| Silero VAD | Verified (`RTF 0.0023`) | Language-agnostic (energy/speech) | Language-agnostic | Detects speech onset/offset accurately; `0` false positives on 2 s silence. |
| MLX Whisper (`large-v3-turbo-8bit`) | Verified on synthetic audio | Pipeline verified on synthetic TTS; **human audio untested** | **Untested** | Outputs accurate Tamil script on `Vani` TTS (`வணக்கம் இன்று நாம் அறிவியல் படிப்போம்`). |
| Ollama (`qwen3:8b`) | Verified (`24.6 tok/s`) | Verified (`24.1 tok/s`, grammatical Tamil) | Understood prompt (`24.0 tok/s`) | With `think=False`, replies fast (`TTFT 188–491 ms`); on Tanglish prompts it often answered in English + Markdown/emojis unless guided by a voice prompt. |
| Kokoro (`kokoro-onnx`) | Verified (`24 kHz`, high quality) | **Unsupported** | **Unsupported** | Raises `UnsupportedLanguageError` when `ta` is requested. |
| Piper (`piper-tts`) | Verified (`en_US-lessac-medium`) | Verified via community `ta_IN-rasa_female-medium` | Limited (eSpeak `ta` phonemizer) | Official `rhasspy/piper-voices` (`53` languages) has **no Tamil (`ta`)**; community CC-BY-4.0 model works offline at `RTF 0.0201`. |

---

## 6. Procedure to Complete Genuine Human Tamil & Code-Mixed STT Validation

When native Tamil and code-mixed `.wav` recordings are added under `data/test_audio/`:

1. Record 3–5 short clips (5–15 seconds each, 16 kHz mono WAV or M4A):
   - `data/test_audio/human_en.wav` — clean English study question
   - `data/test_audio/human_ta.wav` — clean native Tamil study question
   - `data/test_audio/human_ta_en.wav` — Tamil-English code-mixed study question (e.g., `"Newton's second law எப்படி apply பண்றது?"`)
2. Run MLX Whisper against each recording:
   ```bash
   uv run python scripts/stt_test.py data/test_audio/human_en.wav --language en
   uv run python scripts/stt_test.py data/test_audio/human_ta.wav --language ta
   uv run python scripts/stt_test.py data/test_audio/human_ta_en.wav
   ```
3. Compare `whisper-large-v3-turbo-8bit` vs `whisper-large-v3-turbo` (fp16) via `--model` and record real WER/CER before finalizing Phase 2 STT tuning.

---

## 7. Combined Memory Budget on 24 GB Unified Memory

| Resident Component | Estimated Unified Memory |
| --- | ---: |
| Ollama `qwen3:8b` (`num_ctx=8192`) | ~6,250 MiB (~6.1 GiB) |
| MLX Whisper `whisper-large-v3-turbo-8bit` (peak) | ~1,464 MiB (~1.4 GiB) |
| Kokoro ONNX (`kokoro-v1.0.onnx`) | ~725 MiB (~0.7 GiB) |
| Piper ONNX (`ta_IN-rasa_female-medium`) | ~298 MiB (~0.3 GiB) |
| Silero VAD ONNX (`silero_vad.onnx`) | ~313 MiB (~0.3 GiB) |
| **Combined Peak (all resident simultaneously)** | **~9.05 GiB / 24.0 GiB (~15 GiB free headroom)** |

---

## 8. Limitations & Phase 2 Recommendations

1. **Recommended VAD**: `silero-vad` (`onnx` backend, `2.22 MiB`, `13.4 ms` per ~5.8 s clip). Ready for Phase 2 streaming frame-by-frame integration.
2. **Recommended STT**: `mlx-community/whisper-large-v3-turbo-8bit` (`824.43 MiB`, `~384–410 ms` warm latency, identical synthetic Tamil & English output to fp16 with ~900 MiB less peak RAM). Needs real human Tamil and Tanglish recordings to validate code-mixed accuracy and decide whether `language` auto-detection or explicit switching is required.
3. **Recommended LLM**: `qwen3:8b` via local Ollama with `think=False` for low-latency voice responses (`TTFT ~188–491 ms`, `~24 tok/s`). In Phase 2, tighten the system prompt to forbid Markdown/emojis and match the user's Tamil/English register concisely.
4. **Recommended TTS Architecture**: Dual-engine routing in Phase 2 — **Kokoro ONNX** (`af_heart`, `24 kHz`, `RTF 0.1703`) for natural English speech, and **Piper ONNX** (`ta_IN-rasa_female-medium`, `22.05 kHz`, `RTF 0.0201`) for Tamil speech, with an explicit check of community Tamil pronunciation acceptability (Piper Tamil uses eSpeak `ta` phonemization and showed minor syllable artifacts on round-trip STT).
