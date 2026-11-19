# LocalVox Phase 6 — Jarvis Wake Word, 3-Sample Voice Personalization & Hands-Free Intelligence

## 1. Architecture Overview

Phase 6 transforms LocalVox into an always-ready, 100% offline hands-free voice companion activated by **"Hey Jarvis"** — enhanced with **3-sample local voice personalization** so LocalVox adapts cleanly to the user's exact voice, accent, and microphone acoustics without cloud APIs or raw audio storage.

```text
MacBook Pro Microphone (single shared CoreAudio/sounddevice stream, 16 kHz mono, 512 samples / 32 ms)
   │
   ├── [SLEEPING standby] ──▶ OpenWakeWordEngine (pure onnxruntime + numpy, 80 ms / 1280-sample steps)
   │                              ├── melspectrogram.onnx  (16 kHz PCM → 32-bin log-mel)
   │                              ├── embedding_model.onnx (76×32 mel → 96-dim speech embedding)
   │                              ├── hey_jarvis_v0.1.onnx (16×96 emb → generic probability)
   │                              └── PersonalizedVoiceProfile (optional 3-sample voice_profile.npz, <0.01 ms)
   │                                    • Baseline-centered (16×96) phonetic trajectory cosine
   │                                    • 32-bin active-speech log-mel speaker timbre cosine
   │                                    • Supports Hybrid (default) or "My Voice Only" mode
   │                              │
   │                              ▼ (validated RMS + patience + cooldown)
   │                          WAKE_DETECTED ──▶ optional cached TTS ack ("Yes, I'm listening.")
   │                                            + configurable grace period (350 ms)
   │                                            + bounded in-memory pre-roll (when ack disabled)
   │
   ├── [ENROLLMENT (3×)] ──▶ VoicePipeline.capture_enrollment_utterance() (shared mic + Silero VAD)
   │                              └─▶ OpenWakeWordEngine.extract_enrollment_sample()
   │                                  → raw PCM discarded immediately; ONLY embeddings saved to .npz
   │
   └── [ACTIVE SESSION]  ──▶ Existing Phase 1–5 Voice Pipeline (unchanged, zero wake interference):
                              LISTENING (Silero VAD) → PROCESSING (MLX Whisper)
                              → Memory Retrieval (SQLite FTS5) → GENERATING (Ollama Qwen3:8B)
                              → SPEAKING (Kokoro EN / Piper TA + BargeInMonitor + EchoSuppressor)
                              → INTERRUPTED (instant barge-in cancel)
                              → Inactivity timeout (default 15 s) → RETURNING_TO_STANDBY → SLEEPING
```

## 2. openWakeWord Investigation & Runtime Selection

- **Official Repository**: `https://github.com/dscripka/openWakeWord` (release `v0.5.1`).
- **Why Direct ONNX (`onnxruntime` + `numpy`) Instead of PyPI `openwakeword`**:
  - On macOS Apple Silicon (ARM64, Python 3.11), PyPI `openwakeword` requires `tflite-runtime` (no ARM64 macOS wheel) and pulls heavy training/sklearn dependencies.
  - LocalVox already ships Apple Silicon `onnxruntime` and `numpy`. Running the three official `v0.5.1` ONNX artifacts directly adds **zero new Python dependencies** and keeps the wake engine strictly self-contained in `backend/wakeword/`.
- **Verified ONNX Artifacts (`data/models/wakeword/`, 3.52 MiB total)**:
  1. `melspectrogram.onnx` (`1,087,958` bytes) — input `input` `[batch, samples]` (float32 scaled to int16 `-32768..32767`), output `[1, 1, frames, 32]`, normalized via `spec / 10.0 + 2.0`.
  2. `embedding_model.onnx` (`1,326,578` bytes) — input `input_1` `[batch, 76, 32, 1]`, output `conv2d_19` `[batch, 1, 1, 96]`.
  3. `hey_jarvis_v0.1.onnx` (`1,271,370` bytes) — input `x.1` `[1, 16, 96]`, output `53` `[1, 1]` probability in `[0, 1]`.
- **Execution Provider Choice (`CPUExecutionProvider` vs `CoreMLExecutionProvider`)**:
  - Measured directly on Apple M5: `CPUExecutionProvider` (single-threaded) runs **`2.2×–2.5×` faster** (`1.08 ms` vs `2.41 ms`), loads **`84×–123×` faster** (`3.31–4.62 ms` vs `389.9 ms`), and consumes **less RSS** (`13.48 MiB` vs `17.60 MiB`) because CoreML only partitions 11/18 and 44/65 nodes of these tiny graphs. Therefore `execution_provider: cpu` is the verified default.

## 3. 3-Sample Personalized Voice Enrollment

Generic US/UK TTS-trained wake-word classifiers can under-score accented speech or varied room acoustics. LocalVox includes a 100% offline **3-Sample Voice Personalization** module (`backend/wakeword/enrollment.py`):

1. **Guided Capture**: User clicks **Personalize Voice** and speaks `"Hey Jarvis"` 3 times via the shared microphone stream (`VoicePipeline.capture_enrollment_utterance` with snappy `480 ms` VAD hangover).
2. **Baseline-Centered Neural Trajectory + Speaker Timbre**:
   - Raw Google Speech Embeddings carry a large shared DC offset (`baseline_embedding`). Subtracting `self._baseline_embedding` centers silence near zero norm (`||delta|| ≈ 0`).
   - Each enrolled sample extracts a unit-normalized `(1536,)` centered `(16, 96)` phonetic-speaker trajectory anchored over the active speech window plus embedding tail, plus a `(32,)` mean-centered active-speech log-mel timbre vector.
   - All 3 templates + L2-normalized centroid + timbre centroid + pairwise consistency score are stored atomically in `data/models/wakeword/voice_profile.npz` (`~26.4 KB`).
3. **Zero Raw Audio Persistence**: Raw PCM waveforms are discarded immediately in memory (`del waveform`) right after embedding extraction — no `.wav` or `.pcm` recordings are ever written to disk.
4. **Live Scoring (< 0.01 ms overhead)**:
   - On every 80 ms step during `SLEEPING`, `PersonalizedVoiceProfile.score_window()` evaluates `0.5 * dot(win, centroid) + 0.5 * max_i dot(win, template_i)` weighted by `(0.75 + 0.25 * timbre_sim)`.
   - Supports **Hybrid Mode** (default when enrolled: triggers on personalized match OR generic openWakeWord) and **"My Voice Only" Mode** (`personal_only: true`: triggers strictly on the enrolled user's voice).

## 4. Shared CoreAudio Microphone Ownership

- LocalVox maintains **exactly one** `MicrophoneAudioCapture` (`sounddevice.InputStream`, 16 kHz mono, 512-sample / 32 ms frames).
- Audio callback remains completely non-blocking (pushes into bounded queue, drops oldest if full).
- Ownership handoffs are strictly serialized:
  1. `SLEEPING` standby reads chunks for `OpenWakeWordEngine` and maintains a small bounded in-memory `pre_roll` deque (never persisted).
  2. On `WAKE_DETECTED`, mic is paused (`capture.pause()`), stale buffers cleared, optional cached `"Yes, I'm listening."` ack is played without echo, and grace period elapses before resuming capture for `LISTENING`.
  3. During `LISTENING` and `SPEAKING`, `OpenWakeWordEngine` is completely inactive — zero interference with Silero VAD, MLX Whisper, Kokoro/Piper TTS, `EchoSuppressor`, or `BargeInMonitor`.
  4. During 3-sample enrollment, active standby pauses cleanly, enrollment reuses the same `MicrophoneAudioCapture` + Silero VAD, and standby resumes automatically afterward.
  5. Disabling wake-word monitoring (`POST /api/wakeword/disable`) stops `MicrophoneAudioCapture` immediately and cleanly releases the CoreAudio input device.

## 5. Conversation State Machine (`backend/pipeline/state.py`)

Validated Phase 6 states and transitions:

- `IDLE` ↔ `SLEEPING` (standby monitoring active)
- `SLEEPING` → `WAKE_DETECTED` → `SPEAKING` (optional ack) or `LISTENING`
- `LISTENING` → `PROCESSING` → `GENERATING` → `SPEAKING` → `LISTENING`
- `SPEAKING` / `GENERATING` → `INTERRUPTED` → `LISTENING` (existing barge-in preserved; wake engine never spawns duplicate sessions)
- `LISTENING` (after `conversation_timeout_s` inactivity) → `RETURNING_TO_STANDBY` → `SLEEPING`
- Any active state → `STOPPING` → `IDLE` (clean shutdown)

## 6. Privacy Guarantees

- **100% Local Inference**: All ONNX sessions, personalized embeddings, Silero VAD, MLX Whisper, Ollama Qwen3:8B, and Kokoro/Piper TTS run exclusively on localhost / local CPU & Metal.
- **No Standby STT/LLM**: Audio captured in `SLEEPING` is fed strictly into `OpenWakeWordEngine` and never sent to Whisper or Ollama.
- **No Background Audio Recording**: Only bounded transient in-memory ring buffers (`pre_roll_ms`, `76×32` mel window, `16×96` embedding window) exist during standby; zero `.wav` files are saved during standby or voice enrollment.
- **Visible Microphone Indicator & Instant Kill-Switch**: Status bar, sidebar, and Settings view show live microphone state (`Mic Standby · Hey Jarvis`, `Mic Active`, or `Mic Off`). Clicking **Disable** immediately stops `MicrophoneAudioCapture`.

## 7. Real Apple M5 Performance Benchmarks

Measured on **Apple M5 MacBook Pro (24 GB Unified Memory)** via `uv run python scripts/wakeword_benchmark.py` (raw JSON saved in [`docs/phase6_wakeword_benchmarks.json`](phase6_wakeword_benchmarks.json)):

| Metric | Measured Apple M5 Value |
| --- | --- |
| ONNX Artifacts Disk Footprint | **3.52 MiB** (`melspectrogram` + `embedding_model` + `hey_jarvis_v0.1`) |
| Personalized 3-Sample Profile Disk Size | **26.4 KB** (`voice_profile.npz`, embeddings only) |
| Model Load Time (`CPUExecutionProvider`) | **3.31–4.62 ms** (vs `389.88 ms` CoreML) |
| Per-Frame Inference Time (80 ms step, CPU) | **1.080 ms mean** (`p50 = 1.027 ms`, `p95 = 1.325 ms`) |
| Per-Frame Inference Time (CoreML comparison) | `2.406 ms mean` (`p50 = 2.315 ms`, `p95 = 2.697 ms`) |
| Wake-Word Detection Offset (vs speech end) | **-224 ms to +74.6 ms** (triggers near final syllable of `"Jarvis"`) |
| Standby CPU Utilization (real-time 32 ms loop) | **5.6%** of 1 core (~0.56% system CPU) |
| Memory Consumption (RSS Delta) | **+13.48 MiB** (`296.34 → 333.84 MiB` process RSS) |
| Positive Recall (`hey_jarvis_v0.1.onnx`) | **10 / 10 (100%)**, 0 missed activations (`0.9737–0.9951` scores) |
| False Activation Rate (13 EN/TA/noise/silence negatives) | **0 / 13 (0%)** (`0.0001–0.3202` scores) |
| 3-Sample Personalized Voice Recall (`personal_only=True`) | **3 / 3 (100%)**, **0 / 3 (0%)** false activations (`95.6%` consistency) |
| Wake-to-Listening Latency (Instant / `ack_enabled=False`) | **2.58 ms** |
| Wake-to-Listening Latency (Cached `"Yes, I'm listening."` + 350 ms grace) | **1566.0 ms** (`1216 ms` audio + `350 ms` grace; first synth `335.41 ms`) |
| Barge-In Detection Latency (with wake engine loaded) | **0.53 ms** (zero regression) |
| E2E Speech-to-Response (Manual vs Wake-Word Active Turn) | Manual cold KV: `STT 846.0 ms / TTFT 3162.0 ms / 1st Audio 4903.8 ms`; Wake-Word warm turn: `STT 729.4 ms / TTFT 420.5 ms / 1st Audio 2012.4 ms` (zero overhead) |
