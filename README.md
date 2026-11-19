# LocalVox

Offline real-time Tamil+English voice study assistant running locally on macOS Apple Silicon (no cloud APIs).

## Current Status (Phase 6 Complete)

| Component | Engine / Model | Status | Notes |
| --- | --- | --- | --- |
| Configuration & Logging | Pydantic + YAML + JSONL | Ready | Phase 0–6 config (`wakeword`, `memory`, `pipeline`, `server`, paths) + structured JSON logs + `stage_timer` |
| Wake Word & Personalization (Phase 6) | openWakeWord ONNX (`hey_jarvis_v0.1`) + 3-Sample Voice Profile | Ready | Pure `onnxruntime` + `numpy` (`1.08 ms`/80 ms frame, `3.52 MiB`), 3-sample local voice enrollment (`voice_profile.npz`, zero raw audio saved), Hybrid & "My Voice Only" modes |
| Audio I/O & Echo Control | `sounddevice` + `EchoSuppressor` | Ready | Single shared CoreAudio mic stream across standby wake-word, 3-sample enrollment, Silero VAD, and `BargeInMonitor` |
| VAD & Endpointing | Silero VAD (`onnx`) + `SpeechEndpointDetector` | Ready | Pre-roll (`288 ms`), hangover (`832 ms`), post-pad (`160 ms`), seeded barge-in continuity |
| STT | `mlx-whisper` (`whisper-large-v3-turbo-8bit`) | Ready (EN); Incomplete (Human TA/Tanglish) | In-memory Metal GPU transcription (`~701–857 ms` warm); real human Tamil/Tanglish WER still untested |
| LLM | Ollama (`qwen3:8b`) | Ready | Cancellable streaming client (`~22–25.8 tok/s`), bounded session history (`ConversationManager`), grounded persistent memory injection |
| Streaming TTS | Kokoro ONNX (EN, 24 kHz) + Piper ONNX (TA/EN, 22.05 kHz) | Ready | Incremental sentence chunking (`SentenceBuffer`) + single-stream `play_stream` with immediate `abort()` cancellation |
| Persistent Memory (Phase 4) | SQLite (`WAL` + `FTS5 unicode61`) + Qwen3 Extractor | Ready | 8 structured categories, non-blocking async extraction, FTS5/BM25 + importance + recency ranking, duplicate/contradiction merge, privacy secret filters |
| Desktop Backend API (Phase 5–6) | FastAPI + WebSocket (`backend/app/`) | Ready | Localhost `127.0.0.1:8000` REST & `/ws/events` streaming, wake-word standby + 3-sample voice enrollment routes, `ConversationHistoryStore`, strict CORS/origin validation, safe YAML settings persistence |
| Desktop Frontend (Phase 5–6) | Next.js + TypeScript + Three.js (`frontend/`) | Ready | Large 3D audio-reactive orb responding to all 7 hands-free states, live standby wake meter, 3-sample Voice Personalization wizard, Memory Dashboard, History, System Monitor, Settings |
| RAG, VLM | `backend/rag/`, VLM integration | Not started | Planned for Phase 7+ |

> - Model benchmarks: [docs/model_benchmarks.md](docs/model_benchmarks.md)
> - Phase 2/2.1 audio pipeline: [docs/phase2_audio_architecture.md](docs/phase2_audio_architecture.md)
> - Phase 3 barge-in & streaming TTS: [docs/phase3_architecture.md](docs/phase3_architecture.md)
> - Phase 4 database schema: [docs/memory_schema.md](docs/memory_schema.md)
> - Phase 5 desktop architecture: [docs/phase5_architecture.md](docs/phase5_architecture.md)
> - Phase 5 REST & WebSocket API: [docs/phase5_api.md](docs/phase5_api.md)
> - Phase 6 Jarvis wake word & voice personalization: [docs/phase6_architecture.md](docs/phase6_architecture.md)

## Architecture

```text
Microphone (sounddevice, 16 kHz)
   │ 512-sample chunks (32 ms)
   ▼
SpeechEndpointDetector + BargeInMonitor (+ EchoSuppressor / PlaybackReferenceBuffer)
   │ float32 speech utterance
   ▼
MlxWhisperStt.transcribe_array ──▶ MemoryCommandHandler (inspect / remember / update / forget / clear)
   │ normal turn transcript
   ▼
FtsMemoryRetriever (SQLite FTS5 + BM25 + importance + recency, < 2 ms)
   │ formatted persistent context (user-confirmed vs inferred)
   ▼
ConversationManager ──▶ OllamaClient (qwen3:8b, cancellable stream)
   │ streamed tokens ──▶ SentenceBuffer ──▶ TTS Worker (Kokoro/Piper) ──▶ AudioChunk Queue
   │                                                                            ▼
   └── post-turn async enqueue ──▶ AsyncMemoryWorker (Qwen3 + Privacy Filter)   SpeakerAudioPlayer
                                       ▼
                               SQLite WAL + FTS5 DB
```

## Structure

- `backend/config.py` — Strict Pydantic v2 configuration loader ([`config.yaml`](config.yaml))
- `backend/logger.py` — Structured JSON logging and `stage_timer` ([`logs/latency.jsonl`](logs/latency.jsonl))
- `backend/memory/` — Decoupled SQLite + FTS5 persistent memory layer:
  - `models.py` — 8 `MemoryCategory` enums, provenance `MemorySource`, records, search results
  - `database.py` — WAL SQLite connection manager, versioned migrations, FTS5 triggers, integrity check
  - `store.py` — Deduplication, subject enrichment, contradiction updates, explicit CRUD & confirmed bulk clear
  - `retriever.py` — FTS5 `unicode61` prefix/exact search, category/project filters, hybrid ranking, extensible `MemoryRetriever` interface
  - `extractor.py` — Local Qwen3:8B structured extractor, meaningfulness filter, non-blocking `AsyncMemoryWorker`
  - `commands.py` — Spoken/text memory inspect, remember, update, forget, delete, and confirmed bulk clear handler
  - `privacy.py` — Secret, password, API token, private key, and card/SSN blocker
  - `manager.py` — Session `MemoryManager` facade & clean shutdown
- `backend/wakeword/` — Offline openWakeWord ONNX engine + 3-sample personalized voice enrollment:
  - `detector.py` — Pure `onnxruntime` + `numpy` streaming detector (`OpenWakeWordEngine`)
  - `enrollment.py` — 3-sample local voice embedding extractor & `PersonalizedVoiceProfile` (`.npz`, zero raw audio persisted)
  - `models.py` — Official `v0.5.1` ONNX artifact verification and downloader
  - `events.py` — Strict Pydantic detection & engine telemetry schemas
- `backend/pipeline/` — `audio_io.py`, `barge_in.py`, `conversation.py`, `echo.py`, `state.py`, `voice_pipeline.py`
- `backend/stt/` — `vad.py` (Silero VAD) and `whisper_stt.py` (MLX Whisper)
- `backend/llm/` — `ollama_client.py` (cancellable local Ollama HTTP client)
- `backend/tts/` — `kokoro_tts.py`, `piper_tts.py`, `router.py`, `streaming.py`, `types.py`
- `backend/app/` — `main.py` (FastAPI + WebSocket server), `runtime.py` (`CompanionRuntime`), `history.py` (`ConversationHistoryStore`), `schemas.py`
- `frontend/` — Next.js + TypeScript + Tailwind + Framer Motion + Three.js (`CompanionOrb`, `AudioWaveform`, `MemoryDashboard`, `ConversationHistoryView`, `SystemMonitorView`, `SettingsView`)
- `scripts/` — `voice_assistant.py`, `memory_benchmark.py`, `audio_loopback_test.py`, `vad_test.py`, `stt_test.py`, `llm_test.py`, `tts_test.py`, `download_models.py`, `run_benchmarks.py`
- `tests/` — `test_config.py`, `test_logger.py`, `test_phase1.py`, `test_pipeline.py`, `test_phase3.py`, `test_phase4_memory.py`, `test_phase5_api.py`

## Installation

1. **System prerequisites** (macOS Apple Silicon):
   ```bash
   brew install uv ffmpeg
   # Install Ollama from https://ollama.com or brew install --cask ollama
   ollama pull qwen3:8b
   ```
2. **Python 3.11 environment** (`uv`, no extra packages needed beyond `uv sync`):
   ```bash
   uv sync --python 3.11
   ```
3. **Download local model weights** (stored in gitignored `data/models/`):
   ```bash
   make download-models
   ```

## Launching LocalVox

### Desktop Companion Interface (Phase 5)

Launch the entire local stack (Ollama, FastAPI `127.0.0.1:8000`, and Next.js `127.0.0.1:3000`) with one command:

```bash
make desktop
```

Or run components individually in separate terminals:
1. Install frontend dependencies once (when setting up a new checkout):
   ```bash
   make frontend-install
   ```
2. Start the offline FastAPI + WebSocket backend (`127.0.0.1:8000`) and local Ollama (`http://127.0.0.1:11434`):
   ```bash
   make api
   ```
3. In a second terminal, launch the Next.js desktop companion UI (`http://127.0.0.1:3000`):
   ```bash
   make frontend
   ```

### Terminal Voice Assistant (CLI)

Ensure Ollama is running locally (`http://127.0.0.1:11434`), then launch the continuous offline voice assistant:

```bash
HF_HUB_OFFLINE=1 make voice
```

Useful CLI flags:
```bash
# List audio devices or inspect saved SQLite memories
uv run python scripts/voice_assistant.py --list-devices
uv run python scripts/voice_assistant.py --list-memories

# Customize barge-in sensitivity or disable barge-in / streaming TTS / memory
uv run python scripts/voice_assistant.py --barge-in-sensitivity low
uv run python scripts/voice_assistant.py --no-barge-in
uv run python scripts/voice_assistant.py --no-memory
uv run python scripts/voice_assistant.py --memory-db /tmp/custom_localvox.db
```

### Spoken Voice & Memory Commands

- **Session history**: `"clear conversation"`, `"reset conversation"`, `"new topic"`
- **Exit**: `"goodbye localvox"`, `"exit localvox"`, `"stop listening"` (or `Ctrl+C`)
- **Inspect memories**: `"What do you remember about me?"`, `"What do you remember about my projects?"`
- **Explicit save / update**: `"Remember that my exam is on Friday"`, `"Remember this"`, `"Update this memory to ..."`
- **Forget / wipe**: `"Forget ..."` / `"Delete memory <id>"` / `"Clear all memories"` (prompts for `"confirm clear all memories"`)

## Manual Phase 4 End-to-End Test Guide

1. Start Ollama locally (`ollama serve` or Ollama app) and launch Session 1:
   ```bash
   HF_HUB_OFFLINE=1 make voice
   ```
2. Speak two turns establishing project context:
   - You: `"I am building a computer vision project called CrimeVision."`
   - Wait for reply (e.g., asking what technologies you use), then say: `"YOLO, FastAPI and Next.js."`
   - Verify `[METRICS turn=2]` returns quickly without blocking for extraction.
3. Say `"What do you remember about my projects?"` — verify LocalVox reads back your saved CrimeVision summary directly from SQLite.
4. Say `"Remember my API key is sk-testtoken1234567890"` — verify privacy filtering refuses to store credentials.
5. Exit cleanly with `"goodbye localvox"` (or `Ctrl+C`), wait for `[STOPPING] Shutting down LocalVox cleanly...`, and verify persistence on CLI before restarting:
   ```bash
   uv run python scripts/voice_assistant.py --list-memories
   ```
6. Start Session 2 (`HF_HUB_OFFLINE=1 make voice`) and ask:
   - `"Let's continue my computer vision project."`
   - Verify `[METRICS turn=1]` shows `mem>=1(...)` and Qwen3 references **CrimeVision**, **YOLO**, **FastAPI**, and **Next.js**.
7. Test deletion & confirmation safety:
   - Say `"Clear all memories"` -> verify LocalVox asks for confirmation before deleting.
   - Say `"Cancel"` -> verify memories remain stored.
   - Optionally say `"Forget CrimeVision"` or `"Clear all memories"` followed by `"Confirm clear all memories"`.

## Benchmarks, Tests & Lint

Run all automated tests and linting offline (no microphone, speaker, or live Ollama server required):

```bash
make test
make lint
make frontend-test
make frontend-build
```

Run real offline Qwen3:8B + SQLite FTS5 Phase 4 extraction & retrieval benchmark and Phase 6 M5 wake-word benchmark:

```bash
make memory-bench
make wakeword-bench
```

Launch CLI voice assistant directly in hands-free `"Hey Jarvis"` wake-word standby:

```bash
make voice-wake
```

Run existing component & loopback checks:

```bash
uv run python scripts/audio_loopback_test.py --timeout 10
make benchmark
```

## Known Limitations

- **Software Echo Suppression (Not Kernel Hardware AEC)**: PortAudio on macOS does not invoke CoreAudio `kAudioUnitSubType_VoiceProcessingIO` hardware AEC. `EchoSuppressor` + `PlaybackReferenceBuffer` rejects correlated direct speaker echo cleanly in software, while loud built-in speakers in reverberant rooms can still challenge double-talk detection; headphones or `--no-barge-in` give strict isolation.
- **Live Physical Microphone Barge-In Tuning**: Mocked unit & synthetic DSP tests verify interruption flow; interactive human voice barge-in over physical `MacBook Pro Speakers` may require tuning `pipeline.barge_in_sensitivity` (`low` / `medium` / `high`) for room acoustics.
- **Human Tamil & Tanglish STT Untested**: Whisper (`whisper-large-v3-turbo-8bit`) accuracy on natural human Tamil and Tanglish speech has not yet been formally evaluated.
- **Keyword + FTS5 Retrieval Only (Before Phase 5 Semantic Vectors)**: Phase 4 uses deterministic SQLite FTS5 (`unicode61`) + hybrid BM25/importance/recency ranking; semantic synonym queries without shared keyword stems wait for future local `bge-m3` embeddings via `MemoryRetriever`.
