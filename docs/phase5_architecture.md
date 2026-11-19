# LocalVox Phase 5 Architecture & Implementation Plan

## Objective

Build a local-first, macOS-inspired desktop AI companion interface in Next.js + TypeScript + Tailwind CSS + Framer Motion + Three.js/React Three Fiber communicating with LocalVox's existing Python Phase 1–4 engine via FastAPI REST APIs and WebSockets, bound strictly to `127.0.0.1`.

## Key Design Constraints

- Preserve all Phase 1–4 engines (`SileroVadDetector`, `MlxWhisperStt`, `OllamaClient`, `KokoroTtsEngine`, `PiperTtsEngine`, `TtsRouter`, `VoicePipeline`, `MemoryManager`) without rebuilding them.
- Zero cloud APIs or external inference; strict `127.0.0.1` binding and explicit origin allowlist (`http://127.0.0.1:3000`, `http://localhost:3000`).
- Zero blocking I/O between frontend WebSocket broadcasts and real-time voice threads (`loop.call_soon_threadsafe` fan-out from throttled audio-level and state observers).
- Keep Phase 4 `MemoryDatabase` schema (`v1`) intact; store persistent UI conversation transcripts cleanly in `ConversationHistoryStore` (`data/memory/localvox_conversations.db`).

## Backend Plan (`backend/app/`)

1. **`config.yaml` & `backend/config.py`**:
   - Add `paths.conversations_db_file: data/memory/localvox_conversations.db`.
   - Add `ServerConfig` (`host: 127.0.0.1`, `port: 8000`, `allowed_origins`).
2. **Throttled Audio Waveform Hooks (`backend/pipeline/audio_io.py`, `barge_in.py`, `voice_pipeline.py`)**:
   - Add `compute_waveform_bars()` and optional `on_audio_level(source, rms, bars)` callback (~20 Hz throttled) in Python worker loops (never inside PortAudio C callbacks).
   - Add `VoicePipeline.process_text(text, *, speak=True)` reusing existing `MemoryCommandHandler`, `FtsMemoryRetriever`, `ConversationManager`, `_execute_streaming_turn`, and `AsyncMemoryWorker`.
3. **Persistent Conversation History (`backend/app/history.py`)**:
   - Thread-safe SQLite (`WAL` + `FTS5`) store managing genuine `sessions` and `messages`, search across past sessions, session inspection, and loading historical turns into `ConversationManager` for session continuation.
4. **Companion Runtime (`backend/app/runtime.py`)**:
   - Bridges `VoicePipeline`, `MemoryManager`, `ConversationHistoryStore`, real `psutil`/`Ollama` system monitoring, and thread-safe async WebSocket broadcasting (`ConnectionManager`) with lazy/safe start/stop voice lifecycle (no unrequested hardware capture at import time).
5. **FastAPI App (`backend/app/main.py` & `backend/app/schemas.py`)**:
   - Pydantic request/response models (`extra="forbid"`).
   - Local REST endpoints:
     - `GET /api/health`
     - `GET /api/state` (pipeline state, session, config summary)
     - `POST /api/voice/start`, `POST /api/voice/stop`, `POST /api/voice/stop-speaking`
     - `POST /api/chat` (text input with optional TTS playback)
     - `GET /api/memories`, `POST /api/memories`, `PATCH /api/memories/{id}`, `DELETE /api/memories/{id}`, `DELETE /api/memories` (requires explicit confirmation payload)
     - `GET /api/conversations`, `GET /api/conversations/{session_id}`, `POST /api/conversations/new`, `POST /api/conversations/{session_id}/continue`, `DELETE /api/conversations/{session_id}`
     - `GET /api/system` (real Ollama/Qwen3/VAD/Whisper/Kokoro/Piper/SQLite/worker/CPU/RAM metrics + recent real latencies)
     - `GET /api/settings`, `PUT /api/settings` (safe allowlisted settings validated against `AppConfig` and saved to `config.yaml`)
   - WebSocket `/ws/events`: real-time state changes, audio waveform bars (`mic`/`speaker`), live STT transcripts, streamed LLM tokens, assistant messages, barge-in alerts, turn metrics, and chat/voice control frames.

## Frontend Plan (`frontend/`)

1. **Workspace Layout (`src/app/`)**: Obsidian dark frosted-glass macOS studio shell with sidebar rail (`Companion`, `Memories`, `Conversations`, `System Monitor`, `Settings`) and persistent live status header.
2. **Main Companion Screen**:
   - React Three Fiber / Three.js animated 3D `CompanionOrb` reacting directly to backend `PipelineState` (`IDLE`, `LISTENING`, `PROCESSING`, `GENERATING`, `SPEAKING`, `INTERRUPTED`, `ERROR`) and real `rms` amplitude.
   - Live audio waveform bar visualizer driven exclusively by real backend `audio_level` events.
   - Explicit indicators (`Listening`, `Thinking`, `Speaking`), live conversation feed with memory badges & streaming tokens, text input bar, Mic Start/Stop, and Stop Speaking / Barge-in controls.
3. **Memory Dashboard**: Total stats, 8-category filters, SQLite FTS5 search, create/inspect/edit/delete records, confirmed bulk clear modal.
4. **Conversation History**: Searchable persisted session list with timestamps, message inspector, and **Continue Conversation** handoff back to Companion.
5. **System Monitor**: Live Ollama/Qwen3, VAD, Whisper, Kokoro, Piper, SQLite integrity, extraction worker, host CPU/RAM, and recent stage latency table.
6. **Settings**: Voice, speaking speed, language, TTS routing, barge-in sensitivity, LLM model/temperature, and memory extraction/privacy persistence via `/api/settings`.

## Documentation & Verification Plan

- Backend FastAPI + WebSocket tests in `tests/test_phase5_api.py` + `tests/test_pipeline.py` & `tests/test_phase4_memory.py` regression checks (`make test`, `make lint`).
- Frontend unit/component tests + TypeScript check + `npm run build` production verification.
- Document REST + WS APIs (`docs/phase5_api.md`), update `Makefile` and `README.md`, and capture real rendered screenshots via Chrome DevTools (`browser` subagent).
