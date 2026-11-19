# LocalVox Phase 5 Desktop Companion API & WebSocket Specification

All desktop companion API routes and WebSocket channels bind strictly to local loopback (`127.0.0.1:8000` by default) and enforce an explicit origin allowlist (`http://127.0.0.1:3000`, `http://localhost:3000`). No cloud services or remote inference endpoints are used.

## Security Model

- **Host restriction**: `ServerConfig.host` accepts only `"127.0.0.1"` or `"localhost"`.
- **CORS allowlist**: Explicit `allowed_origins` (`allow_credentials=False`); wildcard origins are never configured.
- **WebSocket origin validation**: Connections with a disallowed `Origin` header are closed immediately with code `1008` (Policy Violation).
- **Strict schemas**: All Pydantic request schemas use `ConfigDict(extra="forbid")`.
- **Safe settings updates**: `PUT /api/settings` accepts only allowlisted voice, streaming, barge-in, LLM, and memory toggles—filesystem paths (`paths.*`, `memory.db_path`, `stt.local_model_path`) cannot be modified via the UI or API.
- **Privacy & destructive safeguards**: Stored memories pass through `MemoryStore` secret/credential filtering; bulk wipe requires `confirm_phrase == "CLEAR_ALL_MEMORIES"`.

## REST Endpoints

### Health & Companion State
- `GET /api/health` — Returns `{ status: "ok", pipeline_state, mic_active }` without initializing audio hardware.
- `GET /api/state` — Returns full hydration snapshot `{ pipeline_state, mic_active, session_id, streaming_partial, messages, last_metrics }`.
- `POST /api/chat` — Body `{ text, speak: bool = true, language?: string }`. Offloads `VoicePipeline.process_text(...)` to a worker thread so `/ws/events` streams live tokens and audio levels without blocking the event loop.

### Microphone & Playback Control
- `POST /api/voice/start` — Starts `VoicePipeline.run(warmup=..., shutdown_memory=False)` in a dedicated daemon thread (`mic_active = true`).
- `POST /api/voice/stop` — Signals cancellation and joins the voice loop cleanly without closing SQLite databases.
- `POST /api/voice/stop-speaking` — Immediately aborts active LLM token streaming, TTS synthesis, and speaker playback.

### Persistent Memory Dashboard (`/api/memories`)
- `GET /api/memories?query=&category=&limit=100` — Returns `{ total_count, category_counts, items, match_scores }` using `MemoryStore` + `FtsMemoryRetriever`.
- `POST /api/memories` — Explicitly adds `{ category, subject, content, keywords, importance }` (`400` if blocked by secret/privacy filter).
- `PATCH /api/memories/{memory_id}` — Updates existing memory and reindexes FTS5.
- `DELETE /api/memories/{memory_id}` — Deletes one record permanently.
- `POST /api/memories/clear` — Requires `{ confirm_phrase: "CLEAR_ALL_MEMORIES" }` (`422` otherwise) before clearing SQLite memories.

### Conversation History (`/api/conversations`)
- `GET /api/conversations?query=&limit=50` — Lists persisted sessions from `ConversationHistoryStore` (`localvox_conversations.db`). Fresh databases return `[]` with zero fabricated entries.
- `POST /api/conversations/new` — Clears active `ConversationManager` buffer and assigns a fresh in-memory session ID (persisted only once a real turn completes).
- `GET /api/conversations/{session_id}` — Returns full ordered messages and per-turn metrics.
- `POST /api/conversations/{session_id}/continue` — Hydrates up to `max_history_turns` real Q/A pairs into active `ConversationManager` and switches active UI session.
- `DELETE /api/conversations/{session_id}` — Removes saved session and cascade-deletes messages + FTS5 index rows.

### System Monitor & Settings
- `GET /api/system` — Returns real host hardware summary, CPU %, unified RAM GB/%, process RSS MiB (`psutil`), local Ollama reachability and model memory, engine readiness (`Ollama Qwen LLM`, `Silero VAD`, `MLX Whisper STT`, `Kokoro English TTS`, `Piper Tamil TTS`, `SQLite + FTS5 Memory`), extraction worker counters, and latest records from `logs/latency.jsonl`.
- `GET /api/settings` — Returns current `SettingsPayload`, available Kokoro voice IDs, and available Ollama tags.
- `PUT /api/settings` — Validates allowlisted fields, applies updates live via `VoicePipeline.set_config(...)`, and writes atomically to `config.yaml`.

## WebSocket Channel (`WS /ws/events`)

Events are broadcast non-blockingly from audio/worker threads using `loop.call_soon_threadsafe(...)`. If a slow client queue fills, low-priority `audio_level` frames are dropped first so state and transcript messages never stall voice inference.

- `snapshot` — Sent immediately upon connection with full state hydration.
- `state_changed` — `{ state, previous_state, reason, mic_active }`
- `mic_status` — `{ mic_active }`
- `transcript` — `{ message: LiveMessageItem }`
- `response_token` — `{ token, partial }`
- `response_complete` — `{ message: LiveMessageItem }`
- `turn_complete` — `{ session_id, metrics }`
- `audio_level` — `{ source: "mic" | "speaker", rms, bars }` (24 real frequency-slice bars)
- `barge_in` — `{ detect_latency_ms }`
- `session_changed` / `session_deleted` / `memory_changed` / `settings_updated` / `error`
