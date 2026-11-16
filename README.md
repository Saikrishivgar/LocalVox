# LocalVox

Offline real-time Tamil+English voice study assistant running locally on macOS Apple Silicon (no cloud APIs).

## Stack

- **Runtime**: Python 3.11 (`uv`), FastAPI, Uvicorn
- **Frontend**: Next.js + TypeScript + Tailwind CSS (in `frontend/`)
- **Voice & AI**: Silero VAD, `mlx-whisper`, Ollama, Kokoro / Piper
- **RAG**: Qdrant + `bge-m3`

## Structure

- `backend/app/` — FastAPI application and routes
- `backend/pipeline/` — Real-time voice pipeline orchestration
- `backend/stt/` — VAD and speech-to-text
- `backend/llm/` — Local LLM integration (Ollama)
- `backend/tts/` — Text-to-speech synthesis
- `backend/rag/` — Document indexing and vector retrieval
- `backend/tools/` — Study assistant tools
- `backend/eval/` — Latency and accuracy evaluation
- `frontend/` — Web UI (planned)
- `scripts/` — Setup and utility scripts
- `tests/` — Automated tests
- `docs/` — Architecture and design notes

## Commands

```bash
uv sync --python 3.11
make test
make lint
make run
```

