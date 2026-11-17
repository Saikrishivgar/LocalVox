# LocalVox

Offline real-time Tamil+English voice study assistant running locally on macOS Apple Silicon (no cloud APIs).

## Current Status (Phase 1 Complete)

| Component | Engine / Model | Status | Notes |
| --- | --- | --- | --- |
| Configuration & Logging | Pydantic + YAML + JSONL | Ready | Extended config + structured logs + `stage_timer` |
| VAD (Component A) | Silero VAD (`onnx`) | Ready | Tested locally (`13.4 ms`, `RTF 0.0023`) |
| STT (Component B) | `mlx-whisper` (`whisper-large-v3-turbo-8bit`) | Ready (EN); Incomplete (Human TA) | Tested on synthetic EN & TA audio; human Tamil & Tanglish validation pending recordings |
| LLM (Component C) | Ollama (`qwen3:8b`) | Ready | Tested EN, TA, Tanglish, multi-turn (`~24 tok/s`, `TTFT 188–491 ms` with `think=False`) |
| TTS (Component D) | Kokoro ONNX (EN) + Piper ONNX (EN/TA) | Ready | Kokoro has no Tamil; Piper uses community `ta_IN-rasa_female-medium` |
| Benchmarking & Tests | `backend/eval/benchmark.py`, `pytest`, `ruff` | Ready | `11` automated tests + `ruff` passing |
| End-to-End Pipeline | `backend/pipeline/`, `backend/app/` | Not started | Scheduled for Phase 2 |

> Full hardware measurements, model comparisons, and limitations are documented in [docs/model_benchmarks.md](docs/model_benchmarks.md).

## Stack

- **Runtime**: Python 3.11 (`uv`), FastAPI, Uvicorn
- **Frontend**: Next.js + TypeScript + Tailwind CSS (in `frontend/`, Phase 3+)
- **Voice & AI**: Silero VAD (`onnx`), `mlx-whisper`, Ollama (`qwen3:8b`), `kokoro-onnx` + `piper-tts`
- **RAG**: Qdrant + `bge-m3` (planned)

## Structure

- `backend/config.py` — Pydantic v2 configuration schema and YAML loader (`config.yaml`)
- `backend/logger.py` — Structured JSON logger and `stage_timer` (`logs/latency.jsonl`)
- `backend/stt/` — `vad.py` (Silero VAD) and `whisper_stt.py` (MLX Whisper)
- `backend/llm/` — `ollama_client.py` (local Ollama streaming HTTP client)
- `backend/tts/` — `kokoro_tts.py` (Kokoro ONNX) and `piper_tts.py` (Piper ONNX)
- `backend/eval/` — `benchmark.py` (reusable benchmark metrics, memory/disk helpers, JSON/MD export)
- `backend/app/`, `backend/pipeline/`, `backend/rag/`, `backend/tools/` — Phase 2+ packages
- `scripts/` — Independent verification and benchmark scripts (`vad_test.py`, `stt_test.py`, `llm_test.py`, `tts_test.py`, `download_models.py`, `run_benchmarks.py`)
- `tests/` — Automated unit tests (`test_config.py`, `test_logger.py`, `test_phase1.py`)
- `docs/` — Benchmark results (`model_benchmarks.md`, `model_benchmarks.json`)

## Installation

1. **System prerequisites** (macOS Apple Silicon):
   ```bash
   brew install uv ffmpeg
   # Install Ollama from https://ollama.com or brew install --cask ollama
   ollama pull qwen3:8b
   ```
2. **Python 3.11 environment** (`uv`):
   ```bash
   uv sync --python 3.11
   ```
3. **Download local Phase 1 model weights** (stored in gitignored `data/models/`):
   ```bash
   make download-models
   # Or download all candidate Whisper sizes for comparison:
   uv run python scripts/download_models.py --all-stt
   ```

## Configuration

All model identifiers, local model paths, sample rates, thresholds, LLM parameters, TTS voices, benchmark settings, and logging options are defined in [`config.yaml`](config.yaml). Override the config path using:

```bash
export LOCALVOX_CONFIG=/path/to/custom_config.yaml
```

## Independent Component Verification

```bash
# Component A: Silero VAD (generates local sample WAV if --audio omitted)
uv run python scripts/vad_test.py
uv run python scripts/vad_test.py --audio path/to/speech.wav --threshold 0.5

# Component B: MLX Whisper STT
uv run python scripts/stt_test.py data/test_audio/vad_sample.wav --language en
uv run python scripts/stt_test.py path/to/tamil.wav --language ta

# Component C: Local Ollama LLM (qwen3:8b)
uv run python scripts/llm_test.py --suite

# Component D: Kokoro (EN only) & Piper (EN + community TA)
uv run python scripts/tts_test.py --backend kokoro --language en --text "Hello LocalVox!"
uv run python scripts/tts_test.py --backend piper --language en --text "Hello LocalVox!"
uv run python scripts/tts_test.py --backend piper --language ta --text "வணக்கம்!"

# Isolated Phase 1 benchmark suite (writes docs/model_benchmarks.json)
make benchmark
```

## Offline Execution

Once `make download-models` and `ollama pull qwen3:8b` complete, all components run completely offline without internet access or cloud APIs. Enforce offline Hugging Face mode during execution with:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/stt_test.py data/test_audio/vad_sample.wav
```

Ensure `ollama serve` (or the Ollama desktop app) is running locally on `http://127.0.0.1:11434`.

## Tests & Lint

```bash
make test
make lint
```
