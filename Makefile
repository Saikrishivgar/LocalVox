.PHONY: desktop run api frontend frontend-install frontend-test frontend-build test lint download-models benchmark voice voice-wake memory-bench wakeword-bench

desktop:
	uv run python scripts/launch_desktop.py

run:
	uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

api:
	HF_HUB_OFFLINE=1 uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

frontend-install:
	cd frontend && npm install

frontend-test:
	cd frontend && npm test

frontend-build:
	cd frontend && npx tsc --noEmit && npm run build

frontend:
	cd frontend && npm run dev

test:
	uv run pytest

lint:
	uv run ruff check .

download-models:
	uv run python scripts/download_models.py

benchmark:
	uv run python scripts/run_benchmarks.py

voice:
	uv run python scripts/voice_assistant.py

voice-wake:
	HF_HUB_OFFLINE=1 uv run python scripts/voice_assistant.py --wake-word

memory-bench:
	uv run python scripts/memory_benchmark.py

wakeword-bench:
	uv run python scripts/wakeword_benchmark.py

