.PHONY: run test lint download-models benchmark

run:
	uv run uvicorn backend.app.main:app --reload

test:
	uv run pytest

lint:
	uv run ruff check .

download-models:
	uv run python scripts/download_models.py

benchmark:
	uv run python scripts/run_benchmarks.py

