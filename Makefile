.PHONY: run test lint

run:
	uv run uvicorn backend.app.main:app --reload

test:
	uv run pytest

lint:
	uv run ruff check .

