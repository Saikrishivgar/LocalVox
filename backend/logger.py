"""Structured JSON logging and stage latency timer."""

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.config import AppConfig, get_config


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record into a JSON string."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str, config: AppConfig | None = None) -> logging.Logger:
    """Create or return a structured JSON logger configured from AppConfig."""
    cfg = config if config is not None else get_config()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, cfg.logging.level, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter: logging.Formatter = (
        JsonFormatter()
        if cfg.logging.json_format
        else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    if cfg.logging.stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if cfg.logging.file:
        log_file = cfg.logging.app_log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@contextmanager
def stage_timer(stage: str, log_path: Path | None = None) -> Iterator[dict[str, Any]]:
    """Measure stage latency in milliseconds and append a JSONL record."""
    target_path = log_path if log_path is not None else get_config().logging.latency_log_file
    record: dict[str, Any] = {"stage": stage}
    start = time.perf_counter()
    try:
        yield record
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
        record["timestamp"] = datetime.now(UTC).isoformat()
        record["latency_ms"] = elapsed_ms
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
