"""Tests for structured logging and stage latency timing."""

import json
import time
from pathlib import Path

from backend.logger import stage_timer


def test_stage_timer_logs_latency(tmp_path: Path) -> None:
    """Verify stage_timer records stage latency in ms to a JSONL file."""
    log_file = tmp_path / "latency.jsonl"

    with stage_timer("stt", log_path=log_file) as result:
        time.sleep(0.01)

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["stage"] == "stt"
    assert isinstance(entry["latency_ms"], float)
    assert entry["latency_ms"] > 0.0
    assert entry["latency_ms"] == result["latency_ms"]
    assert "timestamp" in entry

