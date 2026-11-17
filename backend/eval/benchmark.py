"""Reusable benchmarking utilities for LocalVox Phase 1 AI engines."""

import json
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

import psutil
from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class BenchmarkRecord(BaseModel):
    """Single benchmark measurement for a LocalVox component and model."""

    model_config = ConfigDict(extra="forbid")

    component: Literal["VAD", "STT", "LLM", "TTS"]
    model_name: str
    language: str
    measurement_type: Literal["measured", "estimated", "untested"] = "measured"
    model_size_mb: float | None = None
    load_time_ms: float | None = None
    inference_time_ms: float | None = None
    memory_mb: float | None = None
    rss_delta_mb: float | None = None
    audio_duration_s: float | None = None
    rtf: float | None = None
    ttft_ms: float | None = None
    tokens_per_sec: float | None = None
    accuracy: str | None = None
    hardware: str
    notes: str = ""
    timestamp: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Populate UTC timestamp when omitted."""
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


def get_process_memory_mb() -> float:
    """Return current process resident set size (RSS) in MiB."""
    rss_bytes = psutil.Process().memory_info().rss
    return round(rss_bytes / (1024.0 * 1024.0), 2)


def get_disk_size_mb(path: Path) -> float | None:
    """Return total disk size of a file or directory in MiB, or None if missing."""
    if not path.exists():
        return None
    if path.is_file():
        return round(path.stat().st_size / (1024.0 * 1024.0), 2)
    total_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(total_bytes / (1024.0 * 1024.0), 2)


def compute_rtf(inference_time_ms: float, audio_duration_s: float) -> float | None:
    """Calculate real-time factor (processing_s / audio_s); < 1.0 is faster than real time."""
    if audio_duration_s <= 0.0:
        return None
    return round((inference_time_ms / 1000.0) / audio_duration_s, 4)


def detect_hardware_summary() -> str:
    """Detect local CPU brand and physical RAM via sysctl on macOS."""
    cpu = platform.processor() or platform.machine()
    mem_gb: float | None = None
    try:
        cpu_out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if cpu_out:
            cpu = cpu_out
        mem_out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if mem_out:
            mem_gb = round(int(mem_out) / (1024.0**3), 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    if mem_gb is not None:
        return f"{cpu} ({mem_gb:.0f} GB Unified Memory)"
    return cpu


def measure_timed(fn: Callable[[], T]) -> tuple[T, float, float, float]:
    """Execute fn and return (result, elapsed_ms, rss_after_mb, rss_delta_mb)."""
    rss_before = get_process_memory_mb()
    start = time.perf_counter()
    result = fn()
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    rss_after = get_process_memory_mb()
    rss_delta = round(max(0.0, rss_after - rss_before), 2)
    return result, elapsed_ms, rss_after, rss_delta


def save_benchmark_json(records: Sequence[BenchmarkRecord], output_path: Path) -> Path:
    """Persist benchmark records as machine-readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "hardware": detect_hardware_summary(),
        "results": [rec.model_dump() for rec in records],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_benchmark_json(input_path: Path) -> list[BenchmarkRecord]:
    """Load benchmark records from a JSON file created by save_benchmark_json."""
    if not input_path.exists():
        raise FileNotFoundError(f"Benchmark JSON file not found: {input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    raw_results = data.get("results", data) if isinstance(data, dict) else data
    return [BenchmarkRecord.model_validate(item) for item in raw_results]


def format_markdown_table(records: Sequence[BenchmarkRecord]) -> str:
    """Format benchmark records into a Markdown comparison table."""
    header = (
        "| Component | Model | Type | Lang | Size (MiB) | Load (ms) | "
        "Inference (ms) | Memory (MiB) | RTF | Accuracy / Notes |"
    )
    lines = [
        header,
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rec in records:
        size = f"{rec.model_size_mb:.2f}" if rec.model_size_mb is not None else "n/a"
        load_ms = f"{rec.load_time_ms:.1f}" if rec.load_time_ms is not None else "n/a"
        inf_ms = f"{rec.inference_time_ms:.1f}" if rec.inference_time_ms is not None else "n/a"
        mem = f"{rec.memory_mb:.1f}" if rec.memory_mb is not None else "n/a"
        rtf = f"{rec.rtf:.4f}" if rec.rtf is not None else "n/a"
        detail = rec.accuracy if rec.accuracy else rec.notes
        if rec.tokens_per_sec is not None:
            tps_info = f"{rec.tokens_per_sec:.1f} tok/s"
            if rec.ttft_ms is not None:
                tps_info += f", TTFT {rec.ttft_ms:.1f} ms"
            detail = f"{tps_info}; {detail}".strip("; ")
        lines.append(
            f"| {rec.component} | `{rec.model_name}` | {rec.measurement_type} | {rec.language} | "
            f"{size} | {load_ms} | {inf_ms} | {mem} | {rtf} | {detail} |"
        )
    return "\n".join(lines)
