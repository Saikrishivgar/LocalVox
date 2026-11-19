"""Single-command local launcher starting Ollama (if needed), FastAPI, and Next.js."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from backend.config import get_config


def _http_ready(url: str, *, timeout_s: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return 200 <= int(resp.status) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _wait_for_http(
    url: str,
    *,
    name: str,
    proc: subprocess.Popen[bytes] | None,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _http_ready(url, timeout_s=0.8):
            print(f"[LocalVox] {name} ready at {url}", flush=True)
            return
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"{name} exited early with code {proc.returncode}")
        time.sleep(0.3)
    raise TimeoutError(f"Timed out waiting for {name} at {url}")


def parse_args() -> argparse.Namespace:
    """Parse launcher flags."""
    parser = argparse.ArgumentParser(description="Launch LocalVox offline desktop stack.")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run Next.js in hot-reload dev mode instead of production next start.",
    )
    return parser.parse_args()


def main() -> int:
    """Start local Ollama, FastAPI backend, and Next.js UI and supervise cleanly."""
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    frontend_dir = root / "frontend"
    cfg = get_config()

    ollama_url = f"{cfg.llm.base_url.rstrip('/')}/api/tags"
    api_host = cfg.server.host
    api_port = cfg.server.port
    api_url = f"http://{api_host}:{api_port}/api/health"
    ui_url = "http://127.0.0.1:3000"

    children: list[tuple[str, subprocess.Popen[bytes]]] = []

    def shutdown_all(_signum: int | None = None, _frame: object = None) -> None:
        for name, proc in reversed(children):
            if proc.poll() is None:
                print(f"[LocalVox] Stopping {name}...", flush=True)
                proc.terminate()
        deadline = time.monotonic() + 5.0
        for _, proc in reversed(children):
            remaining = max(0.1, deadline - time.monotonic())
            if proc.poll() is None:
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    proc.kill()

    signal.signal(signal.SIGINT, shutdown_all)
    signal.signal(signal.SIGTERM, shutdown_all)

    try:
        if _http_ready(ollama_url):
            print(f"[LocalVox] Reusing active Ollama daemon at {cfg.llm.base_url}", flush=True)
        else:
            print("[LocalVox] Starting local Ollama daemon...", flush=True)
            ollama_proc = subprocess.Popen(["ollama", "serve"], cwd=str(root))
            children.append(("Ollama", ollama_proc))
            _wait_for_http(ollama_url, name="Ollama", proc=ollama_proc, timeout_s=20.0)

        if _http_ready(api_url):
            print(
                f"[LocalVox] Reusing active FastAPI server at http://{api_host}:{api_port}",
                flush=True,
            )
        else:
            print(
                f"[LocalVox] Starting FastAPI server at http://{api_host}:{api_port}...",
                flush=True,
            )
            env = os.environ.copy()
            env["HF_HUB_OFFLINE"] = "1"
            api_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "backend.app.main:app",
                    "--host",
                    api_host,
                    "--port",
                    str(api_port),
                ],
                cwd=str(root),
                env=env,
            )
            children.append(("FastAPI", api_proc))
            _wait_for_http(api_url, name="FastAPI", proc=api_proc, timeout_s=30.0)

        if _http_ready(ui_url):
            print(f"[LocalVox] Reusing active Next.js UI at {ui_url}", flush=True)
        else:
            if not args.dev and not (frontend_dir / ".next" / "BUILD_ID").exists():
                print("[LocalVox] Building Next.js production bundle...", flush=True)
                subprocess.run(["npm", "run", "build"], cwd=str(frontend_dir), check=True)
            ui_cmd = ["npm", "run", "dev" if args.dev else "start"]
            print(f"[LocalVox] Starting Next.js UI at {ui_url}...", flush=True)
            ui_proc = subprocess.Popen(ui_cmd, cwd=str(frontend_dir))
            children.append(("Next.js UI", ui_proc))
            _wait_for_http(ui_url, name="Next.js UI", proc=ui_proc, timeout_s=30.0)

        print(f"[LocalVox] Desktop Companion ready at {ui_url}", flush=True)

        if not children:
            print("[LocalVox] All services were already running externally.", flush=True)
            return 0

        while True:
            for name, proc in children:
                code = proc.poll()
                if code is not None:
                    print(f"[LocalVox] {name} exited with status {code}.", flush=True)
                    shutdown_all()
                    return code
            time.sleep(0.5)
    except Exception as exc:
        print(f"[LocalVox] Startup failed: {exc}", file=sys.stderr, flush=True)
        shutdown_all()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
