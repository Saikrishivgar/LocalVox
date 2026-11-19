"""Automated Phase 5 FastAPI, WebSocket, SQLite history, security, and runtime tests."""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app.main import create_app
from backend.app.runtime import CompanionRuntime
from backend.config import AppConfig, get_config
from backend.llm.ollama_client import LlmResult, OllamaClient, OllamaUnavailableError
from backend.pipeline.voice_pipeline import VoicePipeline
from backend.stt.whisper_stt import MlxWhisperStt, SttResult
from tests.test_phase3 import (
    LiveQueueCapture,
    StreamRecordingPlayer,
    _fake_vad,
    _mock_tts_engines,
)


def _write_isolated_yaml(tmp_path: Path) -> tuple[Path, AppConfig]:
    """Create isolated YAML config inside tmp_path so tests never touch repo config or DBs."""
    raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    db_file = str(tmp_path / "memory.db")
    conv_file = str(tmp_path / "conversations.db")
    latency_file = str(tmp_path / "latency.jsonl")
    log_file = str(tmp_path / "app.log")

    raw["logging"]["app_log_file"] = log_file
    raw["logging"]["latency_log_file"] = latency_file
    raw["paths"]["app_log_file"] = log_file
    raw["paths"]["latency_log_file"] = latency_file
    raw["paths"]["memory_db_file"] = db_file
    raw["paths"]["conversations_db_file"] = conv_file
    raw["memory"]["db_path"] = db_file
    raw["memory"]["enabled"] = True
    raw["memory"]["auto_extract"] = False
    raw["memory"]["async_extraction"] = True
    raw["tts"]["output_dir"] = str(tmp_path / "tts_out")
    raw["tts"]["kokoro"]["model_path"] = str(tmp_path / "kokoro.onnx")
    raw["tts"]["kokoro"]["voices_path"] = str(tmp_path / "voices.bin")
    raw["tts"]["piper"]["en_model_path"] = str(tmp_path / "en.onnx")
    raw["tts"]["piper"]["en_config_path"] = str(tmp_path / "en.onnx.json")
    raw["tts"]["piper"]["ta_model_path"] = str(tmp_path / "ta.onnx")
    raw["tts"]["piper"]["ta_config_path"] = str(tmp_path / "ta.onnx.json")
    raw["pipeline"]["barge_in_enabled"] = False
    raw["pipeline"]["streaming_tts_enabled"] = True
    raw["pipeline"]["tts_chunk_gap_ms"] = 0
    raw["pipeline"]["listen_timeout_s"] = 0.2

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg_path, AppConfig.model_validate(raw)


def _build_runtime(
    tmp_path: Path,
    *,
    ollama_offline: bool = False,
) -> tuple[Path, CompanionRuntime, MagicMock, LiveQueueCapture, StreamRecordingPlayer]:
    cfg_path, cfg = _write_isolated_yaml(tmp_path)
    vad = _fake_vad(cfg)
    stt = MagicMock(spec=MlxWhisperStt)
    stt.load.return_value = 3.0
    stt.model_size_mb.return_value = 400.0
    stt.transcribe_array.return_value = SttResult(
        text="hello",
        language="en",
        audio_duration_s=0.4,
        load_time_ms=0.0,
        processing_time_ms=25.0,
        rtf=0.06,
        model_path="whisper-small-mlx",
        segments=[],
    )

    llm = MagicMock(spec=OllamaClient)
    if ollama_offline:
        llm.list_models.side_effect = OllamaUnavailableError("Ollama daemon offline.")
        llm.get_running_memory_mb.side_effect = OllamaUnavailableError("Ollama daemon offline.")
    else:
        llm.list_models.return_value = [
            {"name": "qwen3:8b", "size": 5200000000},
            {"name": "llama3:latest", "size": 4700000000},
        ]
        llm.get_running_memory_mb.return_value = 5120.5

    def fake_chat(
        messages: list[dict[str, str]],
        *,
        on_token: object = None,
        cancel_event: threading.Event | None = None,
        **_kwargs: object,
    ) -> LlmResult:
        del messages, cancel_event
        tokens = ["Hello ", "from ", "LocalVox."]
        for tok in tokens:
            if callable(on_token):
                on_token(tok)
        return LlmResult(
            model_name="qwen3:8b",
            response_text="".join(tokens),
            thinking_text="",
            ttft_ms=42.0,
            total_time_ms=110.0,
            load_duration_ms=0.0,
            prompt_tokens=18,
            completion_tokens=6,
            tokens_per_sec=54.5,
        )

    llm.chat.side_effect = fake_chat

    tts_router, _, _ = _mock_tts_engines(cfg)
    capture = LiveQueueCapture()
    player = StreamRecordingPlayer(chunk_delay_s=0.01)

    pipeline = VoicePipeline(
        config=cfg,
        capture=capture,
        player=player,
        vad=vad,
        stt=stt,
        llm=llm,
        tts_router=tts_router,
    )
    runtime = CompanionRuntime(config=cfg, config_path=cfg_path, pipeline=pipeline)
    get_config.cache_clear()
    return cfg_path, runtime, llm, capture, player


def test_health_state_and_cors_websocket_security(tmp_path: Path) -> None:
    """Verify localhost CORS allowlist, WebSocket origin guard, and clean empty initial state."""
    from uvicorn.config import Config

    uv_cfg = Config("backend.app.main:app")
    uv_cfg.load()
    assert uv_cfg.ws_protocol_class is not None

    _, runtime, _, _, _ = _build_runtime(tmp_path)
    app = create_app(runtime=runtime)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "pipeline_state": "IDLE",
            "mic_active": False,
        }

        state = client.get("/api/state")
        assert state.status_code == 200
        snap = state.json()
        assert snap["messages"] == []
        assert snap["mic_active"] is False

        allowed_cors = client.get(
            "/api/health",
            headers={"Origin": "http://127.0.0.1:3000"},
        )
        assert allowed_cors.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"

        blocked_cors = client.get(
            "/api/health",
            headers={"Origin": "https://evil.example"},
        )
        assert "access-control-allow-origin" not in blocked_cors.headers

        with pytest.raises(WebSocketDisconnect) as ws_exc:
            with client.websocket_connect(
                "/ws/events",
                headers={"Origin": "https://evil.example"},
            ):
                pass
        assert ws_exc.value.code == 1008


def test_chat_websocket_streaming_and_history_continuity(tmp_path: Path) -> None:
    """Verify live token streaming over WebSocket, TTS playback option, and DB restart reload."""
    _, runtime, _, _, player = _build_runtime(tmp_path)
    app = create_app(runtime=runtime)

    events: list[dict[str, object]] = []
    with TestClient(app) as client:
        # Initially zero fabricated conversations on a fresh DB.
        empty_hist = client.get("/api/conversations")
        assert empty_hist.status_code == 200
        assert empty_hist.json()["items"] == []

        with client.websocket_connect("/ws/events") as ws:
            first = ws.receive_json()
            assert first["type"] == "snapshot"

            resp = client.post(
                "/api/chat",
                json={"text": "Hi LocalVox, explain SQLite FTS5 briefly.", "speak": False},
            )
            assert resp.status_code == 200
            chat_data = resp.json()
            assert chat_data["response_text"] == "Hello from LocalVox."
            assert chat_data["metrics"]["tts_chunks"] == 0
            assert chat_data["pipeline_state"] == "IDLE"

            while True:
                ev = ws.receive_json()
                events.append(ev)
                if ev["type"] == "turn_complete":
                    break

        event_types = [str(ev["type"]) for ev in events]
        assert "transcript" in event_types
        assert "response_token" in event_types
        assert "response_complete" in event_types
        assert "turn_complete" in event_types

        # Speak=True routes response cleanly through mocked Kokoro + player.
        spoken_resp = client.post(
            "/api/chat",
            json={"text": "Say that out loud cleanly.", "speak": True},
        )
        assert spoken_resp.status_code == 200
        assert len(player.played_chunks) >= 1

        convs = client.get("/api/conversations").json()
        assert len(convs["items"]) == 1
        session_id = convs["items"][0]["id"]
        assert convs["items"][0]["turn_count"] == 2

    # Simulate fresh process restart opening the same SQLite database files.
    _, runtime_restarted, _, _, _ = _build_runtime(tmp_path)
    app_restarted = create_app(runtime=runtime_restarted)
    with TestClient(app_restarted) as client2:
        listed = client2.get("/api/conversations?query=FTS5").json()
        assert len(listed["items"]) == 1
        assert listed["items"][0]["id"] == session_id

        detail = client2.get(f"/api/conversations/{session_id}")
        assert detail.status_code == 200
        assert len(detail.json()["messages"]) == 4

        cont = client2.post(f"/api/conversations/{session_id}/continue")
        assert cont.status_code == 200
        assert cont.json()["loaded_turns"] == 2
        assert len(runtime_restarted._pipeline.conversation.turns) == 2

        new_sess = client2.post("/api/conversations/new")
        assert new_sess.status_code == 200
        assert len(runtime_restarted._pipeline.conversation.turns) == 0

        deleted = client2.delete(f"/api/conversations/{session_id}")
        assert deleted.status_code == 200
        assert client2.get(f"/api/conversations/{session_id}").status_code == 404


def test_memory_crud_search_privacy_and_confirmed_clear(tmp_path: Path) -> None:
    """Verify full Memory Dashboard CRUD, FTS5 search, secret filtering, and confirmed wipe."""
    _, runtime, _, _, _ = _build_runtime(tmp_path)
    app = create_app(runtime=runtime)
    with TestClient(app) as client:
        # Privacy filter rejects secret API keys / passwords cleanly.
        secret_resp = client.post(
            "/api/memories",
            json={
                "category": "user_fact",
                "subject": "credentials",
                "content": "My API token is sk-1234567890abcdefghijklmnop",
                "keywords": ["api"],
                "importance": 0.9,
            },
        )
        assert secret_resp.status_code == 400

        created = client.post(
            "/api/memories",
            json={
                "category": "ongoing_project",
                "subject": "LocalVox",
                "content": "Building a completely offline voice companion in Python and Next.js.",
                "keywords": ["localvox", "nextjs"],
                "importance": 0.95,
            },
        )
        assert created.status_code == 201
        mem_id = created.json()["record"]["id"]

        listed = client.get("/api/memories").json()
        assert listed["total_count"] == 1
        assert listed["category_counts"]["ongoing_project"] == 1

        search = client.get("/api/memories?query=offline%20companion").json()
        assert len(search["items"]) == 1
        assert search["items"][0]["id"] == mem_id
        assert mem_id in search["match_scores"]

        patched = client.patch(
            f"/api/memories/{mem_id}",
            json={"content": "Building Phase 5 LocalVox desktop interface offline."},
        )
        assert patched.status_code == 200
        assert "Phase 5" in patched.json()["record"]["content"]

        # Unconfirmed bulk clear rejected by schema guard.
        wrong_clear = client.post(
            "/api/memories/clear",
            json={"confirm_phrase": "DELETE_EVERYTHING"},
        )
        assert wrong_clear.status_code == 422
        assert client.get("/api/memories").json()["total_count"] == 1

        # Exact phrase clears all persisted records.
        confirmed_clear = client.post(
            "/api/memories/clear",
            json={"confirm_phrase": "CLEAR_ALL_MEMORIES"},
        )
        assert confirmed_clear.status_code == 200
        assert confirmed_clear.json()["deleted_count"] == 1
        assert client.get("/api/memories").json()["total_count"] == 0

        assert client.delete(f"/api/memories/{mem_id}").status_code == 404


def test_voice_control_lifecycle_preserves_memory_db(tmp_path: Path) -> None:
    """Starting and stopping mic must leave SQLite MemoryDatabase open and responsive."""
    _, runtime, _, _, _ = _build_runtime(tmp_path)
    app = create_app(runtime=runtime)
    with TestClient(app) as client:
        started = client.post("/api/voice/start")
        assert started.status_code == 200
        assert started.json()["mic_active"] is True

        again = client.post("/api/voice/start")
        assert again.status_code == 200
        assert again.json()["status"] == "already_running"

        cancel_speech = client.post("/api/voice/stop-speaking")
        assert cancel_speech.status_code == 200
        assert cancel_speech.json()["status"] == "cancelled"

        stopped = client.post("/api/voice/stop")
        assert stopped.status_code == 200
        assert stopped.json()["mic_active"] is False

        # SQLite memory stays active and usable after stopping microphone loop.
        mem = client.post(
            "/api/memories",
            json={
                "category": "personal_preference",
                "subject": "theme",
                "content": "User prefers obsidian dark glass theme.",
            },
        )
        assert mem.status_code == 201


def test_system_monitor_real_telemetry_online_and_offline(tmp_path: Path) -> None:
    """System monitor reflects real psutil/SQLite/latency logs and honest Ollama state."""
    offline_dir = tmp_path / "offline"
    offline_dir.mkdir()
    _, offline_rt, _, _, _ = _build_runtime(offline_dir, ollama_offline=True)
    # Seed one genuine latency JSONL entry.
    latency_path = offline_rt.config.logging.latency_log_file
    latency_path.parent.mkdir(parents=True, exist_ok=True)
    latency_path.write_text(
        json.dumps({"stage": "ollama_chat", "latency_ms": 115.2}) + "\n",
        encoding="utf-8",
    )

    with TestClient(create_app(runtime=offline_rt)) as client:
        sys_off = client.get("/api/system")
        assert sys_off.status_code == 200
        data_off = sys_off.json()
        assert data_off["ollama_connected"] is False
        assert data_off["qwen_available"] is False
        assert data_off["ram_total_gb"] > 0
        assert data_off["process_rss_mb"] > 0
        assert data_off["sqlite_integrity"]["ok"] is True
        assert len(data_off["recent_latencies"]) == 1
        assert data_off["recent_latencies"][0]["stage"] == "ollama_chat"
        assert any(
            e["name"] == "Ollama Qwen LLM" and e["status"] == "offline"
            for e in data_off["engines"]
        )

    online_dir = tmp_path / "online"
    online_dir.mkdir()
    _, online_rt, _, _, _ = _build_runtime(online_dir, ollama_offline=False)
    with TestClient(create_app(runtime=online_rt)) as client2:
        data_on = client2.get("/api/system").json()
        assert data_on["ollama_connected"] is True
        assert data_on["qwen_available"] is True
        assert data_on["available_ollama_models"] == ["qwen3:8b", "llama3:latest"]
        llm_eng = next(e for e in data_on["engines"] if e["name"] == "Ollama Qwen LLM")
        assert llm_eng["status"] == "loaded"
        assert llm_eng["memory_mb"] == 5120.5


def test_settings_validation_persistence_and_path_protection(tmp_path: Path) -> None:
    """Allowlisted settings persist to YAML safely while rejecting arbitrary paths."""
    cfg_path, runtime, _, _, _ = _build_runtime(tmp_path)
    before_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    with TestClient(create_app(runtime=runtime)) as client:
        cur = client.get("/api/settings")
        assert cur.status_code == 200
        assert cur.json()["available_ollama_models"] == ["qwen3:8b", "llama3:latest"]
        settings = cur.json()["settings"]

        # Attempt injecting extra unexpected field (e.g. filesystem path) -> 422.
        tampered = {**settings, "db_path": "/etc/passwd"}
        assert client.put("/api/settings", json=tampered).status_code == 422

        updated_payload = {
            **settings,
            "kokoro_speed": 1.15,
            "barge_in_sensitivity": "high",
            "llm_temperature": 0.4,
            "max_history_turns": 15,
            "memory_auto_extract": True,
        }
        put_res = client.put("/api/settings", json=updated_payload)
        assert put_res.status_code == 200
        assert put_res.json()["settings"]["kokoro_speed"] == 1.15
        assert put_res.json()["settings"]["barge_in_sensitivity"] == "high"

    after_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert after_raw["tts"]["kokoro"]["speed"] == 1.15
    assert after_raw["pipeline"]["barge_in_sensitivity"] == "high"
    assert after_raw["llm"]["temperature"] == 0.4
    assert after_raw["pipeline"]["max_history_turns"] == 15
    assert after_raw["memory"]["auto_extract"] is True
    # Paths on disk remain strictly unmodified.
    assert after_raw["paths"] == before_raw["paths"]
