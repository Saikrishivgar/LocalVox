"""Comprehensive Phase 4 automated tests for persistent SQLite memory and pipeline integration."""

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from backend.config import AppConfig
from backend.llm.ollama_client import LlmResult, OllamaClient
from backend.memory import (
    FtsMemoryRetriever,
    MemoryCandidate,
    MemoryCategory,
    MemoryCommandHandler,
    MemoryConfirmationRequiredError,
    MemoryDatabase,
    MemoryDatabaseError,
    MemoryManager,
    MemoryNotFoundError,
    MemoryPrivacyError,
    MemorySource,
    MemoryStore,
    contains_sensitive_data,
)
from backend.pipeline.audio_io import SpeechEndpointDetector
from backend.pipeline.voice_pipeline import VoicePipeline
from backend.stt.vad import SileroVadDetector
from backend.stt.whisper_stt import MlxWhisperStt, SttResult
from backend.tts.kokoro_tts import KokoroTtsEngine
from backend.tts.piper_tts import PiperTtsEngine
from backend.tts.router import TtsRouter
from backend.tts.types import TtsResult
from tests.test_phase3 import (
    LiveQueueCapture,
    StreamRecordingPlayer,
    _fake_vad,
    _make_tone_frames,
)


def _phase4_config(tmp_path: Path, *, db_name: str = "memory.db") -> AppConfig:
    """Build isolated AppConfig with Phase 4 persistent memory enabled inside tmp_path."""
    raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    db_file = str(tmp_path / db_name)
    raw["logging"]["app_log_file"] = str(tmp_path / "app.log")
    raw["logging"]["latency_log_file"] = str(tmp_path / "latency.jsonl")
    raw["paths"]["app_log_file"] = str(tmp_path / "app.log")
    raw["paths"]["latency_log_file"] = str(tmp_path / "latency.jsonl")
    raw["paths"]["memory_db_file"] = db_file
    raw["paths"]["conversations_db_file"] = str(tmp_path / "conversations.db")
    raw["memory"]["db_path"] = db_file
    raw["memory"]["enabled"] = True
    raw["memory"]["auto_extract"] = True
    raw["memory"]["async_extraction"] = True
    raw["tts"]["output_dir"] = str(tmp_path / "tts_out")
    raw["tts"]["kokoro"]["model_path"] = str(tmp_path / "kokoro.onnx")
    raw["tts"]["kokoro"]["voices_path"] = str(tmp_path / "voices.bin")
    raw["tts"]["piper"]["en_model_path"] = str(tmp_path / "en.onnx")
    raw["tts"]["piper"]["en_config_path"] = str(tmp_path / "en.onnx.json")
    raw["tts"]["piper"]["ta_model_path"] = str(tmp_path / "ta.onnx")
    raw["tts"]["piper"]["ta_config_path"] = str(tmp_path / "ta.onnx.json")
    raw["pipeline"]["pre_speech_pad_ms"] = 64
    raw["pipeline"]["post_speech_pad_ms"] = 64
    raw["pipeline"]["vad_hangover_ms"] = 128
    raw["pipeline"]["min_utterance_ms"] = 96
    raw["vad"]["min_speech_duration_ms"] = 64
    raw["pipeline"]["barge_in_enabled"] = False
    raw["pipeline"]["tts_chunk_gap_ms"] = 0
    return AppConfig.model_validate(raw)


def test_db_initialization_migrations_and_integrity(tmp_path: Path) -> None:
    """Cases 1 & 12 partial: DB creates schema, FTS5, idempotent migrations & WAL cleanup."""
    db_path = tmp_path / "nested" / "localvox.db"
    db = MemoryDatabase(db_path)
    migrations = db.applied_migrations()
    assert len(migrations) == 1
    assert migrations[0][0] == 1
    assert migrations[0][1] == "init_memories_and_fts5"

    integrity = db.check_integrity()
    assert integrity["ok"] is True
    assert integrity["fts_ok"] is True
    assert integrity["memory_count"] == 0
    db.close()

    # Re-opening existing DB preserves migration idempotently.
    db2 = MemoryDatabase(db_path)
    assert len(db2.applied_migrations()) == 1
    assert db2.check_integrity()["ok"] is True
    db2.close()


def test_memory_insertion_all_categories_and_persistence(tmp_path: Path) -> None:
    """Cases 2 & 12: Insert all 8 categories, close, reopen from disk, verify persistence."""
    cfg = _phase4_config(tmp_path)
    db = MemoryDatabase(cfg.memory.db_path)
    store = MemoryStore(db, cfg)

    for idx, cat in enumerate(MemoryCategory):
        rec, action, write_ms = store.upsert(
            MemoryCandidate(
                category=cat,
                subject=f"topic_{idx}",
                content=f"Structured memory item {idx} for category {cat.value}.",
                keywords=[f"kw{idx}", cat.value],
                importance=0.5 + (idx * 0.05),
                confidence=0.9,
                source=MemorySource.USER_STATED,
                confirmed_by_user=True,
                session_id="sess-1",
            )
        )
        assert action == "inserted"
        assert write_ms >= 0.0
        assert len(rec.id) == 32
        assert rec.created_at == rec.updated_at

    assert store.count() == 8
    db.close()

    # Fresh DB handle simulating process restart.
    db_reopened = MemoryDatabase(cfg.memory.db_path)
    store_reopened = MemoryStore(db_reopened, cfg)
    assert store_reopened.count() == 8
    for cat in MemoryCategory:
        rows = store_reopened.list_memories(category=cat)
        assert len(rows) == 1
        assert rows[0].category == cat
        assert rows[0].session_id == "sess-1"
    assert db_reopened.check_integrity()["ok"] is True
    db_reopened.close()


def test_memory_update_deletion_and_fts_sync(tmp_path: Path) -> None:
    """Cases 3 & 4: Update reindexes FTS5; deletion removes record & FTS entry permanently."""
    cfg = _phase4_config(tmp_path)
    db = MemoryDatabase(cfg.memory.db_path)
    store = MemoryStore(db, cfg)
    retriever = FtsMemoryRetriever(store, cfg)

    rec, action, _ = store.upsert(
        MemoryCandidate(
            category=MemoryCategory.LEARNING_PROGRESS,
            subject="linear algebra",
            content="Currently studying eigenvalues and eigenvectors.",
            keywords=["eigenvalues", "matrix"],
            importance=0.8,
        )
    )
    assert action == "inserted"
    assert retriever.search("eigenvalues")

    updated = store.update(
        rec.id,
        content="Completed eigenvectors and started singular value decomposition.",
        keywords=["SVD", "decomposition"],
        importance=0.9,
    )
    assert updated.id == rec.id
    assert retriever.search("eigenvalues") == []
    assert len(retriever.search("singular value decomposition")) == 1

    with pytest.raises(MemoryNotFoundError):
        store.update("nonexistent_id", content="Updated")

    with pytest.raises(MemoryConfirmationRequiredError):
        store.clear_all(confirmed=False)
    assert store.count() == 1

    assert store.delete(rec.id) is True
    assert store.delete(rec.id) is False
    assert store.count() == 0
    assert retriever.search("decomposition") == []
    assert db.check_integrity()["ok"] is True
    db.close()


def test_duplicate_detection_and_contradiction_handling(tmp_path: Path) -> None:
    """Cases 5 & 10: Deduplicate repeats, enrich subsets, and resolve contradictions safely."""
    cfg = _phase4_config(tmp_path)
    db = MemoryDatabase(cfg.memory.db_path)
    store = MemoryStore(db, cfg)

    # Exact duplicate prevention.
    r1, a1, _ = store.upsert(
        MemoryCandidate(
            category=MemoryCategory.PERSONAL_PREFERENCE,
            subject="explanation style",
            content="User prefers concise Tamil examples.",
            keywords=["Tamil"],
            importance=0.7,
            confirmed_by_user=True,
        )
    )
    r2, a2, _ = store.upsert(
        MemoryCandidate(
            category=MemoryCategory.PERSONAL_PREFERENCE,
            subject="explanation style",
            content="user prefers concise tamil examples!",
            keywords=["concise"],
            importance=0.85,
            source=MemorySource.USER_EXPLICIT,
            confirmed_by_user=True,
        )
    )
    assert a1 == "inserted"
    assert a2 == "deduplicated"
    assert store.count() == 1
    assert r2.id == r1.id
    assert r2.importance == 0.85
    assert "concise" in r2.keywords

    # Unconfirmed model inference must NOT overwrite a user-confirmed preference.
    r_inf, a_inf, _ = store.upsert(
        MemoryCandidate(
            category=MemoryCategory.PERSONAL_PREFERENCE,
            subject="explanation style",
            content="User wants ultra long English lectures instead.",
            source=MemorySource.ASSISTANT_INFERRED,
            confirmed_by_user=False,
            contradicts=True,
            importance=0.6,
        )
    )
    assert a_inf == "deduplicated"
    assert "Tamil examples" in r_inf.content

    # User-stated contradiction DOES cleanly update the fact without leaving conflicting row.
    r_upd, a_upd, _ = store.upsert(
        MemoryCandidate(
            category=MemoryCategory.PERSONAL_PREFERENCE,
            subject="explanation style",
            content="User switched preference to English technical explanations only.",
            keywords=["English", "technical"],
            source=MemorySource.USER_STATED,
            confirmed_by_user=True,
            contradicts=True,
            importance=0.9,
        )
    )
    assert a_upd == "updated"
    assert r_upd.id == r1.id
    assert store.count() == 1
    assert "English technical" in store.get(r1.id).content
    db.close()


def test_fts5_search_keywords_categories_and_ranking(tmp_path: Path) -> None:
    """Cases 6, 7, 8: Verify FTS5 UTF-8/Next.js syntax safety, category filters, and ranking."""
    cfg = _phase4_config(tmp_path)
    db = MemoryDatabase(cfg.memory.db_path)
    store = MemoryStore(db, cfg)
    retriever = FtsMemoryRetriever(store, cfg)

    old_rec, _, _ = store.upsert(
        MemoryCandidate(
            category=MemoryCategory.CONVERSATION_EVENT,
            subject="old neural network note",
            content="Briefly mentioned neural networks for a small demo.",
            keywords=["neural"],
            importance=0.4,
            source=MemorySource.ASSISTANT_INFERRED,
            confirmed_by_user=False,
        )
    )
    stale_iso = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    with db.transaction() as conn:
        conn.execute("UPDATE memories SET updated_at = ? WHERE id = ?", (stale_iso, old_rec.id))

    proj_rec, _, _ = store.upsert(
        MemoryCandidate(
            category=MemoryCategory.ONGOING_PROJECT,
            subject="crimevision",
            content="Building CrimeVision using YOLO, FastAPI, and Next.js for object detection.",
            keywords=["CrimeVision", "YOLO", "FastAPI", "Next.js", "computer vision"],
            importance=0.95,
            source=MemorySource.USER_STATED,
            confirmed_by_user=True,
        )
    )
    ta_rec, _, _ = store.upsert(
        MemoryCandidate(
            category=MemoryCategory.LEARNING_PROGRESS,
            subject="tamil physics",
            content="நியூட்டனின் இயக்க விதிகள் மற்றும் ஒளிச்சேர்க்கை பாடங்களைப் படித்து வருகிறேன்.",
            keywords=["நியூட்டன்", "ஒளிச்சேர்க்கை"],
            importance=0.85,
            confirmed_by_user=True,
        )
    )

    # Punctuated & FTS reserved words ("and", "or", "Next.js") must not raise syntax errors.
    hits_punct = retriever.search("Tell me about Next.js AND FastAPI OR YOLO!")
    assert hits_punct and hits_punct[0].record.id == proj_rec.id

    # Project continuation query without naming "CrimeVision" must retrieve CrimeVision cleanly.
    hits_proj = retriever.search("Let's continue my computer vision project")
    assert hits_proj and hits_proj[0].record.id == proj_rec.id

    # Exact keyword match helper.
    kw_hits = retriever.search_exact_keywords(["FastAPI", "Next.js"])
    assert kw_hits and kw_hits[0].record.id == proj_rec.id

    # Tamil UTF-8 search via unicode61 FTS5 tokenizer.
    ta_hits = retriever.search("நியூட்டனின் இயக்க விதிகள்")
    assert ta_hits and ta_hits[0].record.id == ta_rec.id

    # Category filter & ranking compare high importance/recent vs stale inferred hit.
    proj_only = retriever.filter_by_category(MemoryCategory.ONGOING_PROJECT)
    assert [r.id for r in proj_only] == [proj_rec.id]
    mixed_hits = retriever.search("neural vision demo project", min_score=0.05)
    assert mixed_hits[0].record.id == proj_rec.id
    db.close()


def test_memory_extraction_async_nonblocking_and_trivial_filtering(tmp_path: Path) -> None:
    """Case 9 & 11 partial: Qwen3 extractor, multi-turn CrimeVision merge, async non-blocking."""
    cfg = _phase4_config(tmp_path)
    cfg.memory.async_extraction = True

    def fake_ollama_chat(messages: list[dict[str, str]], **_kwargs: object) -> LlmResult:
        prompt = messages[-1]["content"]
        time.sleep(0.08)  # Simulate slow LLM call to assert enqueue does not block.
        if "CrimeVision" in prompt and "YOLO" not in prompt:
            payload = {
                "memories": [
                    {
                        "category": "ongoing_project",
                        "subject": "crimevision",
                        "content": "User is building a computer vision project called CrimeVision.",
                        "keywords": ["CrimeVision", "computer vision"],
                        "importance": 0.9,
                        "confidence": 0.95,
                        "source": "user_stated",
                    }
                ]
            }
        elif "YOLO" in prompt:
            payload = {
                "memories": [
                    {
                        "category": "ongoing_project",
                        "subject": "crimevision",
                        "content": (
                            "User is building a computer vision project called CrimeVision "
                            "using YOLO, FastAPI, and Next.js."
                        ),
                        "keywords": ["CrimeVision", "YOLO", "FastAPI", "Next.js"],
                        "importance": 0.95,
                        "confidence": 0.98,
                        "source": "user_stated",
                        "updates_subject": "crimevision",
                    },
                    {
                        "category": "user_fact",
                        "subject": "low value speculative",
                        "content": "Maybe user also drinks coffee.",
                        "importance": 0.4,
                        "confidence": 0.4,
                        "source": "assistant_inferred",
                    },
                ]
            }
        else:
            payload = {"memories": []}
        return LlmResult(
            model_name="qwen3:8b",
            response_text=json.dumps(payload),
            thinking_text="",
            ttft_ms=20.0,
            total_time_ms=80.0,
            load_duration_ms=None,
            prompt_tokens=30,
            completion_tokens=40,
            tokens_per_sec=25.0,
        )

    mock_llm = MagicMock(spec=OllamaClient)
    mock_llm.chat.side_effect = fake_ollama_chat

    # Session 1
    mgr1 = MemoryManager(cfg, ollama=mock_llm)
    # Trivial greetings must be skipped locally without calling Qwen3.
    mgr1.extract_turn("Hello", "Hi there!")
    mgr1.extract_turn("Thanks", "You are welcome.")
    assert mock_llm.chat.call_count == 0

    t_start = time.perf_counter()
    mgr1.extract_turn(
        "I am building a computer vision project called CrimeVision.",
        "What technologies are you using?",
    )
    enqueue_ms = (time.perf_counter() - t_start) * 1000.0
    assert enqueue_ms < 40.0  # Async enqueue returned immediately without waiting for 80ms LLM!

    assert mgr1.worker.wait_until_idle(timeout_s=2.0) is True
    assert mgr1.store.count() == 1

    # Turn 2 answers assistant clarification with prior context -> enriches CrimeVision in-place.
    mgr1.extract_turn(
        "YOLO, FastAPI and Next.js.",
        "Got it, those work well together.",
        recent_turns=[
            (
                "I am building a computer vision project called CrimeVision.",
                "What technologies are you using?",
            )
        ],
    )
    mgr1.shutdown(wait=True, timeout_s=2.0)
    assert mock_llm.chat.call_count == 2

    # Session 2 after restart retrieves complete CrimeVision context automatically.
    mgr2 = MemoryManager(cfg, ollama=mock_llm)
    hits, context_block, ret_metrics = mgr2.retrieve_context(
        "Let's continue my computer vision project."
    )
    assert ret_metrics.memories_retrieved == 1
    assert ret_metrics.retrieval_ms >= 0.0
    assert context_block is not None
    assert "CrimeVision" in context_block
    assert "YOLO, FastAPI, and Next.js" in context_block
    assert hits[0].record.confirmed_by_user is True
    # Low-confidence inferred speculation must have been rejected.
    assert mgr2.store.count() == 1
    mgr2.shutdown(wait=True, timeout_s=2.0)


def test_memory_privacy_filters_and_commands_and_failures(tmp_path: Path) -> None:
    """Cases 14 & 15 + Feature 6: Privacy guards, memory commands, & DB error honesty."""
    cfg = _phase4_config(tmp_path)
    db = MemoryDatabase(cfg.memory.db_path)
    store = MemoryStore(db, cfg)
    retriever = FtsMemoryRetriever(store, cfg)
    commands = MemoryCommandHandler(store, retriever, cfg)

    sensitive_samples = [
        "My API key is sk-livesecret1234567890abcdef",
        "My GitHub token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
        "My password is SuperSecretPassword99",
        "Use postgres://admin:topsecret@localhost:5432/appdb",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAK...",
    ]
    for sample in sensitive_samples:
        blocked, reason = contains_sensitive_data(sample)
        assert blocked is True and reason is not None
        with pytest.raises(MemoryPrivacyError):
            store.upsert(
                MemoryCandidate(
                    category=MemoryCategory.USER_FACT,
                    subject="credentials",
                    content=sample,
                )
            )
    assert store.count() == 0

    # Command: Remember secret must be refused gracefully without writing to DB.
    sec_res = commands.handle("Remember my API key is sk-testtoken1234567890abcd")
    assert sec_res is not None
    assert sec_res.action == "privacy_blocked"
    assert store.count() == 0

    # Command: Remember project + study fact, inspect, update, forget, confirmed bulk wipe.
    r_save = commands.handle("Remember that my thesis project uses Rust and SQLite")
    assert r_save is not None and r_save.action == "remember_inserted"
    assert store.count() == 1

    r_proj = commands.handle("What do you remember about my projects?")
    assert r_proj is not None and "Rust and SQLite" in r_proj.response_text

    r_all = commands.handle("What do you remember about me?")
    assert r_all is not None and "Rust and SQLite" in r_all.response_text

    r_upd = commands.handle("Update this memory to: Thesis project uses Rust, Axum, and SQLite.")
    assert r_upd is not None and "Axum" in r_upd.response_text
    assert store.count() == 1

    # Simulate DB failure on delete -> command must report failure honestly, never fake success!
    failing_store = MagicMock(wraps=store)
    failing_store.delete.side_effect = MemoryDatabaseError("disk I/O error")
    failing_cmds = MemoryCommandHandler(failing_store, retriever, cfg)
    res_fail = failing_cmds.handle("Forget thesis project")
    assert res_fail is not None and res_fail.action == "forget_failed"
    assert "failed" in res_fail.response_text.lower()
    assert store.count() == 1

    # Successful forget.
    r_fgt = commands.handle("Forget thesis project")
    assert r_fgt is not None and r_fgt.action == "forget_memory"
    assert store.count() == 0

    # Bulk destructive wipe confirmation workflow.
    commands.handle("Remember that I prefer Tamil explanations for biology.")
    commands.handle("Remember that my exam deadline is next Friday.")
    assert store.count() == 2

    prompt_confirm = commands.handle("Clear all memories")
    assert prompt_confirm is not None and prompt_confirm.requires_confirmation is True
    assert store.count() == 2  # Still intact before confirmation!

    cancelled = commands.handle("Cancel")
    assert cancelled is not None and cancelled.action == "clear_cancelled"
    assert store.count() == 2

    commands.handle("Clear all memories")
    wiped = commands.handle("Confirm clear all memories")
    assert wiped is not None and wiped.action == "clear_all"
    assert "2 stored memories" in wiped.response_text
    assert store.count() == 0
    db.close()


def test_concurrent_memory_reads_and_writes(tmp_path: Path) -> None:
    """Case 13: 8 threads concurrently insert, update, delete, and search without locking errors."""
    cfg = _phase4_config(tmp_path)
    db = MemoryDatabase(cfg.memory.db_path)
    store = MemoryStore(db, cfg)
    retriever = FtsMemoryRetriever(store, cfg)
    errors: list[Exception] = []

    def worker(tid: int) -> None:
        try:
            for j in range(10):
                rec, _, _ = store.upsert(
                    MemoryCandidate(
                        category=MemoryCategory.USER_FACT,
                        subject=f"worker_{tid:02d}_item_{j:02d}",
                        content=(
                            f"Agent worker_{tid:02d} verified module_{tid:02d}_{j:02d} "
                            f"under concurrency suite."
                        ),
                        keywords=[f"worker_{tid:02d}", f"module_{tid:02d}_{j:02d}"],
                        importance=0.7,
                    )
                )
                if j % 3 == 0:
                    store.update(
                        rec.id,
                        content=(
                            f"Agent worker_{tid:02d} updated module_{tid:02d}_{j:02d} "
                            f"under concurrency suite."
                        ),
                    )
                _ = retriever.search(f"module_{tid:02d}_{j:02d}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert errors == []
    assert store.count() == 80
    assert db.check_integrity()["ok"] is True
    db.close()


def test_voice_pipeline_cross_session_continuity_and_commands(tmp_path: Path) -> None:
    """Case 16 & Feature 5: Full VoicePipeline restart continuity + commands + non-blocking."""
    cfg = _phase4_config(tmp_path)

    def build_mock_tts() -> TtsRouter:
        kokoro = MagicMock(spec=KokoroTtsEngine)
        kokoro.synthesize_array.return_value = (
            np.full(1200, 0.05, dtype=np.float32),
            24000,
            TtsResult(
                backend="kokoro",
                model_name="kokoro",
                voice="af_heart",
                language="en",
                sample_rate=24000,
                audio_duration_s=0.05,
                load_time_ms=0.0,
                processing_time_ms=10.0,
                rtf=0.2,
            ),
        )
        piper = MagicMock(spec=PiperTtsEngine)
        return TtsRouter(cfg, kokoro=kokoro, piper=piper)

    # --- Session 1: Build CrimeVision over 2 voice turns, shut down ---
    vad1 = _fake_vad(cfg)
    cap1 = LiveQueueCapture()
    player1 = StreamRecordingPlayer(chunk_delay_s=0.005)
    stt1 = MagicMock(spec=MlxWhisperStt)
    stt1.load.return_value = 5.0
    stt1.transcribe_array.side_effect = [
        SttResult(
            text="I am building a computer vision project called CrimeVision.",
            language="en",
            audio_duration_s=0.6,
            load_time_ms=0.0,
            processing_time_ms=15.0,
            rtf=0.03,
            model_path="mock",
            segments=[],
        ),
        SttResult(
            text="YOLO, FastAPI and Next.js.",
            language="en",
            audio_duration_s=0.6,
            load_time_ms=0.0,
            processing_time_ms=15.0,
            rtf=0.03,
            model_path="mock",
            segments=[],
        ),
    ]

    llm1 = MagicMock(spec=OllamaClient)
    llm1.warmup.return_value = None

    def llm1_chat(messages: list[dict[str, str]], **kwargs: object) -> LlmResult:
        sys_msg = messages[0]["content"]
        user_msg = messages[-1]["content"]
        if "durable long-term user memories" in sys_msg:
            if "YOLO" in user_msg:
                body = json.dumps(
                    {
                        "memories": [
                            {
                                "category": "ongoing_project",
                                "subject": "crimevision",
                                "content": (
                                    "User is building a computer vision project called "
                                    "CrimeVision using YOLO, FastAPI, and Next.js."
                                ),
                                "keywords": ["CrimeVision", "YOLO", "FastAPI", "Next.js"],
                                "importance": 0.95,
                                "confidence": 0.98,
                                "source": "user_stated",
                                "updates_subject": "crimevision",
                            }
                        ]
                    }
                )
            else:
                body = json.dumps(
                    {
                        "memories": [
                            {
                                "category": "ongoing_project",
                                "subject": "crimevision",
                                "content": (
                                    "User is building a computer vision project "
                                    "called CrimeVision."
                                ),
                                "keywords": ["CrimeVision", "computer vision"],
                                "importance": 0.9,
                                "confidence": 0.95,
                                "source": "user_stated",
                            }
                        ]
                    }
                )
            return LlmResult(
                model_name="qwen3:8b",
                response_text=body,
                thinking_text="",
                ttft_ms=10.0,
                total_time_ms=25.0,
                load_duration_ms=None,
                prompt_tokens=20,
                completion_tokens=30,
                tokens_per_sec=30.0,
            )

        reply = (
            "What technologies are you using?"
            if "CrimeVision" in user_msg
            else "Great stack for real-time inference."
        )
        on_tok = kwargs.get("on_token")
        if callable(on_tok):
            on_tok(reply)
        return LlmResult(
            model_name="qwen3:8b",
            response_text=reply,
            thinking_text="",
            ttft_ms=10.0,
            total_time_ms=20.0,
            load_duration_ms=None,
            prompt_tokens=15,
            completion_tokens=8,
            tokens_per_sec=30.0,
        )

    llm1.chat.side_effect = llm1_chat
    cond1 = threading.Condition()
    seen_listen_1 = 0

    def on_state1(_p: object, curr: object, _r: str) -> None:
        nonlocal seen_listen_1
        if str(curr) == "LISTENING":
            with cond1:
                seen_listen_1 += 1
                cond1.notify_all()

    from backend.pipeline.state import PipelineStateMachine

    p1 = VoicePipeline(
        cfg,
        capture=cap1,
        player=player1,
        vad=vad1,
        stt=stt1,
        llm=llm1,
        tts_router=build_mock_tts(),
        state_machine=PipelineStateMachine(on_change=on_state1),
    )
    tone = (
        _make_tone_frames(1, amp=0.0)
        + _make_tone_frames(4, amp=0.25)
        + _make_tone_frames(5, amp=0.0)
    )

    def feed1() -> None:
        with cond1:
            assert cond1.wait_for(lambda: seen_listen_1 >= 1, timeout=2.0)
        cap1.push_chunks(list(tone))
        with cond1:
            assert cond1.wait_for(lambda: seen_listen_1 >= 2, timeout=2.0)
        cap1.push_chunks(list(tone))

    ft1 = threading.Thread(target=feed1, daemon=True)
    ft1.start()
    m1 = p1.run(max_turns=2, warmup=True)
    ft1.join(timeout=2.0)
    assert len(m1) == 2

    # --- Session 2: Completely fresh VoicePipeline after restart ---
    vad2 = _fake_vad(cfg)
    cap2 = LiveQueueCapture()
    player2 = StreamRecordingPlayer(chunk_delay_s=0.005)
    stt2 = MagicMock(spec=MlxWhisperStt)
    stt2.load.return_value = 5.0
    stt2.transcribe_array.side_effect = [
        SttResult(
            text="Let's continue my computer vision project.",
            language="en",
            audio_duration_s=0.6,
            load_time_ms=0.0,
            processing_time_ms=15.0,
            rtf=0.03,
            model_path="mock",
            segments=[],
        ),
        SttResult(
            text="What do you remember about my projects?",
            language="en",
            audio_duration_s=0.6,
            load_time_ms=0.0,
            processing_time_ms=12.0,
            rtf=0.02,
            model_path="mock",
            segments=[],
        ),
    ]

    captured_prompts: list[list[dict[str, str]]] = []
    llm2 = MagicMock(spec=OllamaClient)
    llm2.warmup.return_value = None

    def llm2_chat(messages: list[dict[str, str]], **kwargs: object) -> LlmResult:
        if "durable long-term user memories" in messages[0]["content"]:
            return LlmResult(
                model_name="qwen3:8b",
                response_text=json.dumps({"memories": []}),
                thinking_text="",
                ttft_ms=5.0,
                total_time_ms=10.0,
                load_duration_ms=None,
                prompt_tokens=20,
                completion_tokens=4,
                tokens_per_sec=30.0,
            )
        captured_prompts.append(messages)
        reply = "Are we continuing CrimeVision with YOLO, FastAPI and Next.js?"
        on_tok = kwargs.get("on_token")
        if callable(on_tok):
            on_tok(reply)
        return LlmResult(
            model_name="qwen3:8b",
            response_text=reply,
            thinking_text="",
            ttft_ms=12.0,
            total_time_ms=25.0,
            load_duration_ms=None,
            prompt_tokens=30,
            completion_tokens=12,
            tokens_per_sec=28.0,
        )

    llm2.chat.side_effect = llm2_chat
    cond2 = threading.Condition()
    seen_listen_2 = 0

    def on_state2(_p: object, curr: object, _r: str) -> None:
        nonlocal seen_listen_2
        if str(curr) == "LISTENING":
            with cond2:
                seen_listen_2 += 1
                cond2.notify_all()

    p2 = VoicePipeline(
        cfg,
        capture=cap2,
        player=player2,
        vad=vad2,
        stt=stt2,
        llm=llm2,
        tts_router=build_mock_tts(),
        state_machine=PipelineStateMachine(on_change=on_state2),
    )

    def feed2() -> None:
        with cond2:
            assert cond2.wait_for(lambda: seen_listen_2 >= 1, timeout=2.0)
        cap2.push_chunks(list(tone))
        with cond2:
            assert cond2.wait_for(lambda: seen_listen_2 >= 2, timeout=2.0)
        cap2.push_chunks(list(tone))

    ft2 = threading.Thread(target=feed2, daemon=True)
    ft2.start()
    m2 = p2.run(max_turns=2, warmup=True)
    ft2.join(timeout=2.0)

    assert len(m2) == 2
    assert m2[0].memories_retrieved >= 1
    assert m2[0].memory_context_chars > 0
    assert "CrimeVision" in captured_prompts[0][0]["content"]
    assert "YOLO, FastAPI, and Next.js" in captured_prompts[0][0]["content"]
    assert "CrimeVision" in m2[1].response_text
    assert len(captured_prompts) == 1  # Turn 2 handled by MemoryCommandHandler cleanly!

    lingering = [
        t.name
        for t in threading.enumerate()
        if t.name.startswith("localvox-") and t.is_alive()
    ]
    assert lingering == []
    assert isinstance(SileroVadDetector(cfg), SileroVadDetector)
    assert isinstance(SpeechEndpointDetector(vad2, cfg), SpeechEndpointDetector)
