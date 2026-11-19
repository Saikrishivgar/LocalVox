"""Real offline Phase 4 SQLite + FTS5 + Qwen3:8B memory benchmark and continuity check."""

import argparse
import tempfile
import time
from pathlib import Path

from backend.config import get_config
from backend.llm.ollama_client import OllamaClient
from backend.memory import MemoryManager
from backend.pipeline.conversation import ConversationManager


def run_memory_benchmark(db_path: Path | None = None) -> int:
    """Execute real Qwen3:8B extraction, restart persistence, FTS5 retrieval, and reply checks."""
    cfg = get_config().model_copy(deep=True)
    with tempfile.TemporaryDirectory(prefix="localvox_mem_bench_") as tmp_dir:
        target_db = db_path if db_path is not None else Path(tmp_dir) / "bench_memory.db"
        cfg.memory.db_path = target_db
        cfg.paths.memory_db_file = target_db
        cfg.memory.async_extraction = False

        llm = OllamaClient(cfg)
        llm.warmup()

        print("=== Session 1: Real Qwen3:8B Extraction & Persistence ===")
        mgr1 = MemoryManager(cfg, ollama=llm, session_id="bench-session-1")
        try:
            turn1_user = "I am building a computer vision project called CrimeVision."
            turn1_assistant = "What technologies are you using?"
            out1 = mgr1.extract_turn(turn1_user, turn1_assistant)
            assert out1 is not None

            turn2_user = "YOLO, FastAPI and Next.js."
            turn2_assistant = "Those fit a real-time detection web app nicely."
            out2 = mgr1.extract_turn(
                turn2_user,
                turn2_assistant,
                recent_turns=[(turn1_user, turn1_assistant)],
            )
            assert out2 is not None

            # Also verify privacy guard blocks secrets before any LLM call/write.
            priv_res = mgr1.handle_command("Remember my API key is sk-secret1234567890abcdef")
            print(
                f"Turn 1 extract_ms={out1.extraction_ms:.2f} write_ms={out1.write_ms:.2f} "
                f"stored={len(out1.stored_records)} actions={out1.actions}"
            )
            print(
                f"Turn 2 extract_ms={out2.extraction_ms:.2f} write_ms={out2.write_ms:.2f} "
                f"stored={len(out2.stored_records)} actions={out2.actions}"
            )
            print(
                f"Stored records after Session 1: {mgr1.store.count()} | "
                f"Privacy check: {priv_res.action if priv_res else 'none'}"
            )
            for rec in mgr1.list_memories():
                print(
                    f"  - [{rec.category.value} | imp={rec.importance:.2f} | "
                    f"confirmed={rec.confirmed_by_user}] ({rec.subject}) {rec.content}"
                )
        finally:
            mgr1.shutdown(wait=True)

        print("\n=== Session 2 (Fresh Process/Manager Reopen): Retrieval & Continuity ===")
        mgr2 = MemoryManager(cfg, ollama=llm, session_id="bench-session-2")
        try:
            query = "Let's continue my computer vision project."
            hits, extra_context, ret_metrics = mgr2.retrieve_context(query)

            conv_baseline = ConversationManager(cfg.pipeline.voice_system_prompt)
            t0_base = time.perf_counter()
            base_res = llm.chat(conv_baseline.build_messages(query))
            base_wall_ms = round((time.perf_counter() - t0_base) * 1000.0, 2)

            conv_mem = ConversationManager(cfg.pipeline.voice_system_prompt)
            t0_mem = time.perf_counter()
            mem_res = llm.chat(conv_mem.build_messages(query, extra_context=extra_context))
            mem_wall_ms = round((time.perf_counter() - t0_mem) * 1000.0, 2)

            added_latency_ms = round(
                ret_metrics.retrieval_ms + max(0.0, mem_wall_ms - base_wall_ms),
                2,
            )
            print(
                f"Retrieval: latency={ret_metrics.retrieval_ms:.2f}ms "
                f"retrieved={ret_metrics.memories_retrieved} "
                f"context_chars={ret_metrics.context_chars}"
            )
            for hit in hits:
                print(
                    f"  Hit score={hit.score:.3f} (lex={hit.lexical_score:.3f}, "
                    f"imp={hit.importance_score:.3f}, rec={hit.recency_score:.3f}): "
                    f"{hit.record.content}"
                )
            print(f"Baseline reply ({base_wall_ms:.1f}ms): {base_res.response_text}")
            print(f"Memory-aware reply ({mem_wall_ms:.1f}ms): {mem_res.response_text}")
            print(
                f"Overhead: retrieval_ms={ret_metrics.retrieval_ms:.2f}ms, "
                f"delta_response_ms={round(mem_wall_ms - base_wall_ms, 2):.2f}ms, "
                f"estimated_added_ms={added_latency_ms:.2f}ms"
            )
            print("DB integrity:", mgr2.check_integrity())
        finally:
            mgr2.shutdown(wait=True)
    return 0


def main() -> int:
    """CLI entrypoint for Phase 4 local memory benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark LocalVox Phase 4 SQLite + Qwen3 memory."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Optional explicit DB path instead of temporary benchmark DB.",
    )
    args = parser.parse_args()
    return run_memory_benchmark(args.db_path)


if __name__ == "__main__":
    raise SystemExit(main())
