"""Local Qwen3:8B conversational memory extractor and non-blocking background worker."""

import json
import queue
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from backend.config import AppConfig, get_config
from backend.llm.ollama_client import OllamaClient, OllamaError
from backend.logger import get_logger, stage_timer
from backend.memory.models import (
    MemoryCandidate,
    MemoryCategory,
    MemoryPrivacyError,
    MemoryRecord,
    MemorySource,
)
from backend.memory.privacy import contains_sensitive_data
from backend.memory.store import MemoryStore

_TRIVIAL_UTTERANCES = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "ok",
        "okay",
        "thanks",
        "thank you",
        "thanks a lot",
        "yes",
        "no",
        "yeah",
        "yep",
        "nope",
        "sure",
        "cool",
        "nice",
        "great",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "bye",
        "goodbye",
        "hmm",
        "uh huh",
    }
)
_DURABLE_CUE_RE = re.compile(
    r"\b(?:i\s+am|i'm|im|my|i\s+like|i\s+prefer|i\s+love|i\s+hate|i\s+use|"
    r"i\s+want|i\s+need|i\s+decided|we\s+decided|building|working\s+on|"
    r"studying|learning|preparing|exam|deadline|project|todo|remind|switched|"
    r"actually|instead|call\s+it|called)\b",
    flags=re.IGNORECASE,
)
_EXTRACTION_SYSTEM_PROMPT = """You extract durable long-term user memories for a local
study assistant. Output ONLY a strict JSON object: {"memories": [ ... ]}.

Allowed category values:
- "personal_preference" (tone, language, study format, tool preferences)
- "user_fact" (background, role, location, durable personal details stated by user)
- "interest_hobby" (recurring topics/interests/hobbies)
- "ongoing_project" (projects being built, stack, architecture, goals)
- "important_decision" (explicit technical or study decisions)
- "learning_progress" (subjects, completed topics, weak areas, exam goals)
- "conversation_event" (notable milestones or events explicitly discussed)
- "pending_task" (commitments, assignments, follow-ups to finish later)

STRICT RULES:
1. Extract ONLY durable facts worth remembering across future sessions.
2. Ignore greetings, filler, general trivia answers, temporary states ("I'm tired"), or guesses.
3. NEVER extract passwords, PINs, API keys, tokens, secret keys, SSNs, or card numbers.
4. Combine context when user answers a clarifying question (e.g., naming project tech stack).
5. Set "source" to "user_stated" ONLY if confirmed by user; else "assistant_inferred" (<=0.65).
6. If updating/correcting an existing memory, set "updates_subject" and "contradicts" flag.
7. Importance is 0.0..1.0 (skip <0.35). If nothing durable was shared, return {"memories": []}.

Each memory object schema:
{
  "category": "...",
  "subject": "short canonical entity or topic key (lowercase words)",
  "content": "concise factual standalone statement",
  "keywords": ["tag1", "tag2"],
  "importance": 0.0_to_1.0,
  "confidence": 0.0_to_1.0,
  "source": "user_stated" | "assistant_inferred",
  "contradicts": false,
  "updates_subject": ""
}"""


def is_meaningful_turn(
    user_text: str,
    *,
    has_prior_question: bool = False,
) -> bool:
    """Return True if turn potentially contains durable information worth extracting."""
    cleaned = " ".join(user_text.strip().split())
    if not cleaned:
        return False
    blocked, _ = contains_sensitive_data(cleaned)
    if blocked:
        return False

    lowered = cleaned.lower().strip(" .!?,:;")
    if lowered in _TRIVIAL_UTTERANCES:
        return False
    if _DURABLE_CUE_RE.search(cleaned):
        return True
    if has_prior_question and len(cleaned.split()) >= 2:
        return True
    # Pure factual questions without personal/project/decision cues are not permanent memories.
    if lowered.endswith("?") or lowered.startswith(
        ("what ", "why ", "how ", "when ", "where ", "who ", "explain ", "define ")
    ):
        return False
    return len(cleaned.split()) >= 5


def _parse_json_payload(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"memories": []}


@dataclass(frozen=True)
class ExtractionOutcome:
    """Summary of one memory extraction + database upsert pass."""

    candidates_parsed: int = 0
    stored_records: list[MemoryRecord] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    skipped_privacy: int = 0
    skipped_low_value: int = 0
    extraction_ms: float = 0.0
    write_ms: float = 0.0


@dataclass(frozen=True)
class _ExtractionJob:
    user_text: str
    assistant_text: str
    recent_turns: tuple[tuple[str, str], ...]
    session_id: str | None


class MemoryExtractor:
    """Extract and validate structured memories using local Qwen3:8B via Ollama."""

    def __init__(
        self,
        ollama: OllamaClient,
        store: MemoryStore,
        config: AppConfig | None = None,
    ) -> None:
        self._ollama = ollama
        self._store = store
        self._config = config if config is not None else get_config()
        self._logger = get_logger("localvox.memory.extractor", self._config)

    def extract_candidates(
        self,
        user_text: str,
        assistant_text: str,
        *,
        recent_turns: Sequence[tuple[str, str]] = (),
        existing_memories: Sequence[MemoryRecord] = (),
        session_id: str | None = None,
    ) -> tuple[list[MemoryCandidate], float, int, int]:
        """Call Qwen3:8B to extract validated non-sensitive MemoryCandidate items."""
        has_prior_question = bool(
            recent_turns and "?" in recent_turns[-1][1]
        )
        if not is_meaningful_turn(user_text, has_prior_question=has_prior_question):
            return [], 0.0, 0, 0

        existing_lines = "\n".join(
            f"- [{m.category.value}] ({m.subject}) {m.content}" for m in existing_memories[:8]
        ) or "(none)"

        recent_block = "\n".join(
            f"Prior User: {u}\nPrior Assistant: {a}" for u, a in recent_turns[-2:]
        ) or "(none)"

        user_prompt = (
            f"Existing Known Memories:\n{existing_lines}\n\n"
            f"Recent Context Turns:\n{recent_block}\n\n"
            f"Latest Turn:\nUser: {user_text.strip()}\nAssistant: {assistant_text.strip()}"
        )
        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.perf_counter()
        llm_res = self._ollama.chat(
            messages,
            think=False,
            max_tokens=self._config.memory.extraction_max_tokens,
            temperature=self._config.memory.extraction_temperature,
        )
        extraction_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        payload = _parse_json_payload(llm_res.response_text)
        raw_items = payload.get("memories", [])
        if not isinstance(raw_items, list):
            return [], extraction_ms, 0, 0

        candidates: list[MemoryCandidate] = []
        skipped_privacy = 0
        skipped_low = 0
        min_imp = self._config.memory.min_importance_to_store

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                category = MemoryCategory(str(item.get("category", "")).strip())
            except ValueError:
                skipped_low += 1
                continue

            content = " ".join(str(item.get("content", "")).strip().split())
            subject = str(item.get("subject", "")).strip()
            updates_subject = str(item.get("updates_subject", "")).strip()
            if not content:
                skipped_low += 1
                continue

            if self._config.memory.privacy_filter_enabled:
                blocked, _ = contains_sensitive_data(f"{subject} {content}")
                if blocked:
                    skipped_privacy += 1
                    continue

            try:
                importance = float(item.get("importance", 0.5))
                confidence = float(item.get("confidence", 0.8))
            except (TypeError, ValueError):
                importance = 0.5
                confidence = 0.7

            raw_source = str(item.get("source", "user_stated")).strip()
            try:
                source = MemorySource(raw_source)
            except ValueError:
                source = MemorySource.ASSISTANT_INFERRED

            confirmed = source in (MemorySource.USER_EXPLICIT, MemorySource.USER_STATED)
            if not confirmed:
                confidence = min(confidence, 0.65)
                if confidence < 0.5 or importance < 0.55:
                    skipped_low += 1
                    continue

            if importance < min_imp:
                skipped_low += 1
                continue

            raw_kw = item.get("keywords", [])
            keywords = (
                [str(k).strip() for k in raw_kw if str(k).strip()]
                if isinstance(raw_kw, list)
                else []
            )
            if self._config.memory.privacy_filter_enabled:
                keywords = [k for k in keywords if not contains_sensitive_data(k)[0]]

            candidates.append(
                MemoryCandidate(
                    category=category,
                    subject=subject,
                    content=content,
                    keywords=keywords,
                    importance=min(1.0, max(0.0, importance)),
                    confidence=min(1.0, max(0.0, confidence)),
                    source=source,
                    confirmed_by_user=confirmed,
                    contradicts=bool(item.get("contradicts", False)),
                    updates_subject=updates_subject,
                    session_id=session_id,
                )
            )

        return candidates, extraction_ms, skipped_privacy, skipped_low

    def extract_and_store(
        self,
        user_text: str,
        assistant_text: str,
        *,
        recent_turns: Sequence[tuple[str, str]] = (),
        session_id: str | None = None,
    ) -> ExtractionOutcome:
        """Extract durable candidates from a turn and persist them safely in SQLite."""
        existing = self._store.list_memories(limit=12)
        candidates, extraction_ms, skipped_priv, skipped_low = self.extract_candidates(
            user_text,
            assistant_text,
            recent_turns=recent_turns,
            existing_memories=existing,
            session_id=session_id,
        )
        records: list[MemoryRecord] = []
        actions: list[str] = []
        total_write_ms = 0.0

        with stage_timer("memory_write", log_path=self._config.logging.latency_log_file) as rec:
            for cand in candidates:
                try:
                    stored, action, w_ms = self._store.upsert(cand)
                    records.append(stored)
                    actions.append(action)
                    total_write_ms += w_ms
                except MemoryPrivacyError:
                    skipped_priv += 1
            rec["stored_count"] = len(records)

        if records:
            self._logger.info(
                "Memory extraction finished (stored=%d, extract_ms=%.1f, write_ms=%.1f)",
                len(records),
                extraction_ms,
                total_write_ms,
            )

        return ExtractionOutcome(
            candidates_parsed=len(candidates),
            stored_records=records,
            actions=actions,
            skipped_privacy=skipped_priv,
            skipped_low_value=skipped_low,
            extraction_ms=extraction_ms,
            write_ms=round(total_write_ms, 2),
        )


class AsyncMemoryWorker:
    """Non-blocking single-worker queue executing post-turn memory extraction cleanly."""

    def __init__(
        self,
        extractor: MemoryExtractor,
        config: AppConfig | None = None,
        *,
        foreground_busy: threading.Event | None = None,
    ) -> None:
        self._extractor = extractor
        self._config = config if config is not None else get_config()
        self._logger = get_logger("localvox.memory.worker", self._config)
        self._foreground_busy = foreground_busy
        self._queue: queue.Queue[_ExtractionJob | None] = queue.Queue(maxsize=32)
        self._stop_event = threading.Event()
        self._idle_event = threading.Event()
        self._idle_event.set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.completed_jobs = 0
        self.last_outcome: ExtractionOutcome | None = None
        self.errors: list[Exception] = []

    def _ensure_started(self) -> None:
        with self._lock:
            if self._stop_event.is_set():
                self._stop_event.clear()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._worker_loop,
                    name="localvox-memory-extractor",
                    daemon=True,
                )
                self._thread.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if job is None:
                self._queue.task_done()
                break
            self._idle_event.clear()
            try:
                # Wait briefly so foreground speech/LLM generation takes GPU/CPU priority.
                if self._foreground_busy is not None:
                    deadline = time.monotonic() + 1.5
                    while (
                        self._foreground_busy.is_set()
                        and not self._stop_event.is_set()
                        and time.monotonic() < deadline
                    ):
                        self._stop_event.wait(timeout=0.05)
                if self._stop_event.is_set():
                    continue
                outcome = self._extractor.extract_and_store(
                    job.user_text,
                    job.assistant_text,
                    recent_turns=job.recent_turns,
                    session_id=job.session_id,
                )
                self.last_outcome = outcome
                self.completed_jobs += 1
            except (OllamaError, Exception) as exc:
                self._logger.warning("Background memory extraction skipped cleanly: %s", exc)
                self.errors.append(exc)
            finally:
                self._queue.task_done()
                if self._queue.empty():
                    self._idle_event.set()

    def enqueue(
        self,
        user_text: str,
        assistant_text: str,
        *,
        recent_turns: Sequence[tuple[str, str]] = (),
        session_id: str | None = None,
    ) -> bool:
        """Queue a turn for background extraction without blocking voice playback."""
        self._ensure_started()
        job = _ExtractionJob(
            user_text=user_text,
            assistant_text=assistant_text,
            recent_turns=tuple(recent_turns[-2:]),
            session_id=session_id,
        )
        try:
            self._idle_event.clear()
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            self._logger.debug("Memory extraction queue full; dropping oldest queued job.")
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(job)
                return True
            except queue.Full:
                return False

    def wait_until_idle(self, timeout_s: float = 5.0) -> bool:
        """Wait for currently queued extraction jobs to finish cleanly."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._idle_event.is_set() and self._queue.empty():
                return True
            time.sleep(0.02)
        return self._idle_event.is_set() and self._queue.empty()

    def shutdown(self, *, wait: bool = True, timeout_s: float = 5.0) -> None:
        """Stop the background worker cleanly and join thread."""
        thread = self._thread
        if thread is None:
            return
        if wait:
            self.wait_until_idle(timeout_s=timeout_s)
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread.join(timeout=timeout_s)
