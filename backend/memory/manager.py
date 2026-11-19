"""Unified local memory manager facade independent of audio/voice engines."""

import threading
import uuid
from collections.abc import Sequence
from typing import Any

from backend.config import AppConfig, get_config
from backend.llm.ollama_client import OllamaClient
from backend.logger import get_logger
from backend.memory.commands import MemoryCommandHandler, MemoryCommandResult
from backend.memory.database import MemoryDatabase
from backend.memory.extractor import AsyncMemoryWorker, ExtractionOutcome, MemoryExtractor
from backend.memory.models import (
    MemoryCategory,
    MemoryRecord,
    MemoryRetrievalMetrics,
    MemorySearchResult,
)
from backend.memory.retriever import FtsMemoryRetriever
from backend.memory.store import MemoryStore


class MemoryManager:
    """High-level coordinator owning database lifecycle, retrieval, extraction, and commands."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        ollama: OllamaClient | None = None,
        foreground_busy: threading.Event | None = None,
        session_id: str | None = None,
    ) -> None:
        self._config = config if config is not None else get_config()
        self._logger = get_logger("localvox.memory.manager", self._config)
        self._session_id = session_id or uuid.uuid4().hex[:12]
        db_path = self._config.memory.db_path or self._config.paths.memory_db_file
        self._db = MemoryDatabase(db_path)
        self._store = MemoryStore(self._db, self._config)
        self._retriever = FtsMemoryRetriever(self._store, self._config)
        self._ollama = ollama if ollama is not None else OllamaClient(self._config)
        self._extractor = MemoryExtractor(self._ollama, self._store, self._config)
        self._worker = AsyncMemoryWorker(
            self._extractor,
            self._config,
            foreground_busy=foreground_busy,
        )
        self._commands = MemoryCommandHandler(self._store, self._retriever, self._config)

    @property
    def session_id(self) -> str:
        """Return unique identifier for this LocalVox session."""
        return self._session_id

    @property
    def database(self) -> MemoryDatabase:
        """Return local SQLite database manager."""
        return self._db

    @property
    def store(self) -> MemoryStore:
        """Return structured memory repository."""
        return self._store

    @property
    def retriever(self) -> FtsMemoryRetriever:
        """Return FTS5 memory retriever."""
        return self._retriever

    @property
    def extractor(self) -> MemoryExtractor:
        """Return Qwen3:8B memory extractor."""
        return self._extractor

    @property
    def worker(self) -> AsyncMemoryWorker:
        """Return background extraction worker."""
        return self._worker

    @property
    def commands(self) -> MemoryCommandHandler:
        """Return memory management command router."""
        return self._commands

    def handle_command(
        self,
        transcript: str,
        *,
        recent_turns: Sequence[tuple[str, str]] = (),
    ) -> MemoryCommandResult | None:
        """Handle explicit memory inspect/remember/update/forget/clear commands."""
        if not self._config.memory.enabled:
            return None
        return self._commands.handle(
            transcript,
            recent_turns=recent_turns,
            session_id=self._session_id,
        )

    def retrieve_context(
        self,
        query: str,
        *,
        limit: int | None = None,
        max_chars: int | None = None,
    ) -> tuple[list[MemorySearchResult], str | None, MemoryRetrievalMetrics]:
        """Retrieve ranked persistent memories formatted concisely for LLM prompt context."""
        if not self._config.memory.enabled:
            return [], None, MemoryRetrievalMetrics()
        return self._retriever.retrieve_for_context(query, limit=limit, max_chars=max_chars)

    def extract_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        recent_turns: Sequence[tuple[str, str]] = (),
    ) -> ExtractionOutcome | None:
        """Extract and store memories after a conversation turn (async by default)."""
        if not self._config.memory.enabled or not self._config.memory.auto_extract:
            return None
        if self._config.memory.async_extraction:
            self._worker.enqueue(
                user_text,
                assistant_text,
                recent_turns=recent_turns,
                session_id=self._session_id,
            )
            return None
        return self._extractor.extract_and_store(
            user_text,
            assistant_text,
            recent_turns=recent_turns,
            session_id=self._session_id,
        )

    def list_memories(
        self,
        *,
        category: MemoryCategory | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        """Return stored persistent memories directly from SQLite."""
        return self._store.list_memories(category=category, limit=limit)

    def check_integrity(self) -> dict[str, Any]:
        """Run SQLite and FTS5 integrity verification."""
        return self._db.check_integrity()

    def shutdown(self, *, wait: bool = True, timeout_s: float = 5.0) -> None:
        """Drain background extraction worker, checkpoint WAL, and close database cleanly."""
        self._worker.shutdown(wait=wait, timeout_s=timeout_s)
        self._db.close()
