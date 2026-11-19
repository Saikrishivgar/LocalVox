"""Local SQLite + FTS5 persistent conversational memory package for LocalVox."""

from backend.memory.commands import MemoryCommandHandler, MemoryCommandResult
from backend.memory.database import MemoryDatabase
from backend.memory.extractor import AsyncMemoryWorker, ExtractionOutcome, MemoryExtractor
from backend.memory.manager import MemoryManager
from backend.memory.models import (
    MemoryCandidate,
    MemoryCategory,
    MemoryConfirmationRequiredError,
    MemoryDatabaseError,
    MemoryError,
    MemoryNotFoundError,
    MemoryPrivacyError,
    MemoryRecord,
    MemoryRetrievalMetrics,
    MemorySearchResult,
    MemorySource,
)
from backend.memory.privacy import contains_sensitive_data, validate_memory_privacy
from backend.memory.retriever import FtsMemoryRetriever, MemoryRetriever
from backend.memory.store import MemoryStore

__all__ = [
    "AsyncMemoryWorker",
    "ExtractionOutcome",
    "FtsMemoryRetriever",
    "MemoryCandidate",
    "MemoryCategory",
    "MemoryCommandHandler",
    "MemoryCommandResult",
    "MemoryConfirmationRequiredError",
    "MemoryDatabase",
    "MemoryDatabaseError",
    "MemoryError",
    "MemoryExtractor",
    "MemoryManager",
    "MemoryNotFoundError",
    "MemoryPrivacyError",
    "MemoryRecord",
    "MemoryRetrievalMetrics",
    "MemoryRetriever",
    "MemorySearchResult",
    "MemorySource",
    "MemoryStore",
    "contains_sensitive_data",
    "validate_memory_privacy",
]
