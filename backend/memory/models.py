"""Structured memory models, categories, sources, exceptions, and normalization helpers."""

import hashlib
import re
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MemoryCategory(StrEnum):
    """Eight structured long-term conversational memory categories."""

    PERSONAL_PREFERENCE = "personal_preference"
    USER_FACT = "user_fact"
    INTEREST_HOBBY = "interest_hobby"
    ONGOING_PROJECT = "ongoing_project"
    IMPORTANT_DECISION = "important_decision"
    LEARNING_PROGRESS = "learning_progress"
    CONVERSATION_EVENT = "conversation_event"
    PENDING_TASK = "pending_task"


class MemorySource(StrEnum):
    """Provenance distinguishing user-stated facts from model inferences."""

    USER_EXPLICIT = "user_explicit"
    USER_STATED = "user_stated"
    ASSISTANT_INFERRED = "assistant_inferred"


class MemoryError(RuntimeError):
    """Base error for persistent local memory operations."""


class MemoryDatabaseError(MemoryError):
    """Raised when SQLite storage, migration, integrity, or write/delete fails."""


class MemoryNotFoundError(MemoryError):
    """Raised when a targeted memory record does not exist."""


class MemoryPrivacyError(MemoryError):
    """Raised when text contains excluded secrets, passwords, or credentials."""


class MemoryConfirmationRequiredError(MemoryError):
    """Raised when a destructive bulk operation is attempted without confirmation."""


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


_NORM_RE = re.compile(r"[^\w\u0B80-\u0BFF]+", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize text to lowercase space-separated alphanumeric and Tamil tokens."""
    cleaned = _NORM_RE.sub(" ", text.lower()).strip()
    return " ".join(cleaned.split())


def normalize_subject(subject: str) -> str:
    """Normalize entity/project/topic key for deduplication and updates."""
    return normalize_text(subject)


def compute_content_hash(category: MemoryCategory | str, content: str) -> str:
    """Compute stable SHA-256 digest over category and normalized memory content."""
    payload = f"{str(category)}:{normalize_text(content)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MemoryCandidate(BaseModel):
    """Validated candidate produced by LLM extraction or explicit user command."""

    model_config = ConfigDict(extra="forbid")

    category: MemoryCategory
    subject: str = ""
    content: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    source: MemorySource = MemorySource.USER_STATED
    confirmed_by_user: bool = True
    contradicts: bool = False
    updates_subject: str = ""
    session_id: str | None = None


class MemoryRecord(BaseModel):
    """Persisted structured long-term memory row from SQLite."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: MemoryCategory
    subject: str
    content: str
    keywords: list[str] = Field(default_factory=list)
    importance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source: MemorySource
    confirmed_by_user: bool
    normalized_hash: str
    session_id: str | None = None
    access_count: int = 0
    created_at: str
    updated_at: str
    last_accessed_at: str | None = None


class MemorySearchResult(BaseModel):
    """Ranked search hit combining FTS5 lexical match, importance, and recency."""

    model_config = ConfigDict(extra="forbid")

    record: MemoryRecord
    score: float
    lexical_score: float
    importance_score: float
    recency_score: float
    match_reason: str


class MemoryRetrievalMetrics(BaseModel):
    """Per-turn telemetry for persistent memory context injection."""

    model_config = ConfigDict(extra="forbid")

    retrieval_ms: float = 0.0
    memories_retrieved: int = 0
    context_chars: int = 0
