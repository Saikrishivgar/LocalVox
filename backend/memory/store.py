"""Structured memory repository with duplicate detection, updates, and safe deletion."""

import json
import time
import uuid
from typing import Any, Literal

from backend.config import AppConfig, get_config
from backend.memory.database import MemoryDatabase
from backend.memory.models import (
    MemoryCandidate,
    MemoryCategory,
    MemoryConfirmationRequiredError,
    MemoryDatabaseError,
    MemoryNotFoundError,
    MemoryRecord,
    MemorySource,
    compute_content_hash,
    normalize_subject,
    normalize_text,
    utc_now_iso,
)
from backend.memory.privacy import validate_memory_privacy


def row_to_record(row: Any) -> MemoryRecord:
    """Convert a sqlite3.Row from the memories table into a MemoryRecord."""
    raw_kw = row["keywords"]
    try:
        keywords = list(json.loads(raw_kw)) if raw_kw else []
    except json.JSONDecodeError:
        keywords = []
    return MemoryRecord(
        id=str(row["id"]),
        category=MemoryCategory(str(row["category"])),
        subject=str(row["subject"]),
        content=str(row["content"]),
        keywords=[str(k) for k in keywords],
        importance=float(row["importance"]),
        confidence=float(row["confidence"]),
        source=MemorySource(str(row["source"])),
        confirmed_by_user=bool(row["confirmed_by_user"]),
        normalized_hash=str(row["normalized_hash"]),
        session_id=row["session_id"],
        access_count=int(row["access_count"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        last_accessed_at=row["last_accessed_at"],
    )


def _merge_keywords(existing: list[str], incoming: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for kw in [*existing, *incoming]:
        cleaned = kw.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen[key] = cleaned
    return list(seen.values())[:20]


def _token_set(text: str) -> set[str]:
    return {t for t in normalize_text(text).split() if len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / float(len(union)) if union else 0.0


class MemoryStore:
    """Thread-safe CRUD and deduplication layer backed by local SQLite."""

    def __init__(
        self,
        database: MemoryDatabase,
        config: AppConfig | None = None,
    ) -> None:
        self._db = database
        self._config = config if config is not None else get_config()

    @property
    def database(self) -> MemoryDatabase:
        """Return underlying MemoryDatabase."""
        return self._db

    def _validate_privacy(self, *texts: str) -> None:
        if not self._config.memory.privacy_filter_enabled:
            return
        combined = " ".join(t for t in texts if t)
        validate_memory_privacy(combined, enabled=True)

    def upsert(
        self,
        candidate: MemoryCandidate,
    ) -> tuple[MemoryRecord, Literal["inserted", "updated", "deduplicated"], float]:
        """Insert or cleanly merge/update duplicate or contradictory memories."""
        content = " ".join(candidate.content.strip().split())
        if not content:
            raise ValueError("Memory content must not be empty.")
        self._validate_privacy(candidate.subject, content, " ".join(candidate.keywords))

        norm_subj = normalize_subject(candidate.updates_subject or candidate.subject)
        clean_keywords = _merge_keywords([], candidate.keywords)
        norm_hash = compute_content_hash(candidate.category, content)
        now = utc_now_iso()
        new_tokens = _token_set(content)

        t0 = time.perf_counter()
        with self._db.transaction() as conn:
            # 1. Exact duplicate lookup via normalized content hash.
            exact_row = conn.execute(
                "SELECT * FROM memories WHERE normalized_hash = ? LIMIT 1",
                (norm_hash,),
            ).fetchone()
            if exact_row is not None:
                existing = row_to_record(exact_row)
                merged_kw = _merge_keywords(existing.keywords, clean_keywords)
                confirmed = existing.confirmed_by_user or candidate.confirmed_by_user
                source = (
                    MemorySource.USER_EXPLICIT
                    if MemorySource.USER_EXPLICIT in (existing.source, candidate.source)
                    else (candidate.source if confirmed else existing.source)
                )
                importance = max(existing.importance, candidate.importance)
                confidence = max(existing.confidence, candidate.confidence)
                conn.execute(
                    """
                    UPDATE memories
                    SET keywords = ?, importance = ?, confidence = ?,
                        source = ?, confirmed_by_user = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(merged_kw, ensure_ascii=False),
                        importance,
                        confidence,
                        source.value,
                        1 if confirmed else 0,
                        now,
                        existing.id,
                    ),
                )
                refreshed = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (existing.id,)
                ).fetchone()
                write_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return row_to_record(refreshed), "deduplicated", write_ms

            # 2. Same-category + same-subject check for enrichment or contradiction resolution.
            target_existing: MemoryRecord | None = None
            if norm_subj:
                subj_row = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE category = ? AND subject = ?
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (candidate.category.value, norm_subj),
                ).fetchone()
                if subj_row is not None:
                    target_existing = row_to_record(subj_row)

            # 3. Near-duplicate token containment / Jaccard check inside same category.
            if target_existing is None:
                recent_rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE category = ?
                    ORDER BY updated_at DESC LIMIT 50
                    """,
                    (candidate.category.value,),
                ).fetchall()
                for row in recent_rows:
                    rec = row_to_record(row)
                    if norm_subj and rec.subject and norm_subj != rec.subject:
                        continue
                    old_tokens = _token_set(rec.content)
                    sim = _jaccard(old_tokens, new_tokens)
                    subset_or_superset = (
                        len(old_tokens) >= 3
                        and len(new_tokens) >= 3
                        and (old_tokens.issubset(new_tokens) or new_tokens.issubset(old_tokens))
                    )
                    if sim >= 0.85 or subset_or_superset:
                        target_existing = rec
                        break

            if target_existing is not None:
                # Never overwrite a user-confirmed memory with an unconfirmed model inference.
                if target_existing.confirmed_by_user and not candidate.confirmed_by_user:
                    write_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    return target_existing, "deduplicated", write_ms

                old_tokens = _token_set(target_existing.content)
                should_replace_text = (
                    candidate.contradicts
                    or candidate.source == MemorySource.USER_EXPLICIT
                    or (len(new_tokens) >= len(old_tokens))
                    or (not target_existing.confirmed_by_user and candidate.confirmed_by_user)
                )
                if not should_replace_text:
                    # Still merge any newly discovered keywords/importance quietly.
                    merged_kw = _merge_keywords(target_existing.keywords, clean_keywords)
                    conn.execute(
                        "UPDATE memories SET keywords = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(merged_kw, ensure_ascii=False), now, target_existing.id),
                    )
                    refreshed = conn.execute(
                        "SELECT * FROM memories WHERE id = ?", (target_existing.id,)
                    ).fetchone()
                    write_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    return row_to_record(refreshed), "deduplicated", write_ms

                new_content = content
                merged_kw = (
                    clean_keywords
                    if candidate.contradicts and clean_keywords
                    else _merge_keywords(target_existing.keywords, clean_keywords)
                )
                new_hash = compute_content_hash(candidate.category, new_content)
                new_subj = norm_subj or target_existing.subject
                new_imp = (
                    candidate.importance
                    if candidate.contradicts
                    else max(target_existing.importance, candidate.importance)
                )
                conn.execute(
                    """
                    UPDATE memories
                    SET subject = ?, content = ?, keywords = ?, importance = ?,
                        confidence = ?, source = ?, confirmed_by_user = ?,
                        normalized_hash = ?, updated_at = ?, session_id = COALESCE(?, session_id)
                    WHERE id = ?
                    """,
                    (
                        new_subj,
                        new_content,
                        json.dumps(merged_kw, ensure_ascii=False),
                        new_imp,
                        candidate.confidence,
                        candidate.source.value,
                        1 if candidate.confirmed_by_user else 0,
                        new_hash,
                        now,
                        candidate.session_id,
                        target_existing.id,
                    ),
                )
                refreshed = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (target_existing.id,)
                ).fetchone()
                write_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return row_to_record(refreshed), "updated", write_ms

            # 4. Fresh insert.
            mem_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO memories (
                    id, category, subject, content, keywords,
                    importance, confidence, source, confirmed_by_user,
                    normalized_hash, session_id, access_count,
                    created_at, updated_at, last_accessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL)
                """,
                (
                    mem_id,
                    candidate.category.value,
                    norm_subj,
                    content,
                    json.dumps(clean_keywords, ensure_ascii=False),
                    candidate.importance,
                    candidate.confidence,
                    candidate.source.value,
                    1 if candidate.confirmed_by_user else 0,
                    norm_hash,
                    candidate.session_id,
                    now,
                    now,
                ),
            )
            inserted = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
            write_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return row_to_record(inserted), "inserted", write_ms

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Fetch one memory record by unique ID."""
        with self._db.transaction() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return row_to_record(row) if row is not None else None

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        category: MemoryCategory | None = None,
        subject: str | None = None,
        keywords: list[str] | None = None,
        importance: float | None = None,
        source: MemorySource | None = None,
        confirmed_by_user: bool | None = None,
    ) -> MemoryRecord:
        """Explicitly update an existing memory record and reindex FTS5."""
        with self._db.transaction() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                raise MemoryNotFoundError(f"Memory record '{memory_id}' not found.")
            existing = row_to_record(row)

            new_category = category if category is not None else existing.category
            new_content = (
                " ".join(content.strip().split()) if content is not None else existing.content
            )
            if not new_content:
                raise ValueError("Updated memory content cannot be empty.")
            new_subject = (
                normalize_subject(subject) if subject is not None else existing.subject
            )
            new_keywords = (
                _merge_keywords([], keywords) if keywords is not None else existing.keywords
            )
            self._validate_privacy(new_subject, new_content, " ".join(new_keywords))

            new_importance = importance if importance is not None else existing.importance
            new_source = source if source is not None else existing.source
            new_confirmed = (
                confirmed_by_user if confirmed_by_user is not None else existing.confirmed_by_user
            )
            new_hash = compute_content_hash(new_category, new_content)
            now = utc_now_iso()

            cur = conn.execute(
                """
                UPDATE memories
                SET category = ?, subject = ?, content = ?, keywords = ?,
                    importance = ?, source = ?, confirmed_by_user = ?,
                    normalized_hash = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_category.value,
                    new_subject,
                    new_content,
                    json.dumps(new_keywords, ensure_ascii=False),
                    new_importance,
                    new_source.value,
                    1 if new_confirmed else 0,
                    new_hash,
                    now,
                    memory_id,
                ),
            )
            if cur.rowcount != 1:
                raise MemoryDatabaseError(
                    f"Expected 1 updated row for '{memory_id}', got {cur.rowcount}"
                )
            updated = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return row_to_record(updated)

    def delete(self, memory_id: str) -> bool:
        """Permanently delete a memory by ID; returns True only if SQLite removed 1 row."""
        with self._db.transaction() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cur.rowcount == 1

    def clear_all(self, *, confirmed: bool = False) -> int:
        """Delete all memories permanently only when explicitly confirmed."""
        if not confirmed:
            raise MemoryConfirmationRequiredError(
                "Bulk memory deletion requires explicit confirmation (confirmed=True)."
            )
        with self._db.transaction() as conn:
            total_before = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
            conn.execute("DELETE FROM memories")
            total_after = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
            if total_after != 0:
                raise MemoryDatabaseError(
                    f"Bulk memory clear incomplete: {total_after} records remained."
                )
            return total_before

    def list_memories(
        self,
        *,
        category: MemoryCategory | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        """List stored memories ordered by most recently updated."""
        with self._db.transaction() as conn:
            if category is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE category = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (category.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [row_to_record(r) for r in rows]

    def count(self, *, category: MemoryCategory | None = None) -> int:
        """Return number of persisted memories."""
        with self._db.transaction() as conn:
            if category is not None:
                row = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE category = ?",
                    (category.value,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            return int(row[0])

    def mark_accessed(self, memory_ids: list[str]) -> None:
        """Update access_count and last_accessed_at cleanly for retrieved records."""
        if not memory_ids:
            return
        now = utc_now_iso()
        with self._db.transaction() as conn:
            placeholders = ",".join("?" for _ in memory_ids)
            conn.execute(
                f"""
                UPDATE memories
                SET access_count = access_count + 1, last_accessed_at = ?
                WHERE id IN ({placeholders})
                """,
                (now, *memory_ids),
            )
