"""SQLite FTS5 memory retrieval, relevance ranking, and prompt context builder."""

import math
import re
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from backend.config import AppConfig, get_config
from backend.memory.models import (
    MemoryCategory,
    MemoryRecord,
    MemoryRetrievalMetrics,
    MemorySearchResult,
)
from backend.memory.store import MemoryStore, row_to_record

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0B80-\u0BFF_]+", flags=re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "am",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "can",
        "continue",
        "did",
        "do",
        "does",
        "for",
        "from",
        "have",
        "had",
        "has",
        "how",
        "i",
        "in",
        "is",
        "it",
        "let",
        "lets",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "recall",
        "remember",
        "s",
        "tell",
        "that",
        "the",
        "this",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "you",
        "your",
    }
)
_PROJECT_HINT_WORDS = frozenset({"project", "projects", "app", "build", "building", "repo", "code"})
_LEARNING_HINT_WORDS = frozenset({"study", "studying", "learn", "learning", "exam", "course"})
_TASK_HINT_WORDS = frozenset({"task", "tasks", "todo", "deadline", "remind", "reminder"})


def extract_query_tokens(query: str) -> list[str]:
    """Extract deduplicated searchable tokens while pruning conversational filler."""
    raw_tokens = _TOKEN_RE.findall(query.lower())
    filtered = [t for t in raw_tokens if len(t) >= 2 and t not in _STOPWORDS]
    chosen = filtered if filtered else [t for t in raw_tokens if len(t) >= 2]
    seen: set[str] = set()
    ordered: list[str] = []
    for tok in chosen:
        if tok not in seen:
            seen.add(tok)
            ordered.append(tok)
    return ordered


def build_fts5_or_query(tokens: Sequence[str]) -> str | None:
    """Build safe FTS5 prefix OR query string from sanitized alphanumeric tokens."""
    parts = [f'"{tok}"*' for tok in tokens if tok]
    if not parts:
        return None
    return " OR ".join(parts)


def _infer_query_categories(tokens: Sequence[str]) -> set[MemoryCategory]:
    tok_set = set(tokens)
    inferred: set[MemoryCategory] = set()
    if tok_set & _PROJECT_HINT_WORDS:
        inferred.add(MemoryCategory.ONGOING_PROJECT)
    if tok_set & _LEARNING_HINT_WORDS:
        inferred.add(MemoryCategory.LEARNING_PROGRESS)
    if tok_set & _TASK_HINT_WORDS:
        inferred.add(MemoryCategory.PENDING_TASK)
    return inferred


def _compute_recency_score(updated_at_iso: str, half_life_days: float) -> float:
    try:
        dt = datetime.fromisoformat(updated_at_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        age_days = max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400.0)
        return math.exp(-age_days / max(0.1, half_life_days))
    except ValueError:
        return 0.5


class MemoryRetriever(Protocol):
    """Extensible interface allowing FTS5 today and local embedding reranking later."""

    def search(
        self,
        query: str,
        *,
        category: MemoryCategory | None = None,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> list[MemorySearchResult]:
        """Search relevant persisted memories."""

    def retrieve_for_context(
        self,
        query: str,
        *,
        limit: int | None = None,
        max_chars: int | None = None,
    ) -> tuple[list[MemorySearchResult], str | None, MemoryRetrievalMetrics]:
        """Retrieve ranked memories formatted cleanly for prompt context."""


class FtsMemoryRetriever:
    """SQLite FTS5 retriever with keyword, category, project, recency, and importance ranking."""

    def __init__(
        self,
        store: MemoryStore,
        config: AppConfig | None = None,
    ) -> None:
        self._store = store
        self._config = config if config is not None else get_config()

    def _score_record(
        self,
        record: MemoryRecord,
        *,
        tokens: Sequence[str],
        bm25_raw: float | None = None,
        exact_phrase_bonus: float = 0.0,
        inferred_categories: set[MemoryCategory] | None = None,
        match_reason: str = "fts5",
    ) -> MemorySearchResult:
        mem_cfg = self._config.memory
        haystack = f"{record.subject} {record.content} {' '.join(record.keywords)}".lower()

        if tokens:
            hits = sum(1 for tok in tokens if tok in haystack)
            overlap_ratio = hits / float(len(tokens))
        else:
            overlap_ratio = 0.0

        if bm25_raw is not None:
            neg_bm25 = max(0.0, -bm25_raw)
            bm25_norm = neg_bm25 / (neg_bm25 + 2.0)
        else:
            bm25_norm = 0.0

        cat_bonus = 0.15 if inferred_categories and record.category in inferred_categories else 0.0
        lexical = min(
            1.0,
            (0.45 * bm25_norm) + (0.55 * overlap_ratio) + exact_phrase_bonus + cat_bonus,
        )

        confirm_adj = 0.05 if record.confirmed_by_user else -0.08
        importance = min(1.0, max(0.0, record.importance + confirm_adj))
        recency = _compute_recency_score(record.updated_at, mem_cfg.recency_half_life_days)

        weight_sum = mem_cfg.weight_lexical + mem_cfg.weight_importance + mem_cfg.weight_recency
        denom = weight_sum if weight_sum > 0 else 1.0
        combined = (
            (mem_cfg.weight_lexical * lexical)
            + (mem_cfg.weight_importance * importance)
            + (mem_cfg.weight_recency * recency)
        ) / denom

        return MemorySearchResult(
            record=record,
            score=round(combined, 4),
            lexical_score=round(lexical, 4),
            importance_score=round(importance, 4),
            recency_score=round(recency, 4),
            match_reason=match_reason,
        )

    def search(
        self,
        query: str,
        *,
        category: MemoryCategory | None = None,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> list[MemorySearchResult]:
        """Search memories via FTS5 BM25 + keyword overlap + category/project/recency ranking."""
        mem_cfg = self._config.memory
        max_results = limit if limit is not None else mem_cfg.max_retrieved_memories
        threshold = min_score if min_score is not None else mem_cfg.min_relevance_score
        tokens = extract_query_tokens(query)
        fts_query = build_fts5_or_query(tokens)
        inferred_cats = _infer_query_categories(tokens)

        candidates: list[tuple[MemoryRecord, float | None, str]] = []
        seen_ids: set[str] = set()

        with self._store.database.transaction() as conn:
            if fts_query:
                if category is not None:
                    rows = conn.execute(
                        """
                        SELECT m.*, bm25(memories_fts, 0.0, 0.5, 2.5, 1.0, 2.0) AS rank_score
                        FROM memories_fts
                        JOIN memories m ON memories_fts.id = m.id
                        WHERE memories_fts MATCH ? AND m.category = ?
                        ORDER BY rank_score ASC
                        LIMIT ?
                        """,
                        (fts_query, category.value, max(20, max_results * 4)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT m.*, bm25(memories_fts, 0.0, 0.5, 2.5, 1.0, 2.0) AS rank_score
                        FROM memories_fts
                        JOIN memories m ON memories_fts.id = m.id
                        WHERE memories_fts MATCH ?
                        ORDER BY rank_score ASC
                        LIMIT ?
                        """,
                        (fts_query, max(20, max_results * 4)),
                    ).fetchall()
                for row in rows:
                    rec = row_to_record(row)
                    seen_ids.add(rec.id)
                    candidates.append((rec, float(row["rank_score"]), "fts5"))

            # Include category-inferred or explicit category recent rows in case query is broad.
            fallback_categories = {category} if category is not None else inferred_cats
            for cat in fallback_categories:
                extra_rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE category = ?
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (cat.value, max_results * 2),
                ).fetchall()
                for row in extra_rows:
                    rec = row_to_record(row)
                    if rec.id not in seen_ids:
                        seen_ids.add(rec.id)
                        candidates.append((rec, None, "category_hint"))

        clean_query = " ".join(query.lower().split())
        scored: list[MemorySearchResult] = []
        for rec, bm25_val, reason in candidates:
            phrase_bonus = 0.0
            if clean_query and len(clean_query) >= 4:
                if clean_query in rec.content.lower() or clean_query in rec.subject.lower():
                    phrase_bonus = 0.2
            result = self._score_record(
                rec,
                tokens=tokens,
                bm25_raw=bm25_val,
                exact_phrase_bonus=phrase_bonus,
                inferred_categories=inferred_cats,
                match_reason=reason,
            )
            if result.score >= threshold:
                scored.append(result)

        scored.sort(
            key=lambda item: (item.score, item.record.importance, item.record.updated_at),
            reverse=True,
        )
        return scored[:max_results]

    def search_exact_keywords(
        self,
        keywords: Sequence[str],
        *,
        category: MemoryCategory | None = None,
        limit: int | None = None,
    ) -> list[MemorySearchResult]:
        """Retrieve memories containing exact keyword strings in subject, keywords, or content."""
        clean_terms = [k.strip().lower() for k in keywords if k.strip()]
        if not clean_terms:
            return []
        max_results = limit if limit is not None else self._config.memory.max_retrieved_memories
        records = self._store.list_memories(category=category, limit=250)
        hits: list[MemorySearchResult] = []
        for rec in records:
            haystack = f"{rec.subject} {rec.content} {' '.join(rec.keywords)}".lower()
            matched = [term for term in clean_terms if term in haystack]
            if not matched:
                continue
            ratio = len(matched) / float(len(clean_terms))
            hits.append(
                self._score_record(
                    rec,
                    tokens=clean_terms,
                    exact_phrase_bonus=0.35 * ratio,
                    match_reason="exact_keyword",
                )
            )
        hits.sort(key=lambda item: (item.score, item.record.updated_at), reverse=True)
        return hits[:max_results]

    def filter_by_category(
        self,
        category: MemoryCategory,
        *,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        """Return memories filtered strictly to a single category."""
        return self._store.list_memories(category=category, limit=limit)

    def get_recent(
        self,
        *,
        limit: int = 10,
        category: MemoryCategory | None = None,
    ) -> list[MemoryRecord]:
        """Return most recently created or updated memories."""
        return self._store.list_memories(category=category, limit=limit)

    def get_project_memories(
        self,
        query: str = "",
        *,
        limit: int = 10,
    ) -> list[MemorySearchResult]:
        """Retrieve project-focused memories ranked against an optional query."""
        if query.strip():
            hits = self.search(
                query,
                category=MemoryCategory.ONGOING_PROJECT,
                limit=limit,
                min_score=0.1,
            )
            if hits:
                return hits
        rows = self.filter_by_category(MemoryCategory.ONGOING_PROJECT, limit=limit)
        return [
            self._score_record(
                rec,
                tokens=extract_query_tokens(query),
                inferred_categories={MemoryCategory.ONGOING_PROJECT},
                match_reason="project_filter",
            )
            for rec in rows
        ]

    def retrieve_for_context(
        self,
        query: str,
        *,
        limit: int | None = None,
        max_chars: int | None = None,
    ) -> tuple[list[MemorySearchResult], str | None, MemoryRetrievalMetrics]:
        """Select top relevant memories within budget and format concise LLM context."""
        t0 = time.perf_counter()
        mem_cfg = self._config.memory
        max_k = limit if limit is not None else mem_cfg.max_retrieved_memories
        char_budget = max_chars if max_chars is not None else mem_cfg.max_context_chars

        results = self.search(query, limit=max_k)
        if not results:
            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            return [], None, MemoryRetrievalMetrics(
                retrieval_ms=elapsed,
                memories_retrieved=0,
                context_chars=0,
            )

        header = (
            "Relevant Persistent Memories "
            "(factual prior user context; do not invent unstated details):"
        )
        lines = [header]
        selected: list[MemorySearchResult] = []
        used_chars = len(header)

        for hit in results:
            rec = hit.record
            provenance = "user-confirmed" if rec.confirmed_by_user else "inferred"
            subj_part = f"({rec.subject}) " if rec.subject else ""
            entry = f"- [{rec.category.value} | {provenance}] {subj_part}{rec.content}"
            if used_chars + len(entry) + 1 > char_budget and selected:
                break
            lines.append(entry)
            selected.append(hit)
            used_chars += len(entry) + 1

        context_str = "\n".join(lines) if selected else None
        if selected:
            self._store.mark_accessed([s.record.id for s in selected])

        elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
        metrics = MemoryRetrievalMetrics(
            retrieval_ms=elapsed,
            memories_retrieved=len(selected),
            context_chars=len(context_str) if context_str else 0,
        )
        return selected, context_str, metrics
