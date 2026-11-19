"""Persistent local SQLite conversation session and message history store."""

import json
import re
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.memory.models import MemoryDatabaseError, utc_now_iso

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES conversation_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    language TEXT,
    metrics_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conv_messages_session
    ON conversation_messages (session_id, id ASC);
CREATE INDEX IF NOT EXISTS idx_conv_sessions_updated
    ON conversation_sessions (updated_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS conversation_messages_fts USING fts5(
    session_id UNINDEXED,
    role UNINDEXED,
    content,
    content='conversation_messages',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS conv_msg_ai AFTER INSERT ON conversation_messages BEGIN
    INSERT INTO conversation_messages_fts(rowid, session_id, role, content)
    VALUES (new.id, new.session_id, new.role, new.content);
END;

CREATE TRIGGER IF NOT EXISTS conv_msg_ad AFTER DELETE ON conversation_messages BEGIN
    INSERT INTO conversation_messages_fts(
        conversation_messages_fts, rowid, session_id, role, content
    )
    VALUES ('delete', old.id, old.session_id, old.role, old.content);
END;
"""
_SEARCH_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0B80-\u0BFF_]+", flags=re.UNICODE)


class ConversationMessageRecord(BaseModel):
    """Persisted single user or assistant turn message."""

    model_config = ConfigDict(extra="forbid")

    id: int
    session_id: str
    role: str
    content: str
    language: str | None = None
    metrics: dict[str, Any] | None = None
    created_at: str


class ConversationSessionSummary(BaseModel):
    """Metadata summary for a persisted conversation session."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    created_at: str
    updated_at: str
    turn_count: int
    preview: str = ""


class ConversationSessionDetail(BaseModel):
    """Complete conversation session with ordered messages."""

    model_config = ConfigDict(extra="forbid")

    session: ConversationSessionSummary
    messages: list[ConversationMessageRecord]


class ConversationHistoryStore:
    """Thread-safe local SQLite store for real conversation transcripts across restarts."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._closed = False
        self._conn: sqlite3.Connection | None = None
        self.initialize()

    def _open_connection(self) -> sqlite3.Connection:
        if str(self._db_path) != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        if str(self._db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def initialize(self) -> None:
        """Create schema and FTS5 index idempotently."""
        with self._lock:
            try:
                if self._conn is None or self._closed:
                    self._conn = self._open_connection()
                    self._closed = False
                self._conn.executescript(_SCHEMA_SQL)
                self._conn.commit()
            except sqlite3.Error as exc:
                raise MemoryDatabaseError(
                    f"Failed to initialize conversation history DB at {self._db_path}: {exc}"
                ) from exc

    def _tx(self) -> sqlite3.Connection:
        if self._closed or self._conn is None:
            raise MemoryDatabaseError("Conversation history database is closed.")
        return self._conn

    @staticmethod
    def _derive_title(text: str) -> str:
        clean = " ".join(text.strip().split())
        if len(clean) <= 58:
            return clean or "New Conversation"
        return clean[:55].rstrip() + "..."

    def create_session(
        self,
        *,
        session_id: str | None = None,
        title: str = "New Conversation",
    ) -> ConversationSessionSummary:
        """Create a new persistent conversation session."""
        sid = session_id or uuid.uuid4().hex[:16]
        now = utc_now_iso()
        with self._lock:
            conn = self._tx()
            conn.execute(
                """
                INSERT OR IGNORE INTO conversation_sessions
                (id, title, created_at, updated_at, turn_count)
                VALUES (?, ?, ?, ?, 0)
                """,
                (sid, title.strip() or "New Conversation", now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM conversation_sessions WHERE id = ?",
                (sid,),
            ).fetchone()
            return ConversationSessionSummary(
                id=str(row["id"]),
                title=str(row["title"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                turn_count=int(row["turn_count"]),
                preview="",
            )

    def record_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        *,
        language: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> ConversationSessionSummary:
        """Append one real user+assistant exchange to session history atomically."""
        clean_user = user_text.strip()
        clean_assistant = assistant_text.strip()
        now = utc_now_iso()
        metrics_raw = json.dumps(metrics, ensure_ascii=False) if metrics else None

        with self._lock:
            conn = self._tx()
            existing = conn.execute(
                "SELECT * FROM conversation_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            title = self._derive_title(clean_user)
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO conversation_sessions
                    (id, title, created_at, updated_at, turn_count)
                    VALUES (?, ?, ?, ?, 0)
                    """,
                    (session_id, title, now, now),
                )
            elif str(existing["title"]) == "New Conversation" and clean_user:
                conn.execute(
                    "UPDATE conversation_sessions SET title = ? WHERE id = ?",
                    (title, session_id),
                )

            if clean_user:
                conn.execute(
                    """
                    INSERT INTO conversation_messages
                    (session_id, role, content, language, metrics_json, created_at)
                    VALUES (?, 'user', ?, ?, NULL, ?)
                    """,
                    (session_id, clean_user, language, now),
                )
            if clean_assistant:
                conn.execute(
                    """
                    INSERT INTO conversation_messages
                    (session_id, role, content, language, metrics_json, created_at)
                    VALUES (?, 'assistant', ?, ?, ?, ?)
                    """,
                    (session_id, clean_assistant, language, metrics_raw, now),
                )
            conn.execute(
                """
                UPDATE conversation_sessions
                SET turn_count = turn_count + 1, updated_at = ?
                WHERE id = ?
                """,
                (now, session_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM conversation_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return ConversationSessionSummary(
                id=str(row["id"]),
                title=str(row["title"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                turn_count=int(row["turn_count"]),
                preview=clean_assistant[:120] or clean_user[:120],
            )

    def list_sessions(
        self,
        *,
        query: str = "",
        limit: int = 50,
    ) -> list[ConversationSessionSummary]:
        """List conversation sessions ordered newest-first, optionally filtered by query."""
        tokens = _SEARCH_TOKEN_RE.findall(query.lower())
        with self._lock:
            conn = self._tx()
            if tokens:
                fts_q = " OR ".join(f'"{t}"*' for t in tokens[:10])
                like_q = f"%{query.strip()}%"
                rows = conn.execute(
                    """
                    SELECT DISTINCT s.*
                    FROM conversation_sessions s
                    LEFT JOIN conversation_messages m ON m.session_id = s.id
                    WHERE s.title LIKE ?
                       OR s.id IN (
                           SELECT session_id FROM conversation_messages_fts
                           WHERE conversation_messages_fts MATCH ?
                       )
                    ORDER BY s.updated_at DESC
                    LIMIT ?
                    """,
                    (like_q, fts_q, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM conversation_sessions
                    WHERE turn_count > 0
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            results: list[ConversationSessionSummary] = []
            for row in rows:
                sid = str(row["id"])
                last_msg = conn.execute(
                    """
                    SELECT content FROM conversation_messages
                    WHERE session_id = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (sid,),
                ).fetchone()
                preview = str(last_msg["content"])[:120] if last_msg is not None else ""
                results.append(
                    ConversationSessionSummary(
                        id=sid,
                        title=str(row["title"]),
                        created_at=str(row["created_at"]),
                        updated_at=str(row["updated_at"]),
                        turn_count=int(row["turn_count"]),
                        preview=preview,
                    )
                )
            return results

    def get_session(self, session_id: str) -> ConversationSessionDetail | None:
        """Fetch session metadata and complete ordered message transcript."""
        with self._lock:
            conn = self._tx()
            s_row = conn.execute(
                "SELECT * FROM conversation_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if s_row is None:
                return None
            m_rows = conn.execute(
                """
                SELECT * FROM conversation_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            messages: list[ConversationMessageRecord] = []
            for m in m_rows:
                raw_met = m["metrics_json"]
                parsed_met = json.loads(raw_met) if raw_met else None
                messages.append(
                    ConversationMessageRecord(
                        id=int(m["id"]),
                        session_id=str(m["session_id"]),
                        role=str(m["role"]),
                        content=str(m["content"]),
                        language=m["language"],
                        metrics=parsed_met,
                        created_at=str(m["created_at"]),
                    )
                )
            preview = messages[-1].content[:120] if messages else ""
            summary = ConversationSessionSummary(
                id=str(s_row["id"]),
                title=str(s_row["title"]),
                created_at=str(s_row["created_at"]),
                updated_at=str(s_row["updated_at"]),
                turn_count=int(s_row["turn_count"]),
                preview=preview,
            )
            return ConversationSessionDetail(session=summary, messages=messages)

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session and its messages permanently."""
        with self._lock:
            conn = self._tx()
            cur = conn.execute(
                "DELETE FROM conversation_sessions WHERE id = ?",
                (session_id,),
            )
            conn.commit()
            return cur.rowcount == 1

    def close(self) -> None:
        """Checkpoint WAL and close connection safely."""
        with self._lock:
            if self._conn is not None and not self._closed:
                try:
                    if str(self._db_path) != ":memory:":
                        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    self._conn.close()
                finally:
                    self._closed = True
                    self._conn = None
