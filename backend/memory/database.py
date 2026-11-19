"""SQLite database initialization, versioned migrations, concurrency lock, and integrity checks."""

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from backend.memory.models import MemoryDatabaseError, utc_now_iso

_MIGRATIONS: tuple[tuple[int, str, str], ...] = (
    (
        1,
        "init_memories_and_fts5",
        """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL CHECK (category IN (
                'personal_preference',
                'user_fact',
                'interest_hobby',
                'ongoing_project',
                'important_decision',
                'learning_progress',
                'conversation_event',
                'pending_task'
            )),
            subject TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            keywords TEXT NOT NULL DEFAULT '[]',
            importance REAL NOT NULL DEFAULT 0.5 CHECK (importance >= 0.0 AND importance <= 1.0),
            confidence REAL NOT NULL DEFAULT 0.8 CHECK (confidence >= 0.0 AND confidence <= 1.0),
            source TEXT NOT NULL CHECK (source IN (
                'user_explicit',
                'user_stated',
                'assistant_inferred'
            )),
            confirmed_by_user INTEGER NOT NULL DEFAULT 1 CHECK (confirmed_by_user IN (0, 1)),
            normalized_hash TEXT NOT NULL,
            session_id TEXT,
            access_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_accessed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_memories_category_updated
            ON memories (category, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_memories_hash
            ON memories (normalized_hash);
        CREATE INDEX IF NOT EXISTS idx_memories_subject
            ON memories (subject);
        CREATE INDEX IF NOT EXISTS idx_memories_importance
            ON memories (importance DESC, updated_at DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            id UNINDEXED,
            category,
            subject,
            content,
            keywords,
            content='memories',
            content_rowid='rowid',
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, id, category, subject, content, keywords)
            VALUES (new.rowid, new.id, new.category, new.subject, new.content, new.keywords);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, id, category, subject, content, keywords)
            VALUES (
                'delete', old.rowid, old.id, old.category, old.subject, old.content, old.keywords
            );
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, id, category, subject, content, keywords)
            VALUES (
                'delete', old.rowid, old.id, old.category, old.subject, old.content, old.keywords
            );
            INSERT INTO memories_fts(rowid, id, category, subject, content, keywords)
            VALUES (new.rowid, new.id, new.category, new.subject, new.content, new.keywords);
        END;
        """,
    ),
)


class MemoryDatabase:
    """Thread-safe local SQLite + FTS5 database manager with schema migrations."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._closed = False
        self._conn: sqlite3.Connection | None = None
        self.initialize()

    @property
    def db_path(self) -> Path:
        """Return resolved filesystem path of the SQLite database."""
        return self._db_path

    def _open_connection(self) -> sqlite3.Connection:
        if str(self._db_path) != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        if str(self._db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def initialize(self) -> None:
        """Open connection and run any unapplied forward schema migrations."""
        with self._lock:
            try:
                if self._conn is None or self._closed:
                    self._conn = self._open_connection()
                    self._closed = False
                conn = self._conn
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        applied_at TEXT NOT NULL
                    );
                    """
                )
                applied = {
                    int(row["version"])
                    for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
                }
                for version, name, ddl in _MIGRATIONS:
                    if version in applied:
                        continue
                    conn.executescript(ddl)
                    conn.execute(
                        "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?)",
                        (version, name, utc_now_iso()),
                    )
                conn.commit()
            except sqlite3.Error as exc:
                raise MemoryDatabaseError(
                    f"Failed to initialize memory database at {self._db_path}: {exc}"
                ) from exc

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield a locked atomic SQLite transaction committing or rolling back cleanly."""
        with self._lock:
            if self._closed or self._conn is None:
                raise MemoryDatabaseError("Memory database connection is closed.")
            try:
                yield self._conn
                self._conn.commit()
            except sqlite3.Error as exc:
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    pass
                raise MemoryDatabaseError(f"SQLite transaction failed: {exc}") from exc

    def applied_migrations(self) -> list[tuple[int, str, str]]:
        """Return list of (version, name, applied_at) recorded in schema_migrations."""
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT version, name, applied_at FROM schema_migrations ORDER BY version ASC"
            ).fetchall()
            return [(int(r["version"]), str(r["name"]), str(r["applied_at"])) for r in rows]

    def check_integrity(self) -> dict[str, Any]:
        """Verify SQLite integrity and FTS5 index consistency."""
        with self.transaction() as conn:
            integ_rows = conn.execute("PRAGMA integrity_check;").fetchall()
            integrity = [str(r[0]) for r in integ_rows]
            mem_count = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
            fts_ok = True
            try:
                conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('integrity-check');")
            except sqlite3.Error:
                fts_ok = False
            ok = (integrity == ["ok"]) and fts_ok
            return {
                "ok": ok,
                "integrity_check": integrity,
                "fts_ok": fts_ok,
                "memory_count": mem_count,
            }

    def close(self) -> None:
        """Checkpoint WAL and close the SQLite database connection safely."""
        with self._lock:
            if self._conn is not None and not self._closed:
                try:
                    if str(self._db_path) != ":memory:":
                        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    self._conn.close()
                except sqlite3.Error as exc:
                    raise MemoryDatabaseError(
                        f"Failed to close memory database cleanly: {exc}"
                    ) from exc
                finally:
                    self._closed = True
                    self._conn = None
