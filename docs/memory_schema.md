# LocalVox Phase 4 Persistent Memory Schema

All persistent memories are stored locally in SQLite (`WAL` journal mode, `unicode61` FTS5 full-text index) at `paths.memory_db_file` (`data/memory/localvox_memory.db` by default). Zero cloud storage or remote databases are used.

## Database Settings

- `PRAGMA journal_mode = WAL;` — enables concurrent reads alongside atomic serialized writes
- `PRAGMA synchronous = NORMAL;` — durable commits across restarts with low write latency
- `PRAGMA foreign_keys = ON;`
- `PRAGMA busy_timeout = 5000;`

On graceful shutdown (`MemoryDatabase.close()`), LocalVox executes `PRAGMA wal_checkpoint(TRUNCATE);` before closing handles.

## Schema Migrations (`schema_migrations`)

```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
```

## Core Table (`memories`)

```sql
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
```

### Column Semantics

| Column | Type | Purpose |
| --- | --- | --- |
| `id` | `TEXT` (UUIDv4) | Stable unique record identifier |
| `category` | `TEXT` | One of the 8 supported memory categories |
| `subject` | `TEXT` | Canonical topic/entity key (e.g., `crimevision`) used for updates & contradiction resolution |
| `content` | `TEXT` | Normalized factual summary statement |
| `keywords` | `TEXT` (JSON array) | Searchable keywords / entities / tech stack tags |
| `importance` | `REAL` `[0..1]` | Priority weight used for ranking and storage gating |
| `confidence` | `REAL` `[0..1]` | Extraction certainty |
| `source` | `TEXT` | Origin: `user_explicit`, `user_stated`, or `assistant_inferred` |
| `confirmed_by_user` | `INTEGER` `0 \| 1` | Distinguishes user-confirmed facts (`1`) from model inferences (`0`) |
| `normalized_hash` | `TEXT` (SHA-256) | Hash of normalized content + category for exact duplicate detection |
| `session_id` | `TEXT \| NULL` | Originating session identifier |
| `access_count` | `INTEGER` | Count of retrievals |
| `created_at` | `TEXT` (ISO-8601 UTC) | Creation timestamp |
| `updated_at` | `TEXT` (ISO-8601 UTC) | Last modification timestamp |
| `last_accessed_at` | `TEXT \| NULL` | Last retrieval timestamp |

## Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_memories_category_updated
    ON memories (category, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_hash
    ON memories (normalized_hash);
CREATE INDEX IF NOT EXISTS idx_memories_subject
    ON memories (subject);
CREATE INDEX IF NOT EXISTS idx_memories_importance
    ON memories (importance DESC, updated_at DESC);
```

## Full-Text Search Index (`memories_fts`) & Triggers

```sql
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
    VALUES ('delete', old.rowid, old.id, old.category, old.subject, old.content, old.keywords);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, id, category, subject, content, keywords)
    VALUES ('delete', old.rowid, old.id, old.category, old.subject, old.content, old.keywords);
    INSERT INTO memories_fts(rowid, id, category, subject, content, keywords)
    VALUES (new.rowid, new.id, new.category, new.subject, new.content, new.keywords);
END;
```

Physical `DELETE FROM memories` triggers synchronized `memories_fts` removal automatically so deleted items leave neither `memories` nor `memories_fts`.
