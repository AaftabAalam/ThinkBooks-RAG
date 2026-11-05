from psycopg.rows import tuple_row
from psycopg_pool import ConnectionPool

from app.core.config import get_settings

_pool: ConnectionPool | None = None

# documents: one row per logical document (keyed by filename) = the registry.
# document_versions: one row per distinct content (SHA-256) seen for that document.
# document_chunks: chunks of the current processed version, with stable IDs.
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY,
    filename        TEXT NOT NULL UNIQUE,
    content_type    TEXT,
    size_bytes      BIGINT NOT NULL,
    content_hash    CHAR(64) NOT NULL,
    current_version INT NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_versions (
    id            UUID PRIMARY KEY,
    document_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version       INT NOT NULL,
    sha256        CHAR(64) NOT NULL,
    stored_path   TEXT NOT NULL,
    size_bytes    BIGINT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    error         TEXT,
    chunk_count   INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at  TIMESTAMPTZ,
    UNIQUE (document_id, version)
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id     TEXT PRIMARY KEY,
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version_id   UUID NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    chunk_type   TEXT NOT NULL,
    text         TEXT NOT NULL,
    token_count  INT NOT NULL,
    metadata     JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS acl_roles TEXT[] NOT NULL DEFAULT '{public}';
"""


def _reset_connection(conn) -> None:
    conn.row_factory = tuple_row  # row factories set per-use must not leak to the next borrower


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            get_settings().database_url, min_size=1, max_size=10, reset=_reset_connection, open=True
        )
    return _pool


def init_db() -> None:
    with get_pool().connection() as conn:
        conn.execute(SCHEMA)


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
