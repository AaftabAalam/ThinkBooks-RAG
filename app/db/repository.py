import uuid
from dataclasses import dataclass

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.db.session import get_pool

# Registry outcomes
NEW = "new"
UPDATED = "updated"  # same filename, different content -> new version
UNCHANGED = "unchanged"  # same filename, same hash -> metadata refresh only
RETRY = "retry"  # same hash but the last processing attempt did not succeed


@dataclass
class Registration:
    action: str
    document_id: str
    version_id: str
    version: int


def register_version(
    *,
    filename: str,
    content_type: str | None,
    size_bytes: int,
    sha256: str,
    stored_path: str,
    acl_roles: list[str] | None = None,
) -> Registration:
    """Registry + version check, done atomically under a row lock."""
    roles = acl_roles or ["public"]
    with get_pool().connection() as conn, conn.transaction():
        conn.row_factory = dict_row
        conn.execute(
            """INSERT INTO documents (id, filename, content_type, size_bytes, content_hash, acl_roles)
               VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (filename) DO NOTHING""",
            (uuid.uuid4(), filename, content_type, size_bytes, sha256, roles),
        )
        doc = conn.execute(
            "SELECT * FROM documents WHERE filename = %s FOR UPDATE", (filename,)
        ).fetchone()
        has_versions = conn.execute(
            "SELECT 1 FROM document_versions WHERE document_id = %s LIMIT 1", (doc["id"],)
        ).fetchone()

        def new_version(number: int, action: str) -> Registration:
            vid = uuid.uuid4()
            conn.execute(
                """INSERT INTO document_versions (id, document_id, version, sha256, stored_path, size_bytes)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (vid, doc["id"], number, sha256, stored_path, size_bytes),
            )
            return Registration(action, str(doc["id"]), str(vid), number)

        if not has_versions:  # first time this filename is seen
            return new_version(1, NEW)

        if doc["content_hash"].strip() == sha256:
            conn.execute(
                """UPDATE documents SET content_type = %s, size_bytes = %s, acl_roles = %s,
                          last_seen_at = now(), updated_at = now() WHERE id = %s""",
                (content_type, size_bytes, roles, doc["id"]),
            )
            cur = conn.execute(
                "SELECT id, status FROM document_versions WHERE document_id = %s AND version = %s",
                (doc["id"], doc["current_version"]),
            ).fetchone()
            action = UNCHANGED if cur["status"] in ("processed", "processing", "pending") else RETRY
            return Registration(action, str(doc["id"]), str(cur["id"]), doc["current_version"])

        number = doc["current_version"] + 1
        conn.execute(
            """UPDATE document_versions SET status = 'superseded'
               WHERE document_id = %s AND status <> 'superseded'""",
            (doc["id"],),
        )
        conn.execute(
            """UPDATE documents SET content_hash = %s, current_version = %s, content_type = %s,
                      size_bytes = %s, acl_roles = %s, last_seen_at = now(), updated_at = now()
               WHERE id = %s""",
            (sha256, number, content_type, size_bytes, roles, doc["id"]),
        )
        return new_version(number, UPDATED)


def accessible_document_ids(roles: list[str]) -> list[str]:
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT id::text FROM documents WHERE acl_roles && %s::text[]", (roles,)
        ).fetchall()
    return [r[0] for r in rows]


def validate_sources(sources: list[dict], roles: list[str]) -> bool:
    """True if every cited document still exists at the cited version and the caller may read it."""
    if not sources:
        return False
    ids = list({s["document_id"] for s in sources})
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT id::text, current_version, acl_roles FROM documents WHERE id = ANY(%s::uuid[])",
            (ids,),
        ).fetchall()
    current = {r[0]: (r[1], set(r[2])) for r in rows}
    for s in sources:
        entry = current.get(s["document_id"])
        if entry is None or entry[0] != s["version"] or not (entry[1] & set(roles)):
            return False
    return True


def get_version(version_id: str) -> dict | None:
    with get_pool().connection() as conn:
        conn.row_factory = dict_row
        return conn.execute(
            """SELECT v.id::text, v.document_id::text, v.version, v.sha256, v.stored_path, d.filename
               FROM document_versions v JOIN documents d ON d.id = v.document_id WHERE v.id = %s""",
            (version_id,),
        ).fetchone()


def set_version_status(version_id: str, status: str, error: str | None = None) -> None:
    with get_pool().connection() as conn:
        conn.execute(
            "UPDATE document_versions SET status = %s, error = %s WHERE id = %s",
            (status, error, version_id),
        )


def set_version_stored_path(version_id: str, stored_path: str) -> None:
    with get_pool().connection() as conn:
        updated = conn.execute(
            "UPDATE document_versions SET stored_path = %s WHERE id = %s",
            (stored_path, version_id),
        )
        if updated.rowcount != 1:
            raise ValueError(f"Unknown document version {version_id}")


def replace_chunks(document_id: str, version_id: str, chunks: list[dict]) -> None:
    """Swap in the new version's chunks and mark the version processed, atomically."""
    with get_pool().connection() as conn, conn.transaction():
        conn.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO document_chunks
                   (chunk_id, document_id, version_id, chunk_index, chunk_type, text, token_count, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (c["chunk_id"], document_id, version_id, c["chunk_index"], c["chunk_type"],
                     c["text"], c["token_count"], Jsonb(c["metadata"]))
                    for c in chunks
                ],
            )
        conn.execute(
            """UPDATE document_versions SET status = 'processed', error = NULL,
                      chunk_count = %s, processed_at = now() WHERE id = %s""",
            (len(chunks), version_id),
        )


def list_documents() -> list[dict]:
    with get_pool().connection() as conn:
        conn.row_factory = dict_row
        rows = conn.execute(
            """SELECT d.id::text, d.filename, d.size_bytes, d.content_hash, d.current_version,
                      v.status, v.error, v.chunk_count, d.updated_at
               FROM documents d JOIN document_versions v
                 ON v.document_id = d.id AND v.version = d.current_version
               ORDER BY d.updated_at DESC"""
        ).fetchall()
    for r in rows:
        r["content_hash"] = r["content_hash"].strip()
        r["updated_at"] = r["updated_at"].isoformat()
    return rows


def list_chunks(document_id: str, limit: int = 100) -> list[dict]:
    with get_pool().connection() as conn:
        conn.row_factory = dict_row
        return conn.execute(
            """SELECT chunk_id, chunk_index, chunk_type, text, token_count, metadata
               FROM document_chunks WHERE document_id = %s ORDER BY chunk_index LIMIT %s""",
            (document_id, limit),
        ).fetchall()
