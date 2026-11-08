"""Build a heading-aware Milvus collection without modifying the existing collection."""

import argparse
from collections import Counter

from psycopg.rows import dict_row
from pymilvus import MilvusClient

from app.core.config import get_settings
from app.db.session import close_pool, get_pool
from app.retrieval.context import retrieval_text
from app.vectorstore import milvus_store

_FIELDS = [
    "chunk_id", "vector", "document_id", "filename", "version", "content_hash",
    "chunk_type", "chunk_index", "text", "metadata",
]


def _read_collection(client: MilvusClient, name: str, fields: list[str]) -> dict[str, dict]:
    iterator = client.query_iterator(name, batch_size=500, output_fields=fields)
    rows = {}
    try:
        while batch := iterator.next():
            for row in batch:
                chunk_id = row["chunk_id"]
                if chunk_id in rows:
                    raise ValueError(f"Duplicate chunk ID in {name}: {chunk_id}")
                rows[chunk_id] = row
    finally:
        iterator.close()
    return rows


def rebuild(source: str, target: str) -> int:
    if source == target:
        raise ValueError("Source and target collections must be different")
    settings = get_settings()
    client = milvus_store._connect(settings.milvus_uri)
    try:
        if not client.has_collection(source):
            raise ValueError(f"Source collection does not exist: {source}")
        if client.has_collection(target):
            raise ValueError(f"Target collection already exists: {target}")
        with get_pool().connection() as conn:
            conn.row_factory = dict_row
            documents = conn.execute(
                """SELECT d.id::text, d.filename, d.current_version, v.status, v.chunk_count
                   FROM documents d JOIN document_versions v
                     ON v.document_id = d.id AND v.version = d.current_version"""
            ).fetchall()
            if any(doc["status"] != "processed" for doc in documents):
                raise ValueError("All current document versions must be processed before rebuilding")
            chunks = conn.execute(
                """SELECT c.chunk_id, c.document_id::text, c.chunk_index, c.chunk_type,
                          c.text, c.metadata, d.filename, v.version, v.sha256 AS content_hash
                   FROM document_chunks c
                   JOIN documents d ON d.id = c.document_id
                   JOIN document_versions v ON v.id = c.version_id
                   WHERE v.version = d.current_version
                   ORDER BY c.document_id, c.chunk_index"""
            ).fetchall()
        counts = Counter(chunk["document_id"] for chunk in chunks)
        if not chunks or any(counts[doc["id"]] != doc["chunk_count"] for doc in documents):
            raise ValueError("Postgres has no chunks or its current document chunk counts do not match")
        expected = {chunk["chunk_id"]: chunk for chunk in chunks}
        if len(expected) != len(chunks):
            raise ValueError("Duplicate chunk IDs in Postgres")
        client.load_collection(source)
        old = _read_collection(client, source, _FIELDS)
        if old.keys() != expected.keys():
            raise ValueError("Source Milvus and Postgres chunk IDs do not match")
        for chunk_id, chunk in expected.items():
            for field in ("document_id", "filename", "version", "chunk_type", "chunk_index", "text", "metadata"):
                if old[chunk_id][field] != chunk[field]:
                    raise ValueError(f"Source Milvus and Postgres differ for chunk {chunk_id}, field {field}")
            if old[chunk_id]["content_hash"] != chunk["content_hash"].strip():
                raise ValueError(f"Source Milvus and Postgres content hashes differ for chunk {chunk_id}")

        settings.milvus_collection = target
        milvus_store._client.cache_clear()
        milvus_store._supports_search_text.cache_clear()
        milvus_store._ensure_collection(client)
        for start in range(0, len(chunks), 200):
            batch = chunks[start : start + 200]
            milvus_store.upsert_chunks(batch, [old[c["chunk_id"]]["vector"] for c in batch])

        new = _read_collection(client, target, _FIELDS + ["search_text"])
        if new.keys() != expected.keys():
            raise ValueError("Rebuilt collection has different chunk IDs")
        for chunk_id, chunk in expected.items():
            for field in _FIELDS:
                if new[chunk_id][field] != old[chunk_id][field]:
                    raise ValueError(f"Rebuilt collection differs for chunk {chunk_id}, field {field}")
            if new[chunk_id]["search_text"] != retrieval_text(chunk["text"], chunk["metadata"]):
                raise ValueError(f"Rebuilt search text differs for chunk {chunk_id}")
        return len(chunks)
    finally:
        client.close()
        close_pool()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Existing Milvus collection (left unchanged)")
    parser.add_argument("--target", required=True, help="New collection name (must not already exist)")
    args = parser.parse_args()
    count = rebuild(args.source, args.target)
    print(f"Verified {count} chunks in {args.target}. Source {args.source} was not changed.")
    print(f"To activate after testing, set MILVUS_COLLECTION={args.target} and restart the application.")


if __name__ == "__main__":
    main()
