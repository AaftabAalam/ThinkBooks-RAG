import logging
import time
from contextlib import contextmanager
from typing import Iterator

from app.cache import redis_cache
from app.db import repository
from app.db.session import get_pool
from app.embedding.embedder import embed_texts
from app.processing.chunker import chunk_blocks
from app.processing.parser import parse_document
from app.vectorstore import milvus_store

logger = logging.getLogger(__name__)


@contextmanager
def _timed_stage(version_id: str, stage: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info(
            "Ingestion version=%s stage=%s duration_ms=%d",
            version_id,
            stage,
            int((time.perf_counter() - start) * 1000),
        )


def process_version(version_id: str) -> int:
    """Parse documents, chunk with Chonkie, and persist chunks. Returns chunk count."""
    version = repository.get_version(version_id)
    if version is None:
        raise ValueError(f"Unknown document version {version_id}")
    with get_pool().connection() as lock_conn, lock_conn.transaction():
        lock_conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            ("processing:" + version["filename"],),
        )
        version = repository.get_version(version_id)
        if version is None:
            raise ValueError(f"Unknown document version {version_id}")
        if not repository.claim_version(version_id):
            logger.info("Skipping stale or already processed version %s", version_id)
            return 0
        return _process_locked(version_id, version)


def _process_locked(version_id: str, version: dict) -> int:
    try:
        with _timed_stage(version_id, "parse"):
            blocks = parse_document(version["stored_path"])
        with _timed_stage(version_id, "chunk"):
            chunks = chunk_blocks(
                blocks,
                document_id=version["document_id"],
                version=version["version"],
                filename=version["filename"],
                content_hash=version["sha256"].strip(),
            )
        if not chunks:
            raise ValueError("No extractable content found in document.")
        with _timed_stage(version_id, "embed"):
            vectors = embed_texts([c["text"] for c in chunks])
        with get_pool().connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (version["filename"],))
            current = repository.get_version(version_id)
            if not current or current["version"] != current["current_version"] or current["status"] != "processing":
                logger.info("Skipping superseded version %s after embedding", version_id)
                return 0
            with _timed_stage(version_id, "vector_upsert"):
                milvus_store.upsert_chunks(chunks, vectors)
            with _timed_stage(version_id, "vector_delete_stale"):
                milvus_store.delete_stale(version["document_id"], [c["chunk_id"] for c in chunks])
            with _timed_stage(version_id, "persist_chunks"):
                repository.replace_chunks(version["document_id"], version_id, chunks)
    except Exception as e:
        logger.exception("Processing failed for %s", version["filename"])
        repository.mark_processing_failed(version_id, str(e)[:1000])
        raise
    logger.info("Processed %s v%s into %d chunks", version["filename"], version["version"], len(chunks))
    with _timed_stage(version_id, "invalidate_cache"):
        redis_cache.emit_documents_changed(version["document_id"], version["version"], "processed")
    return len(chunks)
