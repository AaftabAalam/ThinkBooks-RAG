import logging
import time
from contextlib import contextmanager
from typing import Iterator

from app.cache import redis_cache
from app.db import repository
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

    repository.set_version_status(version_id, "processing")
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
        with _timed_stage(version_id, "vector_upsert"):
            milvus_store.upsert_chunks(chunks, vectors)
        with _timed_stage(version_id, "vector_delete_stale"):
            milvus_store.delete_stale(version["document_id"], [c["chunk_id"] for c in chunks])
        with _timed_stage(version_id, "persist_chunks"):
            repository.replace_chunks(version["document_id"], version_id, chunks)
    except Exception as e:
        logger.exception("Processing failed for %s", version["filename"])
        repository.set_version_status(version_id, "failed", str(e)[:1000])
        raise
    logger.info("Processed %s v%s into %d chunks", version["filename"], version["version"], len(chunks))
    with _timed_stage(version_id, "invalidate_cache"):
        redis_cache.emit_documents_changed(version["document_id"], version["version"], "processed")
    return len(chunks)
