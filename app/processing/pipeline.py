import logging

from app.cache import redis_cache
from app.db import repository
from app.embedding.embedder import embed_texts
from app.processing.chunker import chunk_blocks
from app.processing.parser import parse_document
from app.vectorstore import milvus_store

logger = logging.getLogger(__name__)


def process_version(version_id: str) -> int:
    """Parse with Docling, chunk with Chonkie, and persist chunks. Returns chunk count."""
    version = repository.get_version(version_id)
    if version is None:
        raise ValueError(f"Unknown document version {version_id}")

    repository.set_version_status(version_id, "processing")
    try:
        blocks = parse_document(version["stored_path"])
        chunks = chunk_blocks(
            blocks,
            document_id=version["document_id"],
            version=version["version"],
            filename=version["filename"],
            content_hash=version["sha256"].strip(),
        )
        if not chunks:
            raise ValueError("No extractable content found in document.")
        vectors = embed_texts([c["text"] for c in chunks])
        milvus_store.upsert_chunks(chunks, vectors)
        milvus_store.delete_stale(version["document_id"], [c["chunk_id"] for c in chunks])
        repository.replace_chunks(version["document_id"], version_id, chunks)
    except Exception as e:
        logger.exception("Processing failed for %s", version["filename"])
        repository.set_version_status(version_id, "failed", str(e)[:1000])
        raise
    logger.info("Processed %s v%s into %d chunks", version["filename"], version["version"], len(chunks))
    redis_cache.emit_documents_changed(version["document_id"], version["version"], "processed")
    return len(chunks)
