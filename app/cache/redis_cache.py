import hashlib
import json
import logging
from functools import lru_cache

import redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_PREFIX = "rag:cache:"
EVENTS_CHANNEL = "rag:events"


@lru_cache(maxsize=1)
def _redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, socket_timeout=2, socket_connect_timeout=2)


def _namespace() -> str:
    s = get_settings()
    model = hashlib.sha256(
        f"{s.embedding_model}|{s.embedding_dim}|{s.llm_model}|{s.llm_max_tokens}|"
        f"{s.context_max_tokens}|{s.reranker_model}|{s.retrieval_top_k}|{s.rerank_top_n}|"
        f"{s.rerank_min_score}".encode()
    ).hexdigest()[:16]
    return f"{_PREFIX}v6:{s.milvus_collection}:{model}:"


def _exact_key(normalized_query: str) -> str:
    return _namespace() + "exact:" + hashlib.sha256(normalized_query.encode()).hexdigest()


def get_exact(normalized_query: str) -> dict | None:
    try:
        raw = _redis().get(_exact_key(normalized_query))
        return json.loads(raw) if raw else None
    except redis.RedisError:
        logger.warning("Redis unavailable for exact cache lookup", exc_info=True)
        return None


def store(query: str, entry: dict) -> None:
    s = get_settings()
    payload = json.dumps(entry)
    try:
        r = _redis()
        r.set(_exact_key(query), payload, ex=s.cache_ttl_seconds)
    except redis.RedisError:
        logger.warning("Redis unavailable; answer not cached", exc_info=True)


def invalidate_all() -> int:
    """Drop every cached answer (exact + semantic). Returns the number of keys deleted."""
    r = _redis()
    deleted = 0
    for key in r.scan_iter(match=_PREFIX + "*", count=500):
        deleted += r.delete(key)
    return deleted


def emit_documents_changed(document_id: str, version: int, action: str) -> None:
    """Cache-invalidation event: a document was added or changed, so cached answers may be stale.

    New or changed content can alter the answer to any query, so the whole answer cache is cleared.
    Failures never break ingestion; cached entries are also re-validated against Postgres on every hit.
    """
    try:
        deleted = invalidate_all()
        _redis().publish(
            EVENTS_CHANNEL,
            json.dumps({"event": "documents_changed", "document_id": document_id, "version": version, "action": action}),
        )
        logger.info("Cache invalidated (%d keys) after %s of %s v%s", deleted, action, document_id, version)
    except redis.RedisError:
        logger.warning("Cache invalidation failed (Redis unavailable)", exc_info=True)
