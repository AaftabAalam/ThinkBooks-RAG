import hashlib
import json
import logging
import time
import uuid
from functools import lru_cache

import numpy as np
import redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_PREFIX = "rag:cache:"
_EXACT = _PREFIX + "exact:"
_SEM = _PREFIX + "sem:"
_SEM_INDEX = _PREFIX + "semindex"  # sorted set of semantic entry ids by creation time
EVENTS_CHANNEL = "rag:events"


@lru_cache(maxsize=1)
def _redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, socket_timeout=2, socket_connect_timeout=2)


def _exact_key(normalized_query: str) -> str:
    return _EXACT + hashlib.sha256(normalized_query.encode()).hexdigest()


def get_exact(normalized_query: str) -> dict | None:
    try:
        raw = _redis().get(_exact_key(normalized_query))
        return json.loads(raw) if raw else None
    except redis.RedisError:
        logger.warning("Redis unavailable for exact cache lookup", exc_info=True)
        return None


def get_semantic(query_vector: list[float]) -> tuple[dict, float] | None:
    """Best cached entry whose cosine similarity to the query is >= the configured threshold."""
    s = get_settings()
    try:
        r = _redis()
        ids = [i.decode() for i in r.zrevrange(_SEM_INDEX, 0, s.semantic_cache_max_entries - 1)]
        if not ids:
            return None
        pipe = r.pipeline(transaction=False)
        for i in ids:
            pipe.hmget(_SEM + i, "emb", "entry")
        rows = pipe.execute()
        live = [(i, row) for i, row in zip(ids, rows) if row[0] is not None]
        for i in set(ids) - {i for i, _ in live}:  # expired entries
            r.zrem(_SEM_INDEX, i)
        if not live:
            return None
        matrix = np.vstack([np.frombuffer(row[0], dtype=np.float32) for _, row in live])
        query = np.asarray(query_vector, dtype=np.float32)
        # Vectors are L2-normalised, so the dot product equals cosine similarity.
        scores = matrix @ query
        best = int(np.argmax(scores))
        if float(scores[best]) < s.semantic_cache_threshold:
            return None
        return json.loads(live[best][1][1]), float(scores[best])
    except redis.RedisError:
        logger.warning("Redis unavailable for semantic cache lookup", exc_info=True)
        return None


def store(normalized_query: str, query_vector: list[float], entry: dict) -> None:
    s = get_settings()
    payload = json.dumps(entry)
    try:
        r = _redis()
        r.set(_exact_key(normalized_query), payload, ex=s.cache_ttl_seconds)
        sem_id = uuid.uuid4().hex
        key = _SEM + sem_id
        pipe = r.pipeline()
        pipe.hset(key, mapping={"emb": np.asarray(query_vector, dtype=np.float32).tobytes(), "entry": payload})
        pipe.expire(key, s.cache_ttl_seconds)
        pipe.zadd(_SEM_INDEX, {sem_id: time.time()})
        pipe.execute()
        overflow = r.zcard(_SEM_INDEX) - s.semantic_cache_max_entries
        if overflow > 0:
            for old in r.zrange(_SEM_INDEX, 0, overflow - 1):
                r.delete(_SEM + old.decode())
                r.zrem(_SEM_INDEX, old)
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
