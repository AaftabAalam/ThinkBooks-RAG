import logging
import re
import time

from app.api.schemas import Citation, QueryRequest, QueryResponse, Source, normalize_query
from app.cache import redis_cache
from app.core.config import get_settings
from app.db import repository
from app.embedding.embedder import embed_texts
from app.llm.generator import count_tokens, generate_answer
from app.retrieval.context import assemble_context
from app.retrieval.reranker import rerank
from app.vectorstore import milvus_store

logger = logging.getLogger(__name__)

NO_CONTENT_ANSWER = "I couldn't find any relevant information in the documents you have access to."
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _from_cache(entry: dict, req: QueryRequest) -> bool:
    """Cache freshness + ACL check: cited documents must be at the cited version and readable by the caller."""
    return repository.validate_sources(entry["sources"], req.roles)


def _link_citations(answer: str, sources: list[Source]) -> list[Citation]:
    citations = []
    seen = set()
    for match in _CITATION.finditer(answer):
        for value in match.group(1).split(","):
            number = int(value.strip())
            if number < 1 or number > len(sources):
                raise ValueError(f"Generated answer cites source [{number}], but only {len(sources)} sources were provided")
            if number not in seen:
                citations.append(Citation(number=number, source=sources[number - 1]))
                seen.add(number)
    return citations


def _cached_response(entry: dict, kind: str, normalized: str, timings: dict, similarity: float | None = None):
    sources = [Source(**s) for s in entry["sources"]]
    return QueryResponse(
        answer=entry["answer"],
        cache=kind,
        similarity=similarity,
        normalized_query=normalized,
        sources=sources,
        citations=_link_citations(entry["answer"], sources),
        timings_ms=timings,
    )


def answer_query(req: QueryRequest) -> QueryResponse:
    """Validated request -> normalise -> exact cache -> embed -> semantic cache -> hybrid search
    (dense + BM25, RRF) -> rerank -> context assembly -> LLM -> cache."""
    s = get_settings()
    timings: dict[str, int] = {}

    normalized = normalize_query(req.query)

    if req.use_cache:
        t = time.perf_counter()
        entry = redis_cache.get_exact(normalized)
        if entry and _from_cache(entry, req):
            timings["exact_cache"] = _ms(t)
            return _cached_response(entry, "exact", normalized, timings)
        timings["exact_cache"] = _ms(t)

    t = time.perf_counter()
    query_vector = embed_texts([normalized], task="search_query")[0]
    timings["embed"] = _ms(t)

    if req.use_cache and s.semantic_cache_enabled:
        t = time.perf_counter()
        found = redis_cache.get_semantic(query_vector)
        timings["semantic_cache"] = _ms(t)
        if found and _from_cache(found[0], req):
            return _cached_response(found[0], "semantic", normalized, timings, similarity=found[1])

    t = time.perf_counter()
    allowed = repository.accessible_document_ids(req.roles)
    hits = milvus_store.hybrid_search(normalized, query_vector, document_ids=allowed, limit=s.retrieval_top_k)
    timings["hybrid_search"] = _ms(t)
    if not hits:
        return QueryResponse(answer=NO_CONTENT_ANSWER, cache="miss", normalized_query=normalized, sources=[], timings_ms=timings)

    t = time.perf_counter()
    top = rerank(normalized, hits, s.rerank_top_n)
    timings["rerank"] = _ms(t)

    t = time.perf_counter()
    context, used = assemble_context(top, s.context_max_tokens, count_tokens)
    timings["context"] = _ms(t)

    t = time.perf_counter()
    answer = generate_answer(req.query.strip(), context)
    timings["generate"] = _ms(t)

    sources = [
        Source(
            document_id=h["document_id"],
            filename=h["filename"],
            version=h["version"],
            chunk_id=h["chunk_id"],
            chunk_type=h["chunk_type"],
            rerank_score=round(h["rerank_score"], 4),
            pages=(h.get("metadata") or {}).get("pages") or [],
            headings=(h.get("metadata") or {}).get("headings") or [],
        )
        for h in used
    ]
    citations = _link_citations(answer, sources)
    if req.use_cache:
        redis_cache.store(normalized, query_vector, {
            "answer": answer,
            "sources": [x.model_dump() for x in sources],
            "citations": [x.model_dump() for x in citations],
        })
    return QueryResponse(
        answer=answer, cache="miss", normalized_query=normalized, sources=sources,
        citations=citations, timings_ms=timings,
    )
