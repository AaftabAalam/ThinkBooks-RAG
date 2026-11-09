import logging
import math
import re
import time
from collections.abc import Iterator
from contextlib import closing

from app.api.schemas import Citation, QueryRequest, QueryResponse, Source, normalize_query
from app.cache import redis_cache
from app.core.config import get_settings
from app.db import repository
from app.embedding.embedder import embed_texts
from app.llm.generator import ANSWER_MARKER, NO_ANSWER_MARKER, NO_CONTENT_ANSWER, count_tokens, stream_answer
from app.retrieval.context import assemble_context
from app.retrieval.reranker import rerank
from app.vectorstore import milvus_store

logger = logging.getLogger(__name__)

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
    with closing(stream_query(req)) as events:
        for _, response in events:
            if response is not None:
                return response
    raise RuntimeError("Query stream ended without a final response.")


def stream_query(req: QueryRequest) -> Iterator[tuple[str, QueryResponse | None]]:
    start = time.perf_counter()
    s = get_settings()
    timings: dict[str, int] = {}

    normalized = normalize_query(req.query)
    cache_query = req.query.strip()

    if req.use_cache:
        t = time.perf_counter()
        entry = redis_cache.get_exact(cache_query)
        if entry and _from_cache(entry, req):
            timings["exact_cache"] = _ms(t)
            response = _cached_response(entry, "exact", normalized, timings)
            response.timings_ms["total"] = _ms(start)
            yield response.answer, response
            return
        timings["exact_cache"] = _ms(t)

    t = time.perf_counter()
    query_vector = embed_texts([normalized], task="search_query")[0]
    timings["embed"] = _ms(t)

    t = time.perf_counter()
    allowed = repository.accessible_document_ids(req.roles)
    hits = milvus_store.hybrid_search(normalized, query_vector, document_ids=allowed, limit=s.retrieval_top_k)
    timings["hybrid_search"] = _ms(t)
    if not hits:
        timings["total"] = _ms(start)
        response = QueryResponse(answer=NO_CONTENT_ANSWER, cache="miss", normalized_query=normalized, sources=[], timings_ms=timings)
        yield response.answer, response
        return

    t = time.perf_counter()
    top = rerank(normalized, hits, s.rerank_top_n)
    timings["rerank"] = _ms(t)
    for hit in top:
        score = hit.get("rerank_score")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Reranker returned an invalid relevance score.")
    top = [hit for hit in top if hit["rerank_score"] >= s.rerank_min_score]
    if not top:
        timings["total"] = _ms(start)
        response = QueryResponse(answer=NO_CONTENT_ANSWER, cache="miss", normalized_query=normalized, sources=[], timings_ms=timings)
        yield response.answer, response
        return

    t = time.perf_counter()
    context, used = assemble_context(top, s.context_max_tokens, count_tokens)
    timings["context"] = _ms(t)

    t = time.perf_counter()
    parts = []
    prefix = ""
    marker = None
    with closing(stream_answer(req.query.strip(), context)) as fragments:
        for fragment in fragments:
            if marker is None:
                prefix += fragment
                prefix = prefix.lstrip()
                if prefix.startswith("ANSWER:"):
                    prefix = prefix[len("ANSWER:"):].lstrip()
                elif "ANSWER:".startswith(prefix):
                    continue
                if prefix.startswith(NO_ANSWER_MARKER):
                    marker = NO_ANSWER_MARKER
                    break
                if prefix.startswith(ANSWER_MARKER):
                    marker = ANSWER_MARKER
                    fragment = prefix[len(ANSWER_MARKER):].lstrip()
                elif any(value.startswith(prefix) for value in (ANSWER_MARKER, NO_ANSWER_MARKER)):
                    continue
                else:
                    logger.warning("Model did not return a valid answerability marker; refusing the response.")
                    marker = NO_ANSWER_MARKER
                    break
            if not fragment:
                continue
            parts.append(fragment)
            yield fragment, None
    if marker is None:
        logger.warning("Model did not return a complete answerability marker; refusing the response.")
        marker = NO_ANSWER_MARKER
    if marker == NO_ANSWER_MARKER:
        timings["generate"] = _ms(t)
        timings["total"] = _ms(start)
        response = QueryResponse(answer=NO_CONTENT_ANSWER, cache="miss", normalized_query=normalized, sources=[], timings_ms=timings)
        yield response.answer, response
        return
    answer = "".join(parts)
    if not answer.strip():
        raise ValueError("Model returned an empty answer.")
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
        redis_cache.store(cache_query, {
            "answer": answer,
            "sources": [x.model_dump() for x in sources],
            "citations": [x.model_dump() for x in citations],
        })
    timings["total"] = _ms(start)
    response = QueryResponse(
        answer=answer, cache="miss", normalized_query=normalized, sources=sources,
        citations=citations, timings_ms=timings,
    )
    yield "", response
