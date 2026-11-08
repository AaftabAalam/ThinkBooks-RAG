import re
import unicodedata

from pydantic import BaseModel, Field, field_validator


def normalize_query(query: str) -> str:
    """Canonical form used for cache keys and retrieval: NFKC, collapsed whitespace, casefolded, no trailing punctuation."""
    q = unicodedata.normalize("NFKC", query)
    q = re.sub(r"\s+", " ", q).strip().casefold()
    return q.rstrip("?!. ")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    roles: list[str] = Field(default_factory=lambda: ["public"], max_length=20)
    use_cache: bool = True

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not normalize_query(v):
            raise ValueError("query must contain text")
        return v

    @field_validator("roles")
    @classmethod
    def _clean_roles(cls, v: list[str]) -> list[str]:
        cleaned = sorted({r.strip().lower() for r in v if r.strip()})
        return cleaned or ["public"]


class Source(BaseModel):
    document_id: str
    filename: str
    version: int
    chunk_id: str
    chunk_type: str
    rerank_score: float | None = None
    pages: list[int] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    number: int
    source: Source


class QueryResponse(BaseModel):
    answer: str
    cache: str  # "exact" | "semantic" | "miss"
    similarity: float | None = None
    normalized_query: str
    sources: list[Source]
    timings_ms: dict[str, int]
    citations: list[Citation] = Field(default_factory=list)
