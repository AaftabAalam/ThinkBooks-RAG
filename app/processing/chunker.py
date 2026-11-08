import hashlib
import uuid
from functools import lru_cache

from chonkie import RecursiveChunker, TableChunker

from app.core.config import get_settings
from app.processing.parser import Block

_ID_NAMESPACE = uuid.UUID("6f1c2b7e-0d3a-4c55-9a77-3b1d5e0c9a10")


@lru_cache(maxsize=1)
def _text_chunker() -> RecursiveChunker:
    s = get_settings()
    return RecursiveChunker(tokenizer=s.chunk_tokenizer, chunk_size=s.chunk_text_tokens)


@lru_cache(maxsize=1)
def _table_chunker() -> TableChunker:
    return TableChunker(tokenizer="row", chunk_size=get_settings().chunk_table_rows)


def _stable_id(document_id: str, chunk_type: str, text: str, seen: dict[str, int]) -> str:
    """Deterministic ID: identical text in the same document keeps the same ID across versions."""
    key = f"{document_id}|{chunk_type}|{hashlib.sha256(text.encode()).hexdigest()}"
    occurrence = seen[key] = seen.get(key, -1) + 1  # disambiguates repeated identical chunks
    return str(uuid.uuid5(_ID_NAMESPACE, f"{key}|{occurrence}"))


def chunk_blocks(
    blocks: list[Block], *, document_id: str, version: int, filename: str, content_hash: str
) -> list[dict]:
    """Recursive chunking for text, row-based chunking for tables, merged in reading order."""
    chunks: list[dict] = []
    seen: dict[str, int] = {}

    for block_no, block in enumerate(blocks):
        if block.kind == "table":
            lines = [line for line in block.content.splitlines() if line.strip()]
            if len(lines) == 2:
                pieces = [(block.content, 0)]
            else:
                pieces = [(c.text, _row_count(c.text)) for c in _table_chunker().chunk(block.content)]
                if not pieces and block.content.strip():
                    raise ValueError(f"Unable to chunk table in block {block_no}")
        else:
            pieces = [(c.text, c.token_count) for c in _text_chunker().chunk(block.content)]

        total = len(pieces)
        for part, (text, size) in enumerate(pieces):
            if not text.strip():
                continue
            chunks.append(
                {
                    "chunk_id": _stable_id(document_id, block.kind, text, seen),
                    "chunk_index": len(chunks),
                    "chunk_type": block.kind,
                    "text": text,
                    "token_count": size,
                    "metadata": {
                        "document_id": document_id,
                        "filename": filename,
                        "version": version,
                        "content_hash": content_hash,
                        "chunk_type": block.kind,
                        "block_index": block_no,
                        "part": part,
                        "parts_in_block": total,
                        "headings": block.headings,
                        "pages": block.pages,
                        "table_caption": block.caption,
                        "unit": "rows" if block.kind == "table" else "tokens",
                    },
                }
            )
    return chunks


def _row_count(markdown_table: str) -> int:
    lines = [ln for ln in markdown_table.splitlines() if ln.strip()]
    return max(len(lines) - 2, 0)  # minus header and separator rows
