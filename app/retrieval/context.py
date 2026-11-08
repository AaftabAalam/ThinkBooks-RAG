from typing import Callable


def retrieval_text(text: str, metadata: dict) -> str:
    headings = metadata.get("headings") or []
    return f"{' > '.join(headings)}\n\n{text}" if headings else text


def _label(hit: dict) -> str:
    meta = hit.get("metadata") or {}
    parts = [hit["filename"]]
    if meta.get("headings"):
        parts.append(" > ".join(meta["headings"]))
    if meta.get("pages"):
        parts.append("p. " + ", ".join(str(p) for p in meta["pages"]))
    if hit["chunk_type"] == "table":
        parts.append("table")
    return " | ".join(parts)


def assemble_context(hits: list[dict], max_tokens: int, count_tokens: Callable[[str], int]) -> tuple[str, list[dict]]:
    """Numbered context blocks in rerank order, stopping at the token budget. Returns (context, used hits)."""
    blocks: list[str] = []
    used: list[dict] = []
    total = 0
    for hit in hits:
        block = f"[{len(used) + 1}] ({_label(hit)})\n{hit['text']}"
        cost = count_tokens(block)
        if used and total + cost > max_tokens:
            break
        blocks.append(block)
        used.append(hit)
        total += cost
    return "\n\n".join(blocks), used
