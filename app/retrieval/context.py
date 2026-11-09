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
    for hit in hits:
        prefix = f"[{len(used) + 1}] ({_label(hit)})\n"
        block = prefix + hit["text"]
        candidate = "\n\n".join([*blocks, block])
        if count_tokens(candidate) > max_tokens:
            if used:
                break
            text = hit["text"]
            low, high = 0, len(text)
            while low < high:
                mid = (low + high + 1) // 2
                if count_tokens(prefix + text[:mid]) <= max_tokens:
                    low = mid
                else:
                    high = mid - 1
            if low == 0:
                raise ValueError("Context token budget is too small for the first source label.")
            block = prefix + text[:low]
            if count_tokens(block) > max_tokens:
                raise ValueError("Unable to fit the first source within the context token budget.")
            hit = {**hit, "text": text[:low]}
        blocks.append(block)
        used.append(hit)
    return "\n\n".join(blocks), used
