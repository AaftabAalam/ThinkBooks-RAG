import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import (
    DocItemLabel,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

_SKIPPED_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
_convert_lock = threading.Lock()


@dataclass
class Block:
    """A contiguous piece of the document, in reading order."""

    kind: str  # "text" | "table"
    content: str  # plain text, or a markdown table
    headings: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    caption: str | None = None


@lru_cache(maxsize=1)
def _converter() -> DocumentConverter:
    return DocumentConverter()


def parse_document(path: str) -> list[Block]:
    """Parse any supported file into ordered text/table blocks."""
    if Path(path).suffix.lower() == ".txt":  # not a Docling input format
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return [Block("text", text)] if text.strip() else []

    with _convert_lock:  # the shared converter is not guaranteed thread-safe
        doc = _converter().convert(path).document

    blocks: list[Block] = []
    headings: list[str] = []
    parts: list[str] = []
    pages: set[int] = set()

    def flush() -> None:
        text = "\n\n".join(parts).strip()
        if text:
            blocks.append(Block("text", text, list(headings), sorted(pages)))
        parts.clear()
        pages.clear()

    for item, level in doc.iterate_items():
        item_pages = {p.page_no for p in getattr(item, "prov", [])}
        if isinstance(item, TableItem):
            flush()
            markdown = item.export_to_markdown(doc=doc).strip()
            if markdown:
                caption = item.caption_text(doc) or None
                blocks.append(Block("table", markdown, list(headings), sorted(item_pages), caption))
        elif isinstance(item, (TitleItem, SectionHeaderItem)):
            flush()
            depth = 0 if isinstance(item, TitleItem) else max(item.level, 1) - 1
            del headings[depth:]
            headings.append(item.text.strip())
            pages |= item_pages
        elif isinstance(item, TextItem):
            if item.label in _SKIPPED_LABELS or not item.text.strip():
                continue
            parts.append(item.text.strip())
            pages |= item_pages
    flush()
    return blocks
