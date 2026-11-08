import base64
import csv
import io
import logging
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import pypdfium2
from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from openpyxl import load_workbook
from pdf_oxide import PdfDocument
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Block:
    """A contiguous piece of the document in reading order."""

    kind: str
    content: str
    headings: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    caption: str | None = None


def _table_markdown(rows: list[list[object]]) -> str:
    if not rows:
        return ""
    width = max(map(len, rows))
    if not width:
        return ""

    def row(values: list[object]) -> str:
        cells = [str(value if value is not None else "").replace("|", r"\|").replace("\r", " ").replace("\n", " ") for value in values]
        return "| " + " | ".join(cells + [""] * (width - len(cells))) + " |"

    return "\n".join([row(rows[0]), row(["---"] * width), *(row(values) for values in rows[1:])])


def _ocr(data: bytes, location: str) -> str:
    language = get_settings().ocr_language
    try:
        result = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", language],
            input=data, capture_output=True, check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Tesseract is required to OCR {location}; install it and language data for {language!r}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"OCR failed for {location}: {detail}")
    return result.stdout.decode("utf-8").strip()


def _pdf_tables(page, rendered: pypdfium2.PdfDocument, index: int, native_text: str) -> list:
    if page.rotation == 0:
        return page.extract_tables()

    copy = pypdfium2.PdfDocument.new()
    try:
        copy.import_pages(rendered, [index])
        rotated = copy[0]
        try:
            rotated.set_rotation(0)
        finally:
            rotated.close()
        data = io.BytesIO()
        copy.save(data)
        data.seek(0)
        with pdfplumber.open(data) as corrected:
            alternative = corrected.pages[0]

            def overlap(text: str) -> int:
                return sum((Counter(re.findall(r"\w+", native_text.casefold())) &
                            Counter(re.findall(r"\w+", text.casefold()))).values())

            if overlap(alternative.extract_text() or "") > overlap(page.extract_text() or ""):
                return alternative.extract_tables()
        return page.extract_tables()
    finally:
        copy.close()


def _pdf(path: Path) -> list[Block]:
    blocks: list[Block] = []
    headings: list[str] = []
    with PdfDocument(str(path)) as oxide, pdfplumber.open(str(path)) as table_pdf, pypdfium2.PdfDocument(str(path)) as rendered:
        if len(oxide) != len(table_pdf.pages) or len(oxide) != len(rendered):
            raise ValueError(f"PDF readers disagree on page count for {path}")
        for index, page in enumerate(oxide):
            page_number = index + 1
            text = page.text.strip()
            # Markdown is used only for heading hints; its table cells can be inaccurate.
            position = 0
            for match in re.finditer(r"^(#{1,6})[ \t]+(.+)$", page.markdown(detect_headings=True), re.MULTILINE):
                level, title = len(match.group(1)), match.group(2).strip()
                pattern = r"(?m)^[ \t]*" + r"\s+".join(re.escape(word) for word in title.split()) + r"[ \t]*(?=\n|$)"
                found = re.search(pattern, text[position:])
                if not found:
                    continue
                start = position + found.start()
                if text[position:start].strip():
                    blocks.append(Block("text", text[position:start].strip(), list(headings), [page_number]))
                headings = headings[:level - 1] + [title]
                position = start
            if text[position:].strip():
                blocks.append(Block("text", text[position:].strip(), list(headings), [page_number]))
            if get_settings().pdf_do_ocr and page.images:
                render_page = rendered[index]
                try:
                    image = render_page.render(scale=150 / 72).to_pil()
                    data = io.BytesIO()
                    image.save(data, format="PNG")
                    ocr_text = _ocr(data.getvalue(), f"PDF page {page_number}")
                finally:
                    render_page.close()
                if ocr_text:
                    blocks.append(Block("text", ocr_text, list(headings), [page_number]))
                else:
                    logger.warning("OCR returned no text for image-bearing PDF page %d of %s", page_number, path.name)
                logger.info("OCR processed image-bearing PDF page %d of %s", page_number, path.name)
            for table in _pdf_tables(table_pdf.pages[index], rendered, index, text):
                markdown = _table_markdown(table)
                if markdown:
                    blocks.append(Block("table", markdown, list(headings), [page_number]))
    return blocks


def _docx(path: Path) -> list[Block]:
    document = Document(str(path))
    blocks: list[Block] = []
    headings: list[str] = []

    def paragraph_text(paragraph: Paragraph) -> str:
        pieces: list[str] = []
        for node in paragraph._p.iter():
            if node.tag in (
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t",
                "{http://schemas.openxmlformats.org/officeDocument/2006/math}t",
            ):
                pieces.append(node.text or "")
            elif node.tag == "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tab":
                pieces.append("\t")
            elif node.tag in ("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br",
                              "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}cr"):
                pieces.append("\n")
        return "".join(pieces).strip()

    def append_images(element, container) -> None:
        if not get_settings().pdf_do_ocr:
            return
        for blip in element.xpath(".//a:blip"):
            image_id = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
            if image_id:
                text = _ocr(container.part.related_parts[image_id].blob, f"image in {path.name}")
                if text:
                    blocks.append(Block("text", text, list(headings)))

    def cell_text(cell) -> str:
        contents: list[str] = []
        for item in cell.iter_inner_content():
            if isinstance(item, Paragraph):
                contents.append(paragraph_text(item))
            elif isinstance(item, Table):
                contents.extend(" | ".join(cell_text(child) for child in row.cells) for row in item.rows)
        return "\n".join(contents)

    def append_items(container) -> None:
        for item in container.iter_inner_content():
            if isinstance(item, Table):
                markdown = _table_markdown([
                    [cell_text(cell) for cell in row.cells]
                    for row in item.rows
                ])
                if markdown:
                    blocks.append(Block("table", markdown, list(headings)))
                append_images(item._tbl, container)
                continue
            if not isinstance(item, Paragraph):
                continue
            text = paragraph_text(item)
            style = item.style.name if item.style else ""
            match = re.fullmatch(r"Heading ([1-9])", style)
            if text and (match or style == "Title"):
                depth = int(match.group(1)) - 1 if match else 0
                headings[:] = headings[:depth] + [text]
            if text:
                blocks.append(Block("text", text, list(headings)))
            append_images(item._p, container)

    append_items(document)
    seen: set[str] = set()
    for section in document.sections:
        for part in (section.header, section.first_page_header, section.even_page_header,
                     section.footer, section.first_page_footer, section.even_page_footer):
            if part.is_linked_to_previous or str(part.part.partname) in seen:
                continue
            seen.add(str(part.part.partname))
            append_items(part)
    return blocks


def _pptx(path: Path) -> list[Block]:
    presentation = Presentation(str(path))
    blocks: list[Block] = []

    def add_shapes(shapes, page: int, headings: list[str]) -> None:
        for shape in shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                add_shapes(shape.shapes, page, headings)
            elif shape.has_table:
                content = _table_markdown([[cell.text for cell in row.cells] for row in shape.table.rows])
                if content:
                    blocks.append(Block("table", content, list(headings), [page]))
            elif shape.has_chart:
                chart = shape.chart
                title = chart.chart_title.text_frame.text.strip() if chart.has_title else None
                for plot in chart.plots:
                    series = list(plot.series)
                    categories = [str(category.label) for category in plot.categories]
                    if categories:
                        rows = [["Category", *(item.name for item in series)]]
                        values = [list(item.values) for item in series]
                        for index in range(max([len(categories), *(len(v) for v in values)])):
                            rows.append([categories[index] if index < len(categories) else "",
                                         *(v[index] if index < len(v) else None for v in values)])
                        blocks.append(Block("table", _table_markdown(rows), list(headings), [page], title or None))
                    else:
                        for item in series:
                            x_points = {
                                int(point.get("idx")): point.xpath("./c:v")[0].text
                                for point in item._element.xpath("./c:xVal//c:pt")
                                if point.get("idx") is not None and point.xpath("./c:v")
                            }
                            values = list(item.values)
                            rows = [["X", item.name or "Value"]]
                            rows.extend([x_points.get(index, str(index + 1)), value] for index, value in enumerate(values))
                            if values:
                                blocks.append(Block("table", _table_markdown(rows), list(headings), [page], title or None))
            elif shape.has_text_frame and shape.text.strip():
                blocks.append(Block("text", shape.text.strip(), list(headings), [page]))
            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE and get_settings().pdf_do_ocr:
                content = _ocr(shape.image.blob, f"image on slide {page} of {path.name}")
                if content:
                    blocks.append(Block("text", content, list(headings), [page]))

    for page, slide in enumerate(presentation.slides, 1):
        title = slide.shapes.title
        headings = [title.text.strip()] if title and title.has_text_frame and title.text.strip() else []
        add_shapes(slide.shapes, page, headings)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame
            if notes and notes.text.strip():
                blocks.append(Block("text", notes.text.strip(), headings, [page]))
    return blocks


def _xlsx(path: Path) -> list[Block]:
    workbook = load_workbook(str(path), data_only=False)
    cached = load_workbook(str(path), data_only=True, read_only=True)
    blocks: list[Block] = []
    try:
        for sheet in workbook:
            if sheet._charts:
                logger.warning("Chart layout and labels on sheet %s of %s are not extracted; underlying cells are included", sheet.title, path.name)
            cached_sheet = cached[sheet.title]
            rows: list[list[object]] = []
            for row, cached_row in zip(sheet.iter_rows(), cached_sheet.iter_rows(), strict=True):
                values: list[object] = []
                for cell, cached_cell in zip(row, cached_row, strict=True):
                    value = cell.value
                    if cell.data_type == "f":
                        cached_value = cached_cell.value
                        if cached_value is not None:
                            value = f"{value} (cached value: {cached_value})"
                    values.append(value)
                if any(value is not None for value in values):
                    rows.append(values)
            content = _table_markdown(rows)
            if content:
                blocks.append(Block("table", content, [sheet.title]))
            if get_settings().pdf_do_ocr:
                for image in sheet._images:
                    text = _ocr(image._data(), f"image on sheet {sheet.title} of {path.name}")
                    if text:
                        blocks.append(Block("text", text, [sheet.title]))
    finally:
        workbook.close()
        cached.close()
    return blocks


def _html(path: Path) -> list[Block]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    blocks: list[Block] = []
    headings: list[str] = []
    parts: list[str] = []

    def flush() -> None:
        text = " ".join(parts).strip()
        if text:
            blocks.append(Block("text", text, list(headings)))
        parts.clear()

    def visit(node) -> None:
        if isinstance(node, NavigableString):
            text = str(node).strip()
            if text:
                parts.append(text)
            return
        if not isinstance(node, Tag):
            return
        if re.fullmatch(r"h[1-6]", node.name or ""):
            flush()
            text = node.get_text(" ", strip=True)
            depth = int(node.name[1]) - 1
            headings[:] = headings[:depth] + [text]
            if text:
                blocks.append(Block("text", text, list(headings)))
        elif node.name == "table":
            flush()
            rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"], recursive=False)] for row in node.find_all("tr")]
            content = _table_markdown(rows)
            if content:
                caption = node.find("caption")
                blocks.append(Block("table", content, list(headings), caption=caption.get_text(" ", strip=True) if caption else None))
        elif node.name == "img":
            flush()
            alt = node.get("alt", "").strip()
            if alt:
                blocks.append(Block("text", alt, list(headings)))
            src = node.get("src", "")
            if get_settings().pdf_do_ocr and src.startswith("data:image/"):
                header, separator, data = src.partition(",")
                if not separator or ";base64" not in header:
                    raise ValueError(f"Unsupported embedded image encoding in {path.name}")
                text = _ocr(base64.b64decode(data, validate=True), f"embedded HTML image in {path.name}")
                if text:
                    blocks.append(Block("text", text, list(headings)))
            elif src:
                logger.warning("HTML image is not embedded and was not fetched: %s", src)
        else:
            if node.name in ("p", "div", "li", "br", "section", "article"):
                flush()
            for child in node.children:
                visit(child)
            if node.name in ("p", "div", "li", "br", "section", "article"):
                flush()

    visit(soup.body or soup)
    flush()
    return blocks


def parse_document(path: str) -> list[Block]:
    """Extract supported documents into text and table blocks."""
    file = Path(path)
    suffix = file.suffix.lower()
    if suffix == ".pdf":
        return _pdf(file)
    if suffix == ".docx":
        return _docx(file)
    if suffix == ".pptx":
        return _pptx(file)
    if suffix == ".xlsx":
        return _xlsx(file)
    if suffix == ".html":
        return _html(file)
    if suffix == ".csv":
        with file.open(encoding="utf-8-sig", newline="") as stream:
            content = _table_markdown(list(csv.reader(stream)))
        return [Block("table", content)] if content else []
    if suffix in (".txt", ".md"):
        text = file.read_text(encoding="utf-8", errors="replace")
        return [Block("text", text)] if text.strip() else []
    raise ValueError(f"Unsupported document format: {suffix or '<none>'}")
