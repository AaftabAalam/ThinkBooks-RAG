import os

import gradio as gr
from pydantic import ValidationError

from app.api.schemas import QueryRequest
from app.processing.pipeline import process_version
from app.services.document_service import UploadError, describe, needs_processing, register_upload
from app.services.query_service import answer_query


def _handle_upload(files: list[str] | None, roles: str) -> str:
    if not files:
        return "Please select at least one document."
    lines = []
    for path in files:
        name = os.path.basename(path)
        try:
            with open(path, "rb") as fh:
                reg = register_upload(name, fh.read(), roles)
            msg = describe(reg)
            if needs_processing(reg):
                count = process_version(reg.version_id)
                msg += (
                    f"; parsed, chunked ({count} chunks), embedded, indexed (dense + BM25) "
                    "in Milvus, and answer cache invalidated"
                )
            lines.append(f"✅ {name}: {msg}")
        except UploadError as e:
            lines.append(f"❌ {name}: {e}")
        except Exception as e:  # keep the UI alive on infra/processing errors
            lines.append(f"❌ {name}: processing failed ({e})")
    return "\n".join(lines)


def _handle_query(question: str, roles: str, use_cache: bool) -> tuple[str, str]:
    try:
        req = QueryRequest(query=question or "", roles=[r for r in roles.split(",")], use_cache=use_cache)
    except ValidationError as e:
        return "", "Invalid query: " + "; ".join(err["msg"] for err in e.errors())
    try:
        res = answer_query(req)
    except Exception as e:
        return "", f"Query failed: {e}"
    sim = f" (similarity {res.similarity:.3f})" if res.similarity is not None else ""
    sources = "\n".join(
        f"[{i}] {s.filename} v{s.version} [{s.chunk_type}] "
        f"(document {s.document_id}, chunk {s.chunk_id})"
        + (f" pages {', '.join(map(str, s.pages))}" if s.pages else "")
        for i, s in enumerate(res.sources, 1)
    ) or "none"
    cited = ", ".join(f"[{c.number}]" for c in res.citations) or "none"
    info = f"Cache: {res.cache}{sim}\nTimings (ms): {res.timings_ms}\nCited sources: {cited}\nContext sources:\n{sources}"
    return res.answer, info


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG") as demo:
        gr.Markdown("# RAG")
        with gr.Tab("Upload"):
            files = gr.File(label="Documents", file_count="multiple", type="filepath")
            up_roles = gr.Textbox(label="Allowed roles (comma-separated)", value="public")
            btn = gr.Button("Upload", variant="primary")
            out = gr.Textbox(label="Status", lines=6, interactive=False)
            btn.click(_handle_upload, inputs=[files, up_roles], outputs=out)
        with gr.Tab("Ask"):
            question = gr.Textbox(label="Question", lines=2)
            q_roles = gr.Textbox(label="Your roles (comma-separated)", value="public")
            use_cache = gr.Checkbox(label="Use cache", value=True)
            ask = gr.Button("Ask", variant="primary")
            answer = gr.Markdown()
            info = gr.Textbox(label="Details", lines=6, interactive=False)
            ask.click(_handle_query, inputs=[question, q_roles, use_cache], outputs=[answer, info])
    return demo
