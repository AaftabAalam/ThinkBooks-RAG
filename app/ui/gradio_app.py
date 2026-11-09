import os
import logging
from contextlib import closing

import gradio as gr
from pydantic import ValidationError

from app.api.schemas import QueryRequest
from app.services.document_service import UploadError, describe, enqueue_processing, register_upload
from app.services.query_service import stream_query

logger = logging.getLogger(__name__)


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
            if enqueue_processing(reg):
                msg += "; queued for background processing"
            lines.append(f"✅ {name}: {msg}")
        except UploadError as e:
            lines.append(f"❌ {name}: {e}")
        except Exception as e:  # keep the UI alive on infra/processing errors
            logger.exception("Upload failed for %s", name)
            lines.append(f"❌ {name}: upload or queue failed ({e})")
    return "\n".join(lines)


def _handle_query(question: str, roles: str, use_cache: bool):
    try:
        req = QueryRequest(query=question or "", roles=[r for r in roles.split(",")], use_cache=use_cache)
    except ValidationError as e:
        yield "", "Invalid query: " + "; ".join(err["msg"] for err in e.errors())
        return
    answer = ""
    try:
        with closing(stream_query(req)) as events:
            for fragment, response in events:
                answer += fragment
                if response is None:
                    yield answer, "Generating..."
                else:
                    yield answer, f"Total time: {response.timings_ms['total'] / 1000:.2f} s"
    except Exception as e:
        logger.exception("Query failed")
        yield answer, f"Query failed: {e}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG") as demo:
        gr.Markdown("# RAG")
        with gr.Tab("Upload"):
            files = gr.File(label="Documents", file_count="multiple", type="filepath")
            up_roles = gr.Textbox(label="Allowed roles (comma-separated)", value="public")
            btn = gr.Button("Upload", variant="primary")
            out = gr.Textbox(label="Status", lines=6, interactive=False)
            btn.click(_handle_upload, inputs=[files, up_roles], outputs=out, concurrency_limit=None)
        with gr.Tab("Ask"):
            question = gr.Textbox(label="Question", lines=2)
            q_roles = gr.Textbox(label="Your roles (comma-separated)", value="public")
            use_cache = gr.Checkbox(label="Use cache", value=True)
            ask = gr.Button("Ask", variant="primary")
            answer = gr.Markdown()
            info = gr.Textbox(label="Status and total time", interactive=False)
            ask.click(_handle_query, inputs=[question, q_roles, use_cache], outputs=[answer, info])
    return demo
