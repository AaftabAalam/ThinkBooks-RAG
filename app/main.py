from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI

from app.api.routes import documents, query
from app.core.logging import setup_logging
from app.db.session import close_pool, init_db
from app.ui.gradio_app import build_ui


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    init_db()
    yield
    close_pool()


app = FastAPI(title="RAG Backend", lifespan=lifespan)
app.include_router(documents.router, prefix="/api")
app.include_router(query.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}


app = gr.mount_gradio_app(app, build_ui(), path="/ui")
