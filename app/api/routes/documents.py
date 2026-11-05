from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.db import repository
from app.processing.pipeline import process_version
from app.services.document_service import UploadError, describe, needs_processing, register_upload

router = APIRouter(prefix="/documents", tags=["documents"])


def _run_processing(version_id: str) -> None:
    try:
        process_version(version_id)
    except Exception:
        pass  # already logged and recorded as 'failed' on the version


@router.post("/upload")
async def upload_documents(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    acl_roles: str = Form("public"),
):
    results = []
    for f in files:
        name = f.filename or "file"
        try:
            reg = register_upload(name, await f.read(), acl_roles)
        except UploadError as e:
            raise HTTPException(status_code=400, detail=f"{name}: {e}")
        queued = needs_processing(reg)
        if queued:
            background.add_task(_run_processing, reg.version_id)
        results.append(
            {
                "filename": name,
                "document_id": reg.document_id,
                "version": reg.version,
                "action": reg.action,
                "detail": describe(reg),
                "processing_queued": queued,
            }
        )
    return {"message": "Documents uploaded. Processing runs in the background.", "documents": results}


@router.get("")
def list_documents():
    return repository.list_documents()


@router.get("/{document_id}/chunks")
def list_chunks(document_id: str, limit: int = 100):
    return repository.list_chunks(document_id, min(limit, 1000))
