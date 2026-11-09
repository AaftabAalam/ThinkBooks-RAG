from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.db import repository
from app.services.document_service import UploadError, describe, enqueue_processing, register_upload

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    acl_roles: str = Form("public"),
):
    results = []
    for f in files:
        name = f.filename or "file"
        try:
            reg = await run_in_threadpool(register_upload, name, await f.read(), acl_roles)
        except UploadError as e:
            raise HTTPException(status_code=400, detail=f"{name}: {e}")
        queued = await run_in_threadpool(enqueue_processing, reg)
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
