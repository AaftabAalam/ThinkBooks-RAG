import hashlib
import logging
import mimetypes
import re
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.db import repository
from app.db.repository import NEW, RETRY, UNCHANGED, UPDATED, Registration

logger = logging.getLogger(__name__)


class UploadError(Exception):
    pass


def parse_roles(roles: list[str] | str | None) -> list[str]:
    """Accept a list or a comma-separated string; default to the public role."""
    if isinstance(roles, str):
        roles = roles.split(",")
    cleaned = sorted({r.strip().lower() for r in (roles or []) if r.strip()})
    return cleaned or ["public"]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name) or "file"


def register_upload(filename: str, data: bytes, acl_roles: list[str] | None = None) -> Registration:
    """Validate, hash (SHA-256), run the registry/version check and store the file if it is new content.

    The caller should run processing when `needs_processing(reg)` is true.
    """
    settings = get_settings()
    filename = Path(filename).name
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise UploadError(f"Unsupported file type '{ext}'. Allowed: {', '.join(settings.allowed_extensions)}")
    if not data:
        raise UploadError("File is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise UploadError(f"File exceeds {settings.max_upload_mb} MB limit.")

    sha256 = hashlib.sha256(data).hexdigest()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir / f"{uuid.uuid4().hex}_{_safe_name(filename)}"

    reg = repository.register_version(
        filename=filename,
        content_type=mimetypes.guess_type(filename)[0],
        size_bytes=len(data),
        sha256=sha256,
        stored_path=str(path),
        acl_roles=parse_roles(acl_roles),
    )
    store_file = reg.action in (NEW, UPDATED)
    restore_retry = False
    if reg.action == RETRY:
        version = repository.get_version(reg.version_id)
        if version is None:
            raise ValueError(f"Unknown document version {reg.version_id}")
        restore_retry = not Path(version["stored_path"]).is_file()
        store_file = restore_retry
    if store_file:
        try:
            path.write_bytes(data)
            if restore_retry:
                repository.set_version_stored_path(reg.version_id, str(path))
        except Exception:
            repository.set_version_status(reg.version_id, "failed", "Could not store file on disk.")
            raise
    logger.info("Registered %s v%s: %s", filename, reg.version, reg.action)
    return reg


def needs_processing(reg: Registration) -> bool:
    return reg.action in (NEW, UPDATED, RETRY)


def enqueue_processing(reg: Registration) -> bool:
    if not needs_processing(reg):
        return False
    from app.processing.tasks import process_document
    from app.vectorstore.milvus_store import _is_local_uri

    try:
        if _is_local_uri(get_settings().milvus_uri):
            raise RuntimeError("Start a shared Milvus server and configure MILVUS_URI as its URL.")
        process_document.delay(reg.version_id)
    except Exception as exc:
        logger.exception("Could not queue document version %s", reg.version_id)
        repository.mark_enqueue_failed(reg.version_id, f"Could not queue processing: {exc}"[:1000])
        raise
    return True


def describe(reg: Registration) -> str:
    return {
        NEW: f"new document registered (v{reg.version})",
        UPDATED: f"content changed, new version v{reg.version} created",
        UNCHANGED: f"content unchanged (v{reg.version}), metadata refreshed",
        RETRY: f"content unchanged but previous processing failed, retrying (v{reg.version})",
    }[reg.action]
