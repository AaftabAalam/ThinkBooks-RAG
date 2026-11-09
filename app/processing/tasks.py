from celery import Celery

from app.core.config import get_settings
from app.processing.pipeline import process_version
from app.vectorstore.milvus_store import _is_local_uri

celery_app = Celery("rag", broker=get_settings().redis_url)
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_ignore_result=True,
    task_default_queue=get_settings().celery_queue,
    broker_connection_retry_on_startup=True,
)


@celery_app.task(name="rag.process_version")
def process_document(version_id: str) -> int:
    if _is_local_uri(get_settings().milvus_uri):
        raise RuntimeError("Celery requires a shared Milvus server URL, not a local Milvus Lite path.")
    return process_version(version_id)
