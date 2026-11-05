from fastapi import APIRouter

from app.api.schemas import QueryRequest, QueryResponse
from app.services.query_service import answer_query

router = APIRouter(prefix="/query", tags=["query"])


# Sync handler: FastAPI runs it in a worker thread so model inference doesn't block the event loop.
@router.post("", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    return answer_query(req)
