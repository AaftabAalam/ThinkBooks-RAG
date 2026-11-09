import logging
import threading
import time
from functools import lru_cache
from pathlib import Path

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.core.config import get_settings
from app.embedding.embedder import _model, embed_texts
from app.retrieval.context import retrieval_text

logger = logging.getLogger(__name__)

_TEXT_MAX_BYTES = 65535
_init_lock = threading.Lock()
_OUTPUT_FIELDS = ["chunk_id", "document_id", "filename", "version", "chunk_type", "chunk_index", "text", "metadata"]


def _is_local_uri(uri: str) -> bool:
    return "://" not in uri and not uri.startswith("unix:")


def _connect(uri: str) -> MilvusClient:
    if _is_local_uri(uri):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    return MilvusClient(
        uri=uri,
        dedicated=True,
        grpc_options={
            "grpc.keepalive_time_ms": 120000,
            "grpc.keepalive_permit_without_calls": False,
        },
    )


@lru_cache(maxsize=1)
def _client() -> MilvusClient:
    uri = get_settings().milvus_uri
    client = _connect(uri)
    _ensure_collection(client)
    return client


def _wait_for_indexes(client: MilvusClient, collection: str, names: list[str], timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    pending = set(names)
    while pending:
        for name in list(pending):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for Milvus indexes: {', '.join(sorted(pending))}")
            info = client.describe_index(collection, name, timeout=remaining)
            if info is None:
                continue
            if info["state"] == "Finished":
                pending.remove(name)
            elif info["state"] == "Failed":
                raise RuntimeError(f"Milvus index {name} failed: {info.get('index_state_fail_reason', info)}")
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Milvus indexes: {', '.join(sorted(pending))}")
            time.sleep(0.5)


def _ensure_collection(client: MilvusClient) -> None:
    """Create or load a collection with dense vectors and Milvus-generated BM25 vectors."""
    s = get_settings()
    model_dimension = _model().get_embedding_dimension()
    if model_dimension != s.embedding_dim:
        raise RuntimeError(
            f"Embedding model outputs {model_dimension} dimensions, but Milvus is configured for {s.embedding_dim}"
        )
    with _init_lock:
        if client.has_collection(s.milvus_collection):
            description = client.describe_collection(s.milvus_collection)
            field_info = {f["name"]: f for f in description["fields"]}
            fields = set(field_info)
            vector_dim = field_info.get("vector", {}).get("params", {}).get("dim")
            if vector_dim is None or int(vector_dim) != model_dimension:
                raise RuntimeError(
                    f"Collection {s.milvus_collection} vector dimension {vector_dim} "
                    f"does not match embedding model dimension {model_dimension}; rebuild the collection."
                )
            if "sparse" in fields:
                if "search_text" not in fields:
                    logger.warning(
                        "Collection %s uses the legacy text-only BM25 index; rebuild it to search headings.",
                        s.milvus_collection,
                    )
                client.load_collection(s.milvus_collection)
                return
            raise RuntimeError(
                f"Collection {s.milvus_collection} has no sparse field; "
                "rebuild it in a separate collection instead of deleting existing data."
            )

        probe = embed_texts(["Embedding dimension validation"])
        if len(probe) != 1 or len(probe[0]) != model_dimension:
            raise RuntimeError("Embedding output dimension does not match the model dimension")
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=model_dimension)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("document_id", DataType.VARCHAR, max_length=64)
        schema.add_field("filename", DataType.VARCHAR, max_length=512)
        schema.add_field("version", DataType.INT64)
        schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=_TEXT_MAX_BYTES)
        schema.add_field("search_text", DataType.VARCHAR, max_length=_TEXT_MAX_BYTES, enable_analyzer=True)
        schema.add_field("metadata", DataType.JSON)
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["search_text"],
                output_field_names=["sparse"],
            )
        )

        index = client.prepare_index_params()
        index.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        if _is_local_uri(s.milvus_uri):
            client.create_collection(s.milvus_collection, schema=schema)
            client.create_index(s.milvus_collection, index, sync=False)
            _wait_for_indexes(client, s.milvus_collection, ["vector", "sparse"])
        else:
            client.create_collection(s.milvus_collection, schema=schema, index_params=index)
        client.load_collection(s.milvus_collection)


@lru_cache(maxsize=1)
def _supports_search_text() -> bool:
    fields = _client().describe_collection(get_settings().milvus_collection)["fields"]
    return any(field["name"] == "search_text" for field in fields)


def upsert_chunks(chunks: list[dict], vectors: list[list[float]]) -> None:
    """Upsert by stable chunk_id; the BM25 sparse vector is generated by Milvus on insert."""
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")
    searchable_headings = _supports_search_text()
    rows = []
    for c, v in zip(chunks, vectors):
        m = c["metadata"]
        if len(c["text"].encode("utf-8")) > _TEXT_MAX_BYTES:
            raise ValueError(
                f"Chunk {c['chunk_id']} exceeds the Milvus text limit of {_TEXT_MAX_BYTES} UTF-8 bytes"
            )
        row = {
            "chunk_id": c["chunk_id"],
            "vector": v,
            "document_id": m["document_id"],
            "filename": m["filename"],
            "version": m["version"],
            "content_hash": m["content_hash"],
            "chunk_type": c["chunk_type"],
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "metadata": m,
        }
        if searchable_headings:
            search_text = retrieval_text(c["text"], m)
            if len(search_text.encode("utf-8")) > _TEXT_MAX_BYTES:
                raise ValueError(
                    f"Search text for chunk {c['chunk_id']} exceeds the Milvus limit of {_TEXT_MAX_BYTES} UTF-8 bytes"
                )
            row["search_text"] = search_text
        rows.append(row)
    client = _client()
    for i in range(0, len(rows), 200):
        client.upsert(get_settings().milvus_collection, rows[i : i + 200])


def delete_stale(document_id: str, keep_chunk_ids: list[str]) -> None:
    """Remove vectors of a document that are not part of its current version."""
    keep = ", ".join(f'"{cid}"' for cid in keep_chunk_ids)
    _client().delete(
        get_settings().milvus_collection,
        filter=f'document_id == "{document_id}" and chunk_id not in [{keep}]',
    )


def count_for_document(document_id: str) -> int:
    rows = _client().query(
        get_settings().milvus_collection,
        filter=f'document_id == "{document_id}"',
        output_fields=["chunk_id"],
        limit=16384,
    )
    return len(rows)


def hybrid_search(
    query_text: str,
    query_vector: list[float],
    *,
    document_ids: list[str],
    limit: int | None = None,
) -> list[dict]:
    """Dense (cosine) + BM25 retrieval fused with Reciprocal Rank Fusion, restricted to readable documents."""
    if not document_ids:
        return []
    s = get_settings()
    limit = limit or s.retrieval_top_k
    ids = ", ".join(f'"{d}"' for d in document_ids)
    expr = f"document_id in [{ids}]"
    dense = AnnSearchRequest(
        data=[query_vector], anns_field="vector", param={"metric_type": "COSINE"}, limit=limit, expr=expr
    )
    sparse = AnnSearchRequest(
        data=[query_text], anns_field="sparse", param={"metric_type": "BM25"}, limit=limit, expr=expr
    )
    results = _client().hybrid_search(
        s.milvus_collection,
        reqs=[dense, sparse],
        ranker=RRFRanker(s.rrf_k),
        limit=limit,
        output_fields=_OUTPUT_FIELDS,
    )
    hits = []
    for h in results[0]:
        entity = dict(h["entity"])
        entity["rrf_score"] = float(h["distance"])
        hits.append(entity)
    return hits
