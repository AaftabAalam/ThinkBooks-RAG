from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # ".env" in this project is the virtualenv directory, so config lives in ".env.local".
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env.local", extra="ignore")

    database_url: str = "postgresql://apple@localhost:5432/rag"
    # Vector store: a local file path uses Milvus Lite; use http://host:19530 for a Milvus server.
    milvus_uri: str = str(BASE_DIR / "data" / "milvus.db")
    milvus_collection: str = "document_chunks"
    redis_url: str = "redis://localhost:6379/0"

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    llm_model: str = "mlx-community/Qwen3.5-4B-MLX-4bit"

    chunk_tokenizer: str = "BAAI/bge-m3"
    chunk_text_tokens: int = 1024
    chunk_table_rows: int = 12
    embedding_batch_size: int = 8
    embedding_max_seq_length: int = 2048

    # Query pipeline
    semantic_cache_threshold: float = 0.90
    semantic_cache_max_entries: int = 1000
    cache_ttl_seconds: int = 86400
    retrieval_top_k: int = 30  # candidates per retriever and after RRF
    rrf_k: int = 60
    rerank_top_n: int = 5
    reranker_batch_size: int = 4
    reranker_max_length: int = 2048
    context_max_tokens: int = 6000
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7
    llm_top_p: float = 0.8
    llm_enable_thinking: bool = False

    upload_dir: Path = BASE_DIR / "data" / "uploads"
    max_upload_mb: int = 50
    allowed_extensions: tuple[str, ...] = (
        ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt", ".csv",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
