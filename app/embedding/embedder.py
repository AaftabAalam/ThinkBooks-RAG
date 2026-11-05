import threading
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

_encode_lock = threading.Lock()


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    s = get_settings()
    model = SentenceTransformer(s.embedding_model, device=_device())
    model.max_seq_length = s.embedding_max_seq_length
    dim = model.get_sentence_embedding_dimension()
    if dim != s.embedding_dim:
        raise RuntimeError(f"Embedding model outputs {dim} dims, expected {s.embedding_dim}.")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Dense BGE-M3 embeddings (1024-d), L2-normalised so COSINE/IP search are equivalent."""
    if not texts:
        return []
    with _encode_lock, torch.inference_mode():
        vectors = _model().encode(
            texts,
            batch_size=get_settings().embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return vectors.tolist()
