import threading
import math
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
    model = SentenceTransformer(s.embedding_model, device=_device(), trust_remote_code=False)
    model.max_seq_length = s.embedding_max_seq_length
    dim = model.get_embedding_dimension()
    if dim != s.embedding_dim:
        raise RuntimeError(f"Embedding model outputs {dim} dims, expected {s.embedding_dim}.")
    return model


def embed_texts(texts: list[str], *, task: str = "search_document") -> list[list[float]]:
    """Create normalized Nomic document or query vectors."""
    if not texts:
        return []
    if task not in ("search_document", "search_query"):
        raise ValueError(f"Unsupported embedding task: {task}")
    with _encode_lock, torch.inference_mode():
        model = _model()
        inputs = [f"{task}: {text}" for text in texts]
        lengths = [len(ids) for ids in model.tokenizer(
            inputs, truncation=False, padding=False, verbose=False
        )["input_ids"]]
        too_long = [(index, length) for index, length in enumerate(lengths) if length > model.max_seq_length]
        if too_long:
            index, length = too_long[0]
            raise ValueError(
                f"Embedding input {index} has {length} tokens, exceeding the "
                f"{model.max_seq_length}-token limit; split the source content."
            )
        vectors = model.encode(
            inputs,
            batch_size=get_settings().embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    result = vectors.tolist()
    if len(result) != len(texts) or any(
        len(vector) != get_settings().embedding_dim or not all(math.isfinite(value) for value in vector)
        for vector in result
    ):
        raise RuntimeError("Embedding output has invalid dimensions or non-finite values")
    return result
