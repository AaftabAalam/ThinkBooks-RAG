import inspect
import threading
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import get_settings
from app.retrieval.context import retrieval_text

_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load():
    s = get_settings()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(s.reranker_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(s.reranker_model, dtype=dtype).to(device).eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "last_logits_only": "logits_to_keep" in inspect.signature(model.forward).parameters,
        "device": device,
        "yes": tokenizer.convert_tokens_to_ids("yes"),
        "no": tokenizer.convert_tokens_to_ids("no"),
        "prefix": tokenizer.encode(_PREFIX, add_special_tokens=False),
        "suffix": tokenizer.encode(_SUFFIX, add_special_tokens=False),
    }


def rerank_scores(query: str, documents: list[str]) -> list[float]:
    """Relevance probability (P('yes')) of each document for the query using Qwen3-Reranker."""
    if not documents:
        return []
    s = get_settings()
    with _lock, torch.inference_mode():
        m = _load()
        tok = m["tokenizer"]
        budget = s.reranker_max_length - len(m["prefix"]) - len(m["suffix"])
        scores: list[float] = []
        for i in range(0, len(documents), s.reranker_batch_size):
            bodies = [
                f"<Instruct>: {_INSTRUCTION}\n<Query>: {query}\n<Document>: {d}"
                for d in documents[i : i + s.reranker_batch_size]
            ]
            ids = tok(bodies, add_special_tokens=False, truncation=True, max_length=budget)["input_ids"]
            ids = [m["prefix"] + x + m["suffix"] for x in ids]
            batch = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(m["device"])
            kwargs = {"logits_to_keep": 1} if m["last_logits_only"] else {}
            logits = m["model"](**batch, **kwargs).logits[:, -1, :].float()
            pair = torch.stack([logits[:, m["no"]], logits[:, m["yes"]]], dim=1)
            scores.extend(torch.softmax(pair, dim=1)[:, 1].tolist())
    return scores


def rerank(query: str, hits: list[dict], top_n: int | None = None) -> list[dict]:
    top_n = top_n or get_settings().rerank_top_n
    scores = rerank_scores(query, [retrieval_text(h["text"], h.get("metadata") or {}) for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)[:top_n]
    return [{**h, "rerank_score": sc} for h, sc in ranked]
