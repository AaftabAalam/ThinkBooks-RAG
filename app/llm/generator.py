import re
import threading
from functools import lru_cache

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from app.core.config import get_settings

_lock = threading.Lock()
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using ONLY the provided context. "
    "Understand the question, reason over the context, and write a clear, natural, complete answer. "
    "Cite passages that directly support each claim inline as [1], [2], etc. "
    "When identifying a document's authors, distinguish its own author credits from authors "
    "of works cited in its references. "
    "If the context does not contain the answer, say so plainly instead of guessing."
)


@lru_cache(maxsize=1)
def _load():
    return load(get_settings().llm_model)


def count_tokens(text: str) -> int:
    _, tokenizer = _load()
    return len(tokenizer.encode(text))


def generate_answer(question: str, context: str) -> str:
    s = get_settings()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    with _lock:
        model, tokenizer = _load()
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=s.llm_enable_thinking,
        )
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=s.llm_max_tokens,
            sampler=make_sampler(temp=s.llm_temperature, top_p=s.llm_top_p),
            verbose=False,
        )
    return _THINK.sub("", text).strip()
