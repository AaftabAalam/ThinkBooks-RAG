import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.llm import generator


class StreamingTests(unittest.TestCase):
    def test_incremental_filter_matches_complete_output_across_boundaries(self):
        cases = [
            "  hello world  ",
            "start <think>hidden\nthought</think>  visible ",
            "<think>hidden</think>\nanswer<think>other</think> final",
            "before <think>unfinished",
            "literal <thin and <think not a tag",
            "  \n",
        ]
        for text in cases:
            expected = generator._THINK.sub("", text).strip()
            for width in range(1, len(text) + 1):
                fragments = iter(text[i:i + width] for i in range(0, len(text), width))
                with self.subTest(text=text, width=width):
                    self.assertEqual("".join(generator._visible_fragments(fragments)), expected)

    def test_stream_uses_matching_standalone_configuration(self):
        settings = SimpleNamespace(llm_enable_thinking=False, llm_max_tokens=123, llm_temperature=0.4, llm_top_p=0.9)
        tokenizer = unittest.mock.Mock()
        tokenizer.apply_chat_template.return_value = "prompt"
        responses = (SimpleNamespace(text=value, finish_reason=None) for value in ("a", "b"))
        sampler = object()
        with patch.object(generator, "get_settings", return_value=settings), \
             patch.object(generator, "_load", return_value=("model", tokenizer)), \
             patch.object(generator, "make_sampler", return_value=sampler) as make_sampler, \
             patch.object(generator, "stream_generate", return_value=responses) as stream:
            self.assertEqual("".join(generator.stream_standalone(" question ")), "ab")
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "question"}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        make_sampler.assert_called_once_with(temp=0.4, top_p=0.9)
        stream.assert_called_once_with("model", tokenizer, prompt="prompt", max_tokens=123, sampler=sampler)

    def test_closing_stream_releases_underlying_generator(self):
        closed = []

        def responses():
            try:
                yield SimpleNamespace(text="first", finish_reason=None)
                yield SimpleNamespace(text="second", finish_reason=None)
            finally:
                closed.append(True)

        settings = SimpleNamespace(llm_enable_thinking=False, llm_max_tokens=123, llm_temperature=0.4, llm_top_p=0.9)
        tokenizer = unittest.mock.Mock()
        with patch.object(generator, "get_settings", return_value=settings), \
             patch.object(generator, "_load", return_value=("model", tokenizer)), \
             patch.object(generator, "make_sampler"), \
             patch.object(generator, "stream_generate", side_effect=lambda *args, **kwargs: responses()):
            stream = generator.stream_standalone("question")
            self.assertEqual(next(stream), "first")
            stream.close()
        self.assertEqual(closed, [True])

    def test_limit_is_reported_in_stream(self):
        settings = SimpleNamespace(llm_enable_thinking=False, llm_max_tokens=1, llm_temperature=0.4, llm_top_p=0.9)
        tokenizer = unittest.mock.Mock()
        responses = (x for x in [SimpleNamespace(text="Partial answer", finish_reason="length")])
        with patch.object(generator, "get_settings", return_value=settings), \
             patch.object(generator, "_load", return_value=("model", tokenizer)), \
             patch.object(generator, "make_sampler"), \
             patch.object(generator, "stream_generate", return_value=responses):
            output = "".join(generator.stream_answer("question", "context"))
        self.assertIn("Partial answer", output)
        self.assertIn("may be incomplete", output)


if __name__ == "__main__":
    unittest.main()
