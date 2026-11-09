import unittest

from app.retrieval.context import assemble_context


class ContextTests(unittest.TestCase):
    def test_first_oversized_chunk_is_trimmed_to_budget(self):
        hit = {"filename": "file", "text": "a" * 100, "chunk_type": "text"}
        context, used = assemble_context([hit], 30, len)
        self.assertLessEqual(len(context), 30)
        self.assertEqual(len(used), 1)
        self.assertLess(len(used[0]["text"]), len(hit["text"]))

    def test_budget_stops_before_second_oversized_chunk(self):
        hit = {"filename": "file", "text": "1234567890", "chunk_type": "text"}
        context, used = assemble_context([hit, hit], 30, len)
        self.assertLessEqual(len(context), 30)
        self.assertEqual(len(used), 1)


if __name__ == "__main__":
    unittest.main()
