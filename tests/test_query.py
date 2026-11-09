import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from app.api.schemas import QueryRequest, QueryResponse
from app.cache import redis_cache
from app.core.config import Settings
from app.llm import generator
from app.retrieval import reranker
from app.services import query_service
from app.ui import gradio_app


class QueryTimingTests(unittest.TestCase):
    def test_cache_hit_has_total_but_no_reranker(self):
        source = {
            "document_id": "doc", "filename": "file.pdf", "version": 1,
            "chunk_id": "chunk", "chunk_type": "text",
        }
        with patch.object(query_service.redis_cache, "get_exact", return_value={"answer": "Answer [1]", "sources": [source]}), \
             patch.object(query_service.repository, "validate_sources", return_value=True), \
             patch.object(query_service, "rerank") as rerank:
            response = query_service.answer_query(QueryRequest(query="question"))
        self.assertEqual(response.cache, "exact")
        self.assertIn("total", response.timings_ms)
        self.assertNotIn("rerank", response.timings_ms)
        rerank.assert_not_called()

    def test_no_hits_has_total_but_no_reranker(self):
        with patch.object(query_service, "embed_texts", return_value=[[1.0]]), \
             patch.object(query_service.repository, "accessible_document_ids", return_value=["doc"]), \
             patch.object(query_service.milvus_store, "hybrid_search", return_value=[]), \
             patch.object(query_service, "rerank") as rerank, \
             patch.object(query_service.redis_cache, "get_exact") as cache:
            response = query_service.answer_query(QueryRequest(query="question", use_cache=False))
        self.assertIn("total", response.timings_ms)
        self.assertNotIn("rerank", response.timings_ms)
        self.assertEqual(response.answer, query_service.NO_CONTENT_ANSWER)
        self.assertEqual((response.sources, response.citations), ([], []))
        rerank.assert_not_called()
        cache.assert_not_called()

    def test_rag_reranks_and_reports_total_without_cache(self):
        hit = {
            "document_id": "doc", "filename": "file.pdf", "version": 1,
            "chunk_id": "chunk", "chunk_type": "text", "rerank_score": 0.9,
        }
        with patch.object(query_service, "embed_texts", return_value=[[1.0]]), \
             patch.object(query_service.repository, "accessible_document_ids", return_value=["doc"]), \
             patch.object(query_service.milvus_store, "hybrid_search", return_value=[hit]), \
             patch.object(query_service, "rerank", return_value=[hit]) as rerank, \
             patch.object(query_service, "assemble_context", return_value=("context", [hit])), \
             patch.object(query_service, "stream_answer", side_effect=lambda *args: (s for s in ["<ANS", "WER>Answer ", "[1]"])), \
             patch.object(query_service.redis_cache, "store") as store:
            response = query_service.answer_query(QueryRequest(query="question", use_cache=False))
        rerank.assert_called_once()
        store.assert_not_called()
        self.assertEqual(response.citations[0].number, 1)
        self.assertIn("rerank", response.timings_ms)
        self.assertGreaterEqual(response.timings_ms["total"], response.timings_ms["rerank"])

    def test_generation_prompts_keep_rag_and_standalone_separate(self):
        with patch.object(generator, "_generate", return_value="<ANSWER>answer") as generate:
            generator.generate_answer("question", "context")
            rag_messages = generate.call_args.args[0]
            self.assertEqual(rag_messages[0]["content"], generator.SYSTEM_PROMPT)
            self.assertEqual(rag_messages[1]["content"], "Context:\ncontext\n\nQuestion: question")
            generator.generate_standalone(" question ")
            self.assertEqual(generate.call_args.args[0], [{"role": "user", "content": "question"}])

    def test_non_streaming_generator_accepts_optional_answer_prefix(self):
        for output, expected in (
            ("ANSWER: <NO_ANSWER>unrelated text", generator.NO_CONTENT_ANSWER),
            ("ANSWER: <ANSWER>supported answer", "supported answer"),
        ):
            with self.subTest(output=output), patch.object(generator, "_generate", return_value=output):
                self.assertEqual(generator.generate_answer("question", "context"), expected)

    def test_non_streaming_invalid_marker_is_logged_and_refused(self):
        with patch.object(generator, "_generate", return_value="NO_ANSWER"), \
             self.assertLogs(generator.logger, level="WARNING"):
            self.assertEqual(generator.generate_answer("question", "context"), generator.NO_CONTENT_ANSWER)

    def test_ui_streams_and_reports_total(self):
        response = QueryResponse(
            answer="answer", cache="miss", normalized_query="question",
            sources=[], timings_ms={"total": 42},
        )
        with patch.object(gradio_app, "stream_query", side_effect=lambda req: (x for x in [
            ("ans", None), ("wer", None), ("", response),
        ])):
            events = list(gradio_app._handle_query("question", "public", False))
        self.assertEqual(events, [("ans", "Generating..."), ("answer", "Generating..."),
                                  ("answer", "Total time: 0.04 s")])

    def test_ui_no_hits_delivers_answer_once(self):
        response = QueryResponse(
            answer="No matching content.", cache="miss", normalized_query="question",
            sources=[], timings_ms={"total": 42},
        )
        with patch.object(gradio_app, "stream_query", side_effect=lambda req: (x for x in [
            (response.answer, response),
        ])):
            events = list(gradio_app._handle_query("question", "public", False))
        self.assertEqual(events, [("No matching content.", "Total time: 0.04 s")])


class RelevanceTests(unittest.TestCase):
    def setUp(self):
        self.hit = {
            "document_id": "doc", "filename": "private.pdf", "version": 1,
            "chunk_id": "chunk", "chunk_type": "text", "text": "secret text",
            "rerank_score": 0.1,
        }

    def run_query(self, ranked, output=("<ANSWER>answer [1]",), use_cache=False):
        with ExitStack() as stack:
            stack.enter_context(patch.object(query_service, "get_settings", return_value=SimpleNamespace(
                retrieval_top_k=15, rerank_top_n=4, rerank_min_score=0.5, context_max_tokens=3000,
            )))
            stack.enter_context(patch.object(query_service, "embed_texts", return_value=[[1.0]]))
            stack.enter_context(patch.object(query_service.repository, "accessible_document_ids", return_value=["doc"]))
            stack.enter_context(patch.object(query_service.milvus_store, "hybrid_search", return_value=[self.hit]))
            stack.enter_context(patch.object(query_service, "rerank", return_value=ranked))
            assemble = stack.enter_context(patch.object(
                query_service, "assemble_context", side_effect=lambda hits, *args: ("context", hits),
            ))
            stream = stack.enter_context(patch.object(query_service, "stream_answer", return_value=(x for x in output)))
            stack.enter_context(patch.object(query_service.redis_cache, "get_exact", return_value=None))
            store = stack.enter_context(patch.object(query_service.redis_cache, "store"))
            events = list(query_service.stream_query(QueryRequest(query="Question?", use_cache=use_cache)))
        return events, assemble, stream, store

    def test_low_relevance_returns_only_apology_without_generation(self):
        events, assemble, stream, store = self.run_query([self.hit], use_cache=True)
        self.assertEqual(len(events), 1)
        text, response = events[0]
        self.assertEqual(text, query_service.NO_CONTENT_ANSWER)
        self.assertEqual(response.answer, text)
        self.assertEqual((response.sources, response.citations), ([], []))
        assemble.assert_not_called()
        stream.assert_not_called()
        store.assert_not_called()

    def test_only_qualifying_chunks_reach_context_and_sources(self):
        good = {**self.hit, "filename": "relevant.pdf", "chunk_id": "good", "rerank_score": 0.5}
        events, assemble, _, store = self.run_query([good, self.hit], use_cache=True)
        self.assertEqual(assemble.call_args.args[0], [good])
        self.assertEqual([x.filename for x in events[-1][1].sources], ["relevant.pdf"])
        self.assertEqual("".join(text for text, _ in events), "answer [1]")
        self.assertEqual(store.call_args.args[0], "Question?")
        self.assertNotIn("private.pdf", str(events))

    def test_model_refusal_discards_remaining_text_and_sources(self):
        good = {**self.hit, "rerank_score": 0.9}
        for output in (("<NO_", "ANSWER>secret text [1]"),
                       ("ANS", "WER:", " <NO_ANSWER>", "secret text [1]")):
            with self.subTest(output=output):
                events, _, _, store = self.run_query([good], output, use_cache=True)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0][0], query_service.NO_CONTENT_ANSWER)
                self.assertEqual((events[0][1].sources, events[0][1].citations), ([], []))
                store.assert_not_called()

    def test_invalid_or_incomplete_marker_is_logged_and_refused(self):
        good = {**self.hit, "rerank_score": 0.9}
        for output in (("private information",), ("NO_", "ANSWER"), ("<ANS",), ()):
            with self.subTest(output=output), self.assertLogs(query_service.logger, level="WARNING"):
                events, _, _, store = self.run_query([good], output, use_cache=True)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0][0], query_service.NO_CONTENT_ANSWER)
                self.assertEqual((events[0][1].sources, events[0][1].citations), ([], []))
                store.assert_not_called()

    def test_invalid_scores_fail_before_any_text(self):
        for score in (None, float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "relevance score"):
                self.run_query([{**self.hit, "rerank_score": score}])

    def test_reranker_rejects_missing_and_invalid_scores(self):
        for scores in ([], [float("nan")], [2.0]):
            with self.subTest(scores=scores), \
                 patch.object(reranker, "rerank_scores", return_value=scores), \
                 self.assertRaisesRegex(ValueError, "invalid relevance"):
                reranker.rerank("question", [self.hit], 1)

    def test_threshold_validation(self):
        for value in (0, -0.1, 1.1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings(rerank_min_score=value)
        self.assertEqual(Settings(rerank_min_score=1).rerank_min_score, 1)

    def test_cache_namespace_excludes_older_entries_and_includes_threshold(self):
        settings = Settings(rerank_min_score=0.5)
        with patch.object(redis_cache, "get_settings", return_value=settings):
            namespace = redis_cache._namespace()
            key = redis_cache._exact_key("Question?")
            self.assertIn(":v6:", namespace)
            self.assertNotEqual(key, redis_cache._exact_key("question"))
            settings.rerank_min_score = 0.6
            self.assertNotEqual(namespace, redis_cache._namespace())


if __name__ == "__main__":
    unittest.main()
