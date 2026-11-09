import unittest
from unittest.mock import patch

from app.api.schemas import QueryRequest, QueryResponse
from app.llm import generator
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
        self.assertIn("not appear to be relevant", response.answer)
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
             patch.object(query_service, "stream_answer", side_effect=lambda *args: (s for s in ["Answer ", "[1]"])), \
             patch.object(query_service.redis_cache, "store") as store:
            response = query_service.answer_query(QueryRequest(query="question", use_cache=False))
        rerank.assert_called_once()
        store.assert_not_called()
        self.assertEqual(response.citations[0].number, 1)
        self.assertIn("rerank", response.timings_ms)
        self.assertGreaterEqual(response.timings_ms["total"], response.timings_ms["rerank"])

    def test_generation_prompts_keep_rag_and_standalone_separate(self):
        with patch.object(generator, "_generate", return_value="answer") as generate:
            generator.generate_answer("question", "context")
            rag_messages = generate.call_args.args[0]
            self.assertEqual(rag_messages[0]["content"], generator.SYSTEM_PROMPT)
            self.assertEqual(rag_messages[1]["content"], "Context:\ncontext\n\nQuestion: question")
            generator.generate_standalone(" question ")
            self.assertEqual(generate.call_args.args[0], [{"role": "user", "content": "question"}])

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


if __name__ == "__main__":
    unittest.main()
