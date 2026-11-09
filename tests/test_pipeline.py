import unittest
from contextlib import contextmanager
from unittest.mock import patch

from app.processing import pipeline


@contextmanager
def connection():
    class Connection:
        def transaction(self):
            return transaction()

        def execute(self, *args):
            return None

    yield Connection()


@contextmanager
def transaction():
    yield


class Pool:
    def connection(self):
        return connection()


class PipelineTests(unittest.TestCase):
    def test_superseded_version_never_changes_vectors(self):
        version = {
            "filename": "file.txt", "document_id": "doc", "version": 1,
            "sha256": "hash", "stored_path": "file.txt", "current_version": 1,
            "status": "pending",
        }
        stale = {**version, "current_version": 2, "status": "superseded"}
        chunk = {"text": "old"}
        with patch.object(pipeline, "get_pool", return_value=Pool()), \
             patch.object(pipeline.repository, "get_version", side_effect=[version, version, stale]), \
             patch.object(pipeline.repository, "claim_version", return_value=True), \
             patch.object(pipeline, "parse_document", return_value=["block"]), \
             patch.object(pipeline, "chunk_blocks", return_value=[chunk]), \
             patch.object(pipeline, "embed_texts", return_value=[[1.0]]), \
             patch.object(pipeline.milvus_store, "upsert_chunks") as upsert, \
             patch.object(pipeline.repository, "replace_chunks") as persist:
            self.assertEqual(pipeline.process_version("old-version"), 0)
        upsert.assert_not_called()
        persist.assert_not_called()

    def test_processed_version_is_not_reprocessed(self):
        version = {"filename": "file.txt", "version": 1, "current_version": 1, "status": "processed"}
        with patch.object(pipeline, "get_pool", return_value=Pool()), \
             patch.object(pipeline.repository, "get_version", return_value=version), \
             patch.object(pipeline.repository, "claim_version", return_value=False), \
             patch.object(pipeline, "parse_document") as parse:
            self.assertEqual(pipeline.process_version("version"), 0)
        parse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
