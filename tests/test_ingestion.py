import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.db.repository import NEW, UNCHANGED, Registration
from app.services.document_service import enqueue_processing


class IngestionQueueTests(unittest.TestCase):
    def setUp(self):
        settings = patch("app.services.document_service.get_settings",
                         return_value=SimpleNamespace(milvus_uri="http://127.0.0.1:19530"))
        settings.start()
        self.addCleanup(settings.stop)

    def test_new_document_is_queued(self):
        registration = Registration(NEW, "document", "version", 1)
        with patch("app.processing.tasks.process_document.delay") as delay:
            self.assertTrue(enqueue_processing(registration))
        delay.assert_called_once_with("version")

    def test_unchanged_document_is_not_queued(self):
        registration = Registration(UNCHANGED, "document", "version", 1)
        with patch("app.processing.tasks.process_document.delay") as delay:
            self.assertFalse(enqueue_processing(registration))
        delay.assert_not_called()

    def test_broker_error_is_recorded_and_propagated(self):
        registration = Registration(NEW, "document", "version", 1)
        with patch("app.processing.tasks.process_document.delay", side_effect=RuntimeError("broker unavailable")), \
             patch("app.services.document_service.repository.mark_enqueue_failed") as failed:
            with self.assertRaisesRegex(RuntimeError, "broker unavailable"):
                enqueue_processing(registration)
        failed.assert_called_once()

    def test_local_store_is_rejected_and_marked_failed(self):
        registration = Registration(NEW, "document", "version", 1)
        with patch("app.services.document_service.get_settings",
                   return_value=SimpleNamespace(milvus_uri="data/milvus.db")), \
             patch("app.processing.tasks.process_document.delay") as delay, \
             patch("app.services.document_service.repository.mark_enqueue_failed") as failed:
            with self.assertRaisesRegex(RuntimeError, "shared Milvus server"):
                enqueue_processing(registration)
        delay.assert_not_called()
        failed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
