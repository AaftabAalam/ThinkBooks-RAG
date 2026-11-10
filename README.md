# ThinkBooks RAG

A local-first document question-answering application. It accepts documents,
extracts and chunks their content, indexes them for hybrid search, and generates
answers with source citations. FastAPI exposes the API and hosts a Gradio UI;
Celery processes uploads in the background.

The current implementation uses PostgreSQL for document metadata and chunks,
Redis for the task broker and answer cache, Milvus for dense and BM25 search,
Nomic embeddings, a Qwen reranker, and an MLX language model. Model inference
and document parsing run locally, although the configured services can be
remote. See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete file-by-file
architecture, settings, data model, and request flows.

## Prerequisites

- macOS with Apple Silicon for the default MLX generator; a compatible Python
  environment and the dependencies in [requirements.txt](requirements.txt).
  Python 3.10 or later is required by the source syntax; the repository does
  not declare a tested Python version range or lock dependency versions.
- Running PostgreSQL and Redis instances, and a database for this application.
- Tesseract and the language data for `OCR_LANGUAGE` (default `eng`) when OCR
  is enabled. Set `PDF_DO_OCR=false` to disable OCR for all supported formats.
- Access to download the configured Hugging Face models on first use and
  sufficient local memory and disk space for the models and indexes.

## Setup and run

From the repository root, create a virtual environment and install dependencies:

```sh
python3 -m venv .env
.env/bin/python -m pip install -r requirements.txt
cp .env.local.example .env.local
```

Edit `.env.local` for your PostgreSQL user/database and local paths. For
example, `createdb rag` creates a database using your current PostgreSQL
connection settings. The example database URL contains a machine-specific
username and is not portable as-is. The database user must be able to create
tables and indexes. Start PostgreSQL and Redis using your local installation.
Application startup creates the PostgreSQL tables; it does not create the
database itself.

Run these commands in **separate terminals**, from the repository root:

```sh
OMP_NUM_THREADS=1 .env/bin/python -m milvus_lite server --data-dir data/milvus.db --host 127.0.0.1 --port 19530
```

```sh
OMP_NUM_THREADS=1 .env/bin/celery -A app.processing.tasks.celery_app worker --pool=solo --loglevel=INFO
```

```sh
OMP_NUM_THREADS=1 .env/bin/python -m uvicorn app.main:app
```

Open [the UI](http://127.0.0.1:8000/ui) or
[interactive API documentation](http://127.0.0.1:8000/docs). The MLX
generator loads at application startup, which can take time on a cold start.
The embedding and reranking models load when first needed. Each web or worker
process has its own model memory; the default single solo worker avoids
forking GPU libraries and competing for memory on Apple Silicon.

The web process and worker must share `DATABASE_URL`, `REDIS_URL`,
`CELERY_QUEUE`, `MILVUS_URI`, `MILVUS_COLLECTION`, relevant model settings,
and the same accessible `UPLOAD_DIR`. The queued worker rejects a local-file
Milvus URI; use the shared server URL, as above, or a full Milvus server.
Never start multiple Milvus Lite servers on the same data directory. An empty
Milvus directory does not restore previously indexed data.

## Use

1. In **Upload**, select one or more `.pdf`, `.docx`, `.pptx`, `.xlsx`,
   `.html`, `.md`, `.txt`, or `.csv` files and enter comma-separated allowed
   roles. The default maximum is 50 MiB per file.
2. Uploads are registered and queued, not immediately searchable. Keep the
   worker running and check `GET /api/documents` until the current version is
   `processed`; inspect its `error` if it is `failed`.
3. In **Ask**, provide a question and comma-separated roles. The UI streams
   the answer and reports total time. Disable **Use cache** when comparing
   retrieval or answer quality.

For example:

```sh
curl -F 'files=@example.pdf' -F 'acl_roles=public' http://127.0.0.1:8000/api/documents/upload
curl http://127.0.0.1:8000/api/documents
curl -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does the document say?","roles":["public"],"use_cache":true}'
```

`GET /api/documents/{document_id}/chunks?limit=1000` returns stored chunk
text and metadata. The query API returns the answer, all sources supplied as
context, linked numbered citations, cache status, and timing data. The HTTP
query endpoint returns a complete response; streaming is used by the UI.
`GET /health` returns a simple process response, not a dependency readiness
check.

> **Security:** Roles are supplied by the caller, not authenticated. Document
> listing and chunk endpoints also have no access-control checks. This is not
> a security boundary or a multi-tenant deployment: restrict access to trusted
> users and networks, and add authentication and server-enforced authorization
> before exposing it or storing confidential data.

## Configuration and operations

Settings are defined in [app/core/config.py](app/core/config.py). Values can
be supplied as uppercase environment variables or in `.env.local`; environment
variables take precedence. Restart affected processes after changing settings.
[.env.local.example](.env.local.example) is a starting point, not the complete
configuration reference. All defaults and their effects are documented in
[ARCHITECTURE.md](ARCHITECTURE.md#configuration-reference).

Same-filename uploads are versioned by SHA-256. Identical processed, pending,
or processing content is not reprocessed; identical failed content can be
retried. New versions and processing are serialized per filename. If queueing
fails, the version is marked failed and can be retried. Ingestion logs record
per-stage timings.

OCR, table extraction, document reading order, inferred headings, formulas,
charts, and citations can be imperfect. PDF images are OCRed, not stored as
image objects or visually embedded; HTML external images are not fetched.
Retrieved passages and citations do not guarantee a correct or supported
answer. The reranker threshold and prompt instructions do not guarantee
protection against irrelevant content or prompt injection.

On macOS, if a process crashes in OpenMP with `OMP: Error #179`, start a fresh
process with `OMP_NUM_THREADS=1` set **before** Python starts, as in the
commands above. This is a workaround, not a universal fix, and can affect
performance. Do not use `KMP_DUPLICATE_LIB_OK` to hide duplicate runtime
warnings.

### Rebuild a heading-aware index

For an existing compatible collection, stop the application and all workers;
do not write to either collection during the rebuild and switch. Keep the
original collection as a rollback copy:

```sh
OMP_NUM_THREADS=1 .env/bin/python -m app.vectorstore.rebuild_index \
  --source document_chunks --target document_chunks_heading_v2
```

Use the actual source and an unused target. The command checks current
PostgreSQL versions, chunks, metadata, and source vectors, then verifies the
new collection. A failure may leave a partially created target; it is not
activated or removed automatically. After testing, set `MILVUS_COLLECTION` to
the target and restart both application and worker. Rollback to the old
collection is safe only while it still matches PostgreSQL; reconcile or
rebuild it if subsequent ingestion changed data.

This rebuild **reuses vectors**; it does not migrate embedding models. BGE-M3
and Nomic vectors are incompatible even if dimensions match. To change
embedding models, reprocess original documents into a fresh collection and
reconcile the PostgreSQL registry; an identical normal upload is otherwise
treated as unchanged. After a Milvus reset, existing versions must be marked
for reprocessing and stale PostgreSQL chunks cleared before re-uploading; the
repository does not provide an automated reset/migration command.

## Tests

The repository uses `unittest` with mocks for its focused tests:

```sh
.env/bin/python -m unittest discover -s tests -v
```

These tests do not establish end-to-end correctness against real services,
models, or representative documents. See the architecture guide for their
coverage and operational limitations.
