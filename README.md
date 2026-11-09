# RAG operations

The application uses PDFOxide and Chonkie for ingestion, Nomic embeddings,
Milvus dense/BM25 hybrid search, the Qwen reranker, and the configured MLX
generator. Parsing and model inference run locally; configured Postgres,
Milvus, or Redis services may be remote.

## Running the application

Install dependencies with `.env/bin/python -m pip install -r requirements.txt`,
and start PostgreSQL and Redis. To share the existing local index safely between
the application and the Celery worker, stop all processes currently using the
local Milvus Lite directory, then run these three commands in separate terminals
from the repository root:

```sh
OMP_NUM_THREADS=1 .env/bin/python -m milvus_lite server --data-dir data/milvus.db --host 127.0.0.1 --port 19530
OMP_NUM_THREADS=1 .env/bin/celery -A app.processing.tasks.celery_app worker --pool=solo --loglevel=INFO
OMP_NUM_THREADS=1 .env/bin/python -m uvicorn app.main:app
```

Open `http://127.0.0.1:8000/ui`. Both the application and worker must use the
same `MILVUS_URI` (default `http://127.0.0.1:19530`), `MILVUS_COLLECTION`,
Postgres database, Redis broker and shared `UPLOAD_DIR`. If `.env.local` or your
environment overrides `MILVUS_URI` with a local path, replace that override
with the shared server URL. Never run multiple Milvus Lite server processes
against the same data directory. A full Milvus server can also be used.

Uploads are registered and queued; they become searchable after the worker
finishes. Additional uploads can be submitted while processing continues.
Check `GET /api/documents` for pending, processing, processed or failed status
and errors. If queuing fails, the upload reports an error and can be retried.
Keep the worker running; queued jobs do not process without it. Use one solo
worker by default on Apple Silicon to avoid forking GPU libraries and competing
for limited memory. Same-filename versions are serialized.

The application loads the configured MLX model at startup and keeps it in
memory for the lifetime of that application process. First startup may take
time or download model files. Multiple web worker processes load separate
copies and need additional memory.

Nomic (`nomic-ai/nomic-embed-text-v1.5`) uses 768-dimensional vectors, a
batch size of 8, and an 8192-token input limit by default. It uses
`search_document:` and `search_query:` prefixes for their respective tasks.
The batch size can be overridden with `EMBEDDING_BATCH_SIZE` in `.env.local`
or the environment. PyTorch uses the Apple GPU through MPS when available,
otherwise it falls back to CPU.
Larger batches require more memory; reduce the batch size if inference runs
out of memory.

The configured embedding dimension is checked against the loaded model before
Milvus collection creation or loading. Inputs exceeding the model limit fail
explicitly rather than being silently truncated. Chonkie continues to use the
BGE-M3 tokenizer with 1024-token text chunks and 12-row table chunks.

## Diagnosing latency and answers

### macOS OpenMP crash

On macOS, if a query exits with `OMP: Error #179: Function pthread_mutex_init
failed` or crashes in `libomp`, stop the application and start a fresh process
with OpenMP limited to one thread:

```sh
OMP_NUM_THREADS=1 .env/bin/python -m uvicorn app.main:app
```

Set this environment variable on the command line **before** Python starts;
setting it after model or native library imports may be too late. Use the same
prefix for separate benchmark or maintenance commands that load these
libraries. This is a workaround, not a proven fix for every OpenMP failure,
and can reduce CPU throughput or change measured timings. It does not clear
Redis, Postgres, or Milvus data. Do not enable `KMP_DUPLICATE_LIB_OK` to hide
duplicate runtime warnings. If the crash persists, record the launch command,
Python/package versions, and the macOS crash report before investigating further.

An HTTP 404 for Hugging Face's optional `additional_chat_templates` lookup can
occur while loading a model and is not by itself the OpenMP crash; check for
an actual model-loading exception separately.

Ingestion logs contain `Ingestion version=... stage=... duration_ms=...` for
parsing, chunking, embedding, Milvus writes, Postgres persistence, and cache
invalidation. Initial model loading is included in the stage that first uses
the model. Compare cold and warm runs with the same documents.

Install the Python dependencies with `.env/bin/python -m pip install -r requirements.txt`.
PDFOxide handles PDF text; pdfplumber extracts line-detected tables and Tesseract
performs OCR on image-bearing PDF pages and embedded Office images. Install the
Tesseract executable and the language data specified by `OCR_LANGUAGE` (default
`eng`). OCR is enabled by default; `PDF_DO_OCR=false` disables it for all formats
and can omit image-only content. Restart the application after changes.
PDF images are not stored as image objects or given visual embeddings; OCR
extracts readable text from image-bearing pages, while non-text visual details
may be lost.

Office, HTML, CSV, Markdown and plain-text files use dedicated format readers,
not PDFOxide. PDF native text and separately extracted tables or OCR can overlap,
and table detection, formulas, reading order, inferred headings, OCR and complex
layouts are not guaranteed to match the original. HTML external images are not
fetched; Markdown is retained as source text rather than rendering linked images.
Spreadsheet chart layout and labels are not extracted, although cell values are
included. Check representative files and tables before relying on extraction.
Previously indexed documents keep their old chunks until reprocessed. A normal
upload of identical bytes is treated as unchanged and does not reprocess them.
Vectors from BGE-M3 and Nomic are incompatible. An old collection must not be
reused for Nomic, even if its vector dimension happens to match. The rebuild
command below reuses existing vectors and cannot migrate embedding models;
reprocess the original documents into a fresh Nomic collection instead.
After a Milvus reset, mark the existing versions for reprocessing and clear
their stale Postgres chunks. Re-uploading a failed version retries it; if its
stored upload is missing, the uploaded bytes restore it.

The Ask tab streams the answer and displays total time when complete. Disable
the **Use cache** checkbox when testing retrieval and answer quality. An exact
cache entry is still validated against document versions and roles. Answer
reuse is limited to the exact question, since similar questions may have
different answers. The former semantic cache setting no longer enables reuse.
The new cache namespace excludes answers generated under the old policy.
Cache entries are scoped by Milvus collection and model settings and expire according to
`CACHE_TTL_SECONDS`. Source numbers correspond to answer citations. To inspect
the stored text, use the document and chunk IDs from the query API response with
`GET /api/documents/{document_id}/chunks?limit=1000`.
After generation, numbered references are linked to their source metadata in
the API's `citations` list; `sources` still lists all passages supplied as
context. Both the answer and linked citations are cached when caching is
enabled. An out-of-range citation is reported as an error rather than linked
to the wrong source. Citation numbers do not independently verify factual support.

The Milvus Lite server creates its data directory if absent; an empty new
directory does not restore previously indexed documents from Postgres or stored
uploads. Do not start the worker with a local file `MILVUS_URI`: process-local
Milvus Lite instances cannot safely share one directory.

The default query path uses up to 15 fused hybrid-search candidates, reranks
up to four chunks and supplies up to 3,000 context tokens to
`mlx-community/Qwen2.5-3B-Instruct-4bit`. The model's configured maximum
generation is 4,500 tokens, and a response reaching that limit is marked as
possibly incomplete. The model's window is 32,768 tokens; the smaller retrieval
budget is a latency/coverage trade-off and may omit relevant information.
Search can return nearest candidates even for unrelated questions. Each
reranked chunk must meet `RERANK_MIN_SCORE` (default 0.5, greater than 0 and
at most 1) before it can enter the context. If none qualify, or the model
signals that the supplied context cannot answer, the response is an apology
with no sources or citations. An invalid or incomplete model answerability
marker is logged and also returns the apology without displaying model output.
A higher threshold can reject useful information.
This threshold is a heuristic, not a calibrated guarantee: the reranker and
model can misclassify relevance or follow malicious document instructions.
They do not prevent disclosure of unrelated content in all cases. Use proper
document access controls and isolation for confidential data; inspect answers
against source data.

New Milvus collections index headings together with chunk text for BM25 while
preserving the original chunk text separately. The reranker also sees these
headings. Existing collections continue to work with their original text-only
BM25 index until rebuilt. Search ranking
is approximate and a retrieved or cited passage is not by itself a guarantee
that a generated answer is correct.

Milvus limits each chunk's source text and heading-enriched search text to
65,535 UTF-8 bytes. Oversized chunks fail ingestion explicitly rather than
being silently truncated; reduce the chunk/table size or split the source.

## Rebuild an existing collection

Stop the application and all ingestion workers before rebuilding. Keep the
existing collection as the rollback copy; do not write to either collection
while rebuilding or switching. The command below creates a **new** collection
and reuses the existing dense vectors, without re-parsing documents or changing
the source collection:

```sh
.env/bin/python -m app.vectorstore.rebuild_index \
  --source document_chunks --target document_chunks_heading_v2
```

Use the actual existing collection name and an unused target name. The command
checks current Postgres document statuses and counts, compares every source
chunk and its metadata against Postgres, and verifies the rebuilt chunks,
vectors, and search text. A mismatch fails explicitly; a partially created
target is not activated or automatically removed. Rebuild during a maintenance
window to avoid concurrent changes invalidating those checks.

After checking representative search results, set `MILVUS_COLLECTION` in
`.env.local` to the new collection and restart the application. To roll back,
stop the application, set that value back to the original collection, and
restart. This direct rollback is only safe while both collections still match
Postgres; if documents were ingested after the switch, reconcile or rebuild the
old collection before using it again. Collection-specific cache keys prevent
an answer generated against one collection from being reused against the
other. Keep both collections until the new one has been verified against the
actual workload.
