# RAG operations

The application uses PDFOxide and Chonkie for ingestion, Nomic embeddings,
Milvus dense/BM25 hybrid search, the Qwen reranker, and the configured MLX
generator. Parsing and model inference run locally; configured Postgres,
Milvus, or Redis services may be remote.

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

The Ask tab displays cache type, per-stage query timings, and sources. Disable
the **Use cache** checkbox when testing retrieval and answer quality. An exact
cache entry is still validated against document versions and roles. Semantic
answer reuse is disabled by default because similar questions may have
different answers; `SEMANTIC_CACHE_ENABLED=true` opts back in. Cache entries
are scoped by Milvus collection and expire according to `CACHE_TTL_SECONDS`.
Source numbers correspond to answer citations. To inspect the stored text, use
the document and chunk IDs shown in the Ask tab with
`GET /api/documents/{document_id}/chunks?limit=1000`.
After generation, numbered references are linked to their source metadata in
the API's `citations` list; `sources` still lists all passages supplied as
context. Both the answer and linked citations are cached when caching is
enabled. An out-of-range citation is reported as an error rather than linked
to the wrong source. Citation numbers do not independently verify factual support.

For a local `MILVUS_URI`, the parent directory and a new Milvus Lite database
are created on first use if absent. This creates an empty index; if an existing
database is deleted, it does not restore previously indexed documents from
Postgres or uploaded files. Local index creation avoids Milvus Lite's unsupported
`AllocTimestamp` RPC, and idle gRPC keepalive pings are disabled for the local
connection. Remote Milvus uses its normal index and connection behavior.

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
