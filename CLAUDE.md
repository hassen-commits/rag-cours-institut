# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A small RAG (Retrieval-Augmented Generation) pipeline over Arabic institute
course PDFs. Pipeline: PDF → page-aware text extraction (with OpenAI-vision
OCR fallback for scanned PDFs) → recursive char chunking → OpenAI embeddings
→ ChromaDB (persistent) → top-K retrieval → OpenAI chat completion with cited
context. UI is a Streamlit chat app.

The seed PDF in this repo (`رسالة المسترشدين.pdf`) is **image-only** (151
pages, JBIG2-encoded scans, no text layer). That is the reason the OCR path
exists — it isn't theoretical.

## Common commands

```powershell
# one-time setup
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then fill in OPENAI_API_KEY

# ingest (default: scan project root for *.pdf recursively, skipping chroma_db/.venv/.git)
python ingest.py
python ingest.py --source .\some\folder
python ingest.py --source single.pdf
python ingest.py --ocr       # force OCR even when pypdf gets text (use when text is garbled)
python ingest.py --no-ocr    # disable OCR fallback; skip pages with no extractable text

# run UI
streamlit run app.py

# wipe the index (no CLI flag for this — delete the folder)
Remove-Item -Recurse -Force .\chroma_db

# wipe the OCR cache (forces re-OCR on next ingest — expensive)
Remove-Item -Recurse -Force .\.ocr_cache
```

There is no test suite, linter, or build step configured. Don't invent one
unless asked.

## Architecture

Four Python modules, ~one responsibility each. Read them in this order to
get the big picture:

1. **[rag.py](rag.py)** — single source of truth for index access and
   generation. Both `ingest.py` and `app.py` go through `get_collection()`,
   which means: embedding model, collection name, persistence path, and
   distance metric (`cosine`) are configured in **one** place. If you change
   `EMBEDDING_MODEL`, the existing index becomes incompatible — re-ingest.
2. **[ocr.py](ocr.py)** — page-image → text via OpenAI vision
   (`OCR_MODEL`, defaults to `gpt-4o`). Renders pages with `pypdfium2`
   (no Poppler/Tesseract install needed on Windows). Caches per-page
   transcriptions to `.ocr_cache/<stem>_<hash>_pNNNN.txt`. The cache is
   the resume mechanism — if a 150-page OCR run dies at page 80, the next
   run picks up at page 81 for free. Deleting the cache forces re-OCR
   and re-incurs cost.
3. **[ingest.py](ingest.py)** — discovery → extraction → chunking → upsert.
   Key behaviors that aren't obvious:
   - Three extraction modes: default (pypdf, fall back to OCR if empty),
     `--ocr` (force OCR), `--no-ocr` (skip pages with no text). The default
     is what you almost always want.
   - Chunk IDs are SHA1 of `source|page|text` (truncated). This makes ingest
     **idempotent** — re-running upserts identical chunks in place rather
     than duplicating. If you change `CHUNK_SIZE`/`CHUNK_OVERLAP` the IDs
     shift and old chunks linger; wipe `chroma_db/` to get a clean slate.
   - The splitter's separator list is Arabic-aware (`۔`, `؟` included before
     Latin punctuation) so sentence boundaries survive in Arabic text.
   - Per-page extraction (whether from pypdf or OCR) is preserved through
     chunking so retrieved passages carry a real `page` number for citation.
   - Reconfigures stdout/stderr to UTF-8 at startup because Windows
     cp1252 chokes on Arabic filenames and Unicode arrows in log output.
4. **[app.py](app.py)** — Streamlit UI. `answer()` from `rag.py` does the
   actual work; the file is mostly chat-state plumbing and an RTL CSS
   override. The sidebar's "Indexed chunks" metric calls `collection.count()`
   on every rerun, which is fine for a small local index but would need
   caching if the collection grew large.

## Conventions worth knowing

- **Env over flags.** Tunables (`CHUNK_SIZE`, `TOP_K`, model names, etc.)
  live in `.env`, not CLI args. `ingest.py` only takes `--source` because
  the input path genuinely varies per run; everything else should stay in
  `.env` so ingest- and query-time stay in sync.
- **Cite passages.** The system prompt instructs the model to cite as
  `[filename:pN]`. If you change the prompt, keep the citation contract —
  the UI's "Sources" expander relies on the same `source`/`page` metadata
  the model is told to cite.
- **Arabic / RTL.** The system prompt tells the model to reply in the
  user's language and preserve Arabic quotations verbatim. `app.py` applies
  RTL styling globally — if you add LTR UI elements, scope the CSS.
- **PDF extraction is best-effort + OCR fallback.** `pypdf` handles text
  PDFs; image-only PDFs (like the seed file) fall through to `ocr.py`
  automatically. If `pypdf` returns text but it's *garbled* (RTL layout
  edge cases), pass `--ocr` to force the OCR path. OCR costs money per
  uncached page; the `.ocr_cache/` directory amortizes that across reruns.

## Things that will trip you up

- Changing `EMBEDDING_MODEL` without re-ingesting → silent quality collapse.
  Query embeddings won't match indexed ones. Wipe `chroma_db/` after the
  change.
- `OPENAI_API_KEY` is read at `get_collection()` time, not at import time.
  The Streamlit app will start fine without it and only fail on the first
  question. `rag._require_api_key()` raises a clear error in that path.
- The collection uses cosine distance (`hnsw:space: cosine`), set on
  creation. Chroma ignores `metadata={}` changes on existing collections —
  to switch distance metrics, delete and re-create.
- The OCR cache key is a hash of the PDF's **absolute path**, not its
  content. Moving the PDF invalidates its cache. If you need to OCR the
  same file from two locations, copy the cache files or symlink the PDF.
