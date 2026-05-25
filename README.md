# RAG — Cours Institut

Retrieval-Augmented Generation over institute course PDFs (Arabic-friendly).
Stack: **OpenAI** (embeddings + chat) · **ChromaDB** (vector store) · **Streamlit** (UI).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# then edit .env and set OPENAI_API_KEY
```

## Ingest PDFs

Drop PDFs anywhere under the project (the scanner skips `chroma_db/`, `.git/`, `.venv/`).
The provided `رسالة المسترشدين.pdf` is picked up automatically.

```powershell
python ingest.py                 # scan project root recursively
python ingest.py --source .\docs # scan a specific folder
python ingest.py --source file.pdf
python ingest.py --ocr           # force OCR (use when pypdf text is garbled)
python ingest.py --no-ocr        # disable OCR fallback
```

Re-running is idempotent — chunk IDs are content-hashed, so the same chunk
upserts in place instead of duplicating.

### Scanned / image-only PDFs

If `pypdf` can't extract text (image-only PDF), `ingest.py` automatically
falls back to OpenAI vision (`OCR_MODEL`, defaults to `gpt-4o`) and
transcribes each page. Page transcriptions are cached in `.ocr_cache/` —
re-running ingest skips already-OCR'd pages, so cost is paid once.

To force re-OCR, delete `.ocr_cache/` (or just the affected page files).

## Run the app

```powershell
streamlit run app.py
```

Opens a chat UI at <http://localhost:8501>. The sidebar shows the index size and
lets you tune top-K.

## Configuration

All knobs live in `.env` (see `.env.example`):

| Var | Default | What it does |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required. |
| `CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model used to generate answers. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model. Must match between ingest and query — changing it requires re-ingesting. |
| `OCR_MODEL` | `gpt-4o` | Vision model used to transcribe scanned pages. |
| `CHROMA_DIR` | `./chroma_db` | Persistent Chroma directory. |
| `COLLECTION_NAME` | `cours` | Chroma collection name. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Character-based chunking. |
| `TOP_K` | `5` | Default passages retrieved per question. |

## Deploy to Streamlit Community Cloud (free)

The repo is already wired for cloud deployment. Steps:

1. Push the project to a new **GitHub repo** (public or private — Streamlit Cloud
   supports both for personal accounts).
   ```powershell
   git init
   git add .
   git commit -m "Initial RAG app"
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   The repo includes `chroma_db/` (~12 MB) — that's the pre-built vector index
   so the cloud deployment doesn't need to re-OCR or re-embed anything on cold
   start. `.ocr_cache/` is excluded; it's only needed if you re-ingest.

2. Sign in at <https://share.streamlit.io> with your GitHub account, click
   **New app**, point it at this repo, set the main file to `app.py`.

3. In the app's **Settings → Secrets**, paste the contents of
   `.streamlit/secrets.toml.example` (with your real `OPENAI_API_KEY`).
   Streamlit auto-injects these as environment variables visible to `rag.py`.

4. Deploy. Cold start takes ~1–2 minutes (Python deps install); subsequent
   loads are fast.

**One caveat**: `pypdfium2` and OCR are not needed at runtime if you keep
`chroma_db/` committed. If you want users to be able to re-ingest from the
cloud, you'd also need to ship the PDFs and ensure the OCR API quota is
available — possible but not the default flow.

## Notes on Arabic PDFs

`pypdf` handles text PDFs; image-only scans fall through to the OCR path
described above. If `pypdf` returns text but it looks garbled (a known
issue with some RTL layouts), force the OCR path with `python ingest.py --ocr`.

OCR uses OpenAI vision and costs roughly **$0.005–0.01 per page** with
`gpt-4o`. The `.ocr_cache/` directory makes this a one-time cost per PDF.
