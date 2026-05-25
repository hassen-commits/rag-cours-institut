"""Ingest PDFs into the Chroma vector store.

Scans a source directory for PDFs, extracts text page-by-page, splits into
overlapping chunks, embeds with OpenAI, and writes to a persistent Chroma
collection. Re-running is idempotent: chunk IDs are deterministic, so
duplicates are upserted rather than appended.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from ocr import ocr_pdf
from rag import get_collection

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))


def find_pdfs(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() == ".pdf":
        return [source]
    skip_dirs = {"chroma_db", ".git", ".venv", "venv", "__pycache__", "node_modules"}
    pdfs: list[Path] = []
    for path in source.rglob("*.pdf"):
        if any(part in skip_dirs for part in path.parts):
            continue
        pdfs.append(path)
    return sorted(pdfs)


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) tuples for each non-empty page (1-indexed)."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"  ! page {i}: extraction failed ({e})", file=sys.stderr)
            text = ""
        text = text.strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[tuple[int, str]]:
    """Split each page into chunks, preserving the originating page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "۔", ".", "؟", "?", "!", " ", ""],
    )
    chunks: list[tuple[int, str]] = []
    for page_num, text in pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if piece:
                chunks.append((page_num, piece))
    return chunks


def chunk_id(source: str, page: int, text: str) -> str:
    h = hashlib.sha1(f"{source}|{page}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"{Path(source).stem}-p{page}-{h}"


def _book_id(pdf: Path, override: str | None) -> str:
    if override:
        return override
    return pdf.stem


def ingest(
    source_dir: Path,
    force_ocr: bool = False,
    no_ocr: bool = False,
    book: str | None = None,
) -> None:
    pdfs = find_pdfs(source_dir)
    if not pdfs:
        print(f"No PDFs found under {source_dir}", file=sys.stderr)
        sys.exit(1)

    collection = get_collection()
    print(f"Collection: {collection.name} (existing items: {collection.count()})")

    total_chunks = 0
    for pdf in pdfs:
        rel = str(pdf.resolve())
        print(f"\n→ {pdf}")

        if force_ocr:
            print("  --ocr: skipping pypdf extraction, using OCR directly.")
            pages = ocr_pdf(pdf)
        else:
            pages = extract_pages(pdf)
            if not pages and not no_ocr:
                print("  ! no extractable text — falling back to OCR.")
                pages = ocr_pdf(pdf)
            elif not pages:
                print("  ! no extractable text and --no-ocr set; skipping.")
                continue

        if not pages:
            print("  ! still no text after OCR; skipping.")
            continue
        chunks = chunk_pages(pages)
        if not chunks:
            print("  ! no chunks produced.")
            continue

        book_id = _book_id(pdf, book)
        seen_ids: set[str] = set()
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        for page_num, chunk_text in chunks:
            cid = chunk_id(rel, page_num, chunk_text)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            ids.append(cid)
            docs.append(chunk_text)
            metas.append({"source": rel, "page": page_num, "book": book_id})

        batch = 100
        for i in range(0, len(ids), batch):
            collection.upsert(
                ids=ids[i : i + batch],
                documents=docs[i : i + batch],
                metadatas=metas[i : i + batch],
            )
        print(f"  ✓ {len(pages)} pages → {len(ids)} chunks indexed")
        total_chunks += len(ids)

    print(
        f"\nDone. Indexed {total_chunks} chunks from {len(pdfs)} PDF(s). "
        f"Collection now has {collection.count()} items."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Chroma.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("."),
        help="Directory or PDF file to ingest (default: project root).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ocr",
        action="store_true",
        help="Force OCR even when pypdf extracts text (useful for garbled output).",
    )
    group.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback. Pages with no extractable text are skipped.",
    )
    parser.add_argument(
        "--book",
        type=str,
        default=None,
        help="Override book id stored in chunk metadata. Defaults to the PDF filename stem.",
    )
    args = parser.parse_args()
    ingest(args.source, force_ocr=args.ocr, no_ocr=args.no_ocr, book=args.book)


if __name__ == "__main__":
    main()
