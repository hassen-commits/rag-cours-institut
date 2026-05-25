"""OCR fallback for image-only PDFs, using OpenAI vision (gpt-4o by default).

Renders each PDF page with pypdfium2, sends it to a vision-capable chat model,
and caches the transcribed text per page in `.ocr_cache/`. Re-runs are
resumable — only uncached pages incur API cost.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from openai import OpenAI

OCR_MODEL = os.environ.get("OCR_MODEL", "gpt-4o")
OCR_CACHE_DIR = Path(os.environ.get("OCR_CACHE_DIR", ".ocr_cache"))
RENDER_SCALE = float(os.environ.get("OCR_RENDER_SCALE", "2.0"))
OCR_MAX_TOKENS = int(os.environ.get("OCR_MAX_TOKENS", "2500"))
OCR_MAX_CHARS = int(os.environ.get("OCR_MAX_CHARS", "8000"))
OCR_DOMINANT_CHAR_THRESHOLD = float(os.environ.get("OCR_DOMINANT_CHAR_THRESHOLD", "0.4"))

UNREADABLE_SENTINEL = "__OCR_UNREADABLE__"

OCR_PROMPT = (
    "This is a scanned page from an Arabic book. Transcribe ALL visible Arabic text "
    "exactly as it appears, preserving paragraph breaks and the order of lines. "
    "Do not translate, summarize, paraphrase, or add any commentary. "
    "Do not include the page number, headers, or footers unless they are part of the "
    "body content. If a section is illegible, write [غير واضح] in its place. "
    "Output only the transcribed text — no preamble, no explanation."
)


def _cache_key(pdf_path: Path) -> str:
    """Stable per-PDF identifier so two PDFs with the same stem don't collide."""
    h = hashlib.sha1(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{pdf_path.stem}_{h}"


def _render_page_png(pdf: "pdfium.PdfDocument", index: int) -> bytes:
    page = pdf[index]
    pil_image = page.render(scale=RENDER_SCALE).to_pil()
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _transcribe(client: OpenAI, png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    resp = client.chat.completions.create(
        model=OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=OCR_MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""


def _is_degenerate(text: str) -> bool:
    """Detect model output that looped (a single char repeated, or absurdly long).

    A normal Arabic book page is ~500-3000 chars. Anything past OCR_MAX_CHARS,
    or where one non-space character dominates the output, is the model stuck
    in a repetition loop and should be discarded.
    """
    stripped = "".join(c for c in text if not c.isspace())
    if not stripped:
        return False
    if len(stripped) > OCR_MAX_CHARS:
        return True
    most_common_count = Counter(stripped).most_common(1)[0][1]
    return (most_common_count / len(stripped)) > OCR_DOMINANT_CHAR_THRESHOLD


def ocr_pdf(pdf_path: Path, max_retries: int = 2) -> list[tuple[int, str]]:
    """OCR every page of `pdf_path` and return (page_number, text) for non-empty pages.

    Results are cached in `.ocr_cache/<key>_pNNNN.txt`. Cached pages skip the API call.
    """
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(pdf_path)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for OCR. Set it in .env."
        )
    client = OpenAI(api_key=api_key)

    pdf = pdfium.PdfDocument(str(pdf_path))
    total = len(pdf)
    print(f"  OCR via {OCR_MODEL}: {total} pages")

    pages: list[tuple[int, str]] = []
    for i in range(total):
        page_num = i + 1
        cache_file = OCR_CACHE_DIR / f"{key}_p{page_num:04d}.txt"

        cached_text = (
            cache_file.read_text(encoding="utf-8") if cache_file.exists() else None
        )
        if cached_text == UNREADABLE_SENTINEL:
            print(
                f"  [{page_num}/{total}] previously marked unreadable; skipping.",
                file=sys.stderr,
            )
            continue
        if cached_text is not None and not _is_degenerate(cached_text):
            text = cached_text
            status = "cached"
        else:
            if cached_text is not None:
                print(
                    f"  [{page_num}/{total}] cached output looks degenerate "
                    f"({len(cached_text)} chars); re-OCR'ing.",
                    file=sys.stderr,
                )
                cache_file.unlink()

            png = _render_page_png(pdf, i)
            text = ""
            api_error = False
            degenerate_failure = False
            for attempt in range(1, max_retries + 1):
                try:
                    candidate = _transcribe(client, png)
                except Exception as e:
                    msg = str(e)
                    if attempt == max_retries:
                        print(
                            f"  [{page_num}/{total}] OCR failed after {attempt} tries: {msg}",
                            file=sys.stderr,
                        )
                        api_error = True
                        break
                    backoff = 2 ** attempt
                    print(
                        f"  [{page_num}/{total}] attempt {attempt} failed ({msg}); "
                        f"retrying in {backoff}s",
                        file=sys.stderr,
                    )
                    time.sleep(backoff)
                    continue

                if _is_degenerate(candidate):
                    print(
                        f"  [{page_num}/{total}] attempt {attempt} produced degenerate "
                        f"output ({len(candidate)} chars); retrying.",
                        file=sys.stderr,
                    )
                    if attempt == max_retries:
                        degenerate_failure = True
                        break
                    time.sleep(1)
                    continue

                text = candidate
                break

            if api_error:
                status = "API error (not cached, will retry next run)"
            elif degenerate_failure:
                cache_file.write_text(UNREADABLE_SENTINEL, encoding="utf-8")
                status = "unreadable (sentinel cached; delete file to retry)"
                text = ""
            else:
                cache_file.write_text(text, encoding="utf-8")
                status = "OCR'd"

        text = text.strip()
        print(f"  [{page_num}/{total}] {status} ({len(text)} chars)")
        if text:
            pages.append((page_num, text))

    return pages
