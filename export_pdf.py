"""Export a chat conversation to PDF with proper Arabic shaping & RTL.

Uses fpdf2 + arabic-reshaper + python-bidi. Reads Arial (and Arial Bold) from
the Windows fonts directory; falls back gracefully on missing-font errors so
the user gets a clear message instead of a cryptic traceback.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterable

import arabic_reshaper
from bidi.algorithm import get_display
from fpdf import FPDF

ARABIC_CHAR = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

_HERE = Path(__file__).parent

# Bundled font is checked first so deploys to Linux (Streamlit Cloud) work
# without needing any system fonts. Local Windows/Linux paths follow as
# convenience fallbacks if the bundled font ever goes missing.
FONT_CANDIDATES_REGULAR = [
    _HERE / "assets" / "fonts" / "DejaVuSans.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
]
FONT_CANDIDATES_BOLD = [
    _HERE / "assets" / "fonts" / "DejaVuSans-Bold.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path(r"C:\Windows\Fonts\tahomabd.ttf"),
    Path(r"C:\Windows\Fonts\segoeuib.ttf"),
]


def _find_font(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise RuntimeError(
        "No Arabic-capable TTF font found. Looked for: "
        + ", ".join(str(c) for c in candidates)
    )


def _is_arabic(text: str) -> bool:
    return ARABIC_CHAR.search(text) is not None


def _shape(text: str) -> str:
    """Reshape Arabic letters + apply bidi reordering for display."""
    if not _is_arabic(text):
        return text
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def _shape_lines(text: str) -> list[tuple[str, str]]:
    """Split text into lines and return (shaped_line, align) tuples."""
    out: list[tuple[str, str]] = []
    for line in text.splitlines() or [""]:
        align = "R" if _is_arabic(line) else "L"
        out.append((_shape(line), align))
    return out


class _ChatPDF(FPDF):
    def header(self):
        self.set_font("Body", "B", 11)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "RAG — Cours Institut", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Body", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def generate(messages: Iterable[dict]) -> bytes:
    """Build a PDF byte string from a list of Streamlit-style chat messages.

    Each message is {"role": "user"|"assistant", "content": str, "sources": [...]}.
    """
    regular = _find_font(FONT_CANDIDATES_REGULAR)
    bold = _find_font(FONT_CANDIDATES_BOLD)

    pdf = _ChatPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("Body", "", str(regular))
    pdf.add_font("Body", "B", str(bold))
    pdf.add_page()

    pdf.set_font("Body", "B", 14)
    pdf.multi_cell(0, 9, _shape("Conversation — رسالة المسترشدين"), align="R")
    pdf.set_font("Body", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(
        0, 6, f"Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        sources = msg.get("sources") or []

        pdf.set_font("Body", "B", 12)
        if role == "user":
            pdf.set_fill_color(230, 240, 255)
            label = "Question"
        else:
            pdf.set_fill_color(245, 245, 245)
            label = "Réponse"
        pdf.cell(0, 8, label, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        pdf.set_font("Body", "", 11)
        for shaped, align in _shape_lines(content):
            pdf.multi_cell(0, 6, shaped, align=align)
        pdf.ln(2)

        if sources:
            pdf.set_font("Body", "B", 10)
            pdf.cell(0, 6, "Sources", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Body", "", 9)
            for s in sources:
                src = os.path.basename(getattr(s, "source", "?"))
                page = getattr(s, "page", -1)
                dist = getattr(s, "distance", 0.0)
                pdf.set_font("Body", "B", 9)
                pdf.cell(
                    0, 5, f"• {src} — p.{page}  (dist={dist:.3f})",
                    new_x="LMARGIN", new_y="NEXT",
                )
                pdf.set_font("Body", "", 9)
                excerpt = (getattr(s, "text", "") or "").strip()
                if len(excerpt) > 500:
                    excerpt = excerpt[:500] + "…"
                for shaped, align in _shape_lines(excerpt):
                    pdf.multi_cell(0, 5, shaped, align=align)
                pdf.ln(1)

        pdf.ln(4)

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()
