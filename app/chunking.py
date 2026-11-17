"""Layout-aware chunking — the 'layout-aware' half of the RAG pipeline.

Text is split by document section, not by fixed character windows, so a
retrieved chunk is a coherent unit (the totals block, a line item, the tax
block) and citations point at a meaningful section + page.

Sections are detected from headings / label lines; lines that match no
heading inherit the previous section. Chunks are capped at ~600 chars
with a 60-char overlap so a value split across a boundary is still found.
"""
from __future__ import annotations

import re

SECTION_PATTERNS = [
    ("header", re.compile(r"^(invoice|tax invoice|commercial invoice|proforma)", re.I)),
    # "Sub Total" / "Total(RM)" are written with a space or glued to a
    # bracket on real receipts, so match sub-total loosely and `total` alone.
    # Totals is checked before tax because tax lines normally sit *inside*
    # the totals block ("Total GST(RM): 18.81"); a dedicated tax block
    # (GSTSUMMARY, TaxID/TIN) still gets its own section below.
    ("totals", re.compile(r"(sub\s*-?\s*total|grand\s+total|total\s+due|amount\s+due"
                          r"|balance\s+due|\btotal\b|tot\s*qty|totqty)", re.I)),
    ("tax", re.compile(r"(\bVAT|value added tax|withholding tax|\bTIN\b|tax id|\bGST)", re.I)),
    ("line_items", re.compile(r"(description|\bitem\b|\bqty\b|quantity|unit price|\bamount\b)", re.I)),
    ("bank", re.compile(r"(bank|account|sort code|beneficiary|payment details)", re.I)),
    ("terms", re.compile(r"(terms|payment terms|due date|notes|remarks)", re.I)),
]

MAX_CHUNK_CHARS = 600
OVERLAP_CHARS = 60


def _section_for(line: str, current: str) -> str:
    for name, rx in SECTION_PATTERNS:
        if rx.search(line):
            return name
    return current


def chunk_text(text: str, page: int = 1) -> list[dict]:
    """Split OCR text into layout-aware chunks.

    Returns [{chunk_id, section, page, text}]. chunk_id is stable for a
    given invoice id + section + index (the caller prefixes the invoice id).
    """
    if not text or not text.strip():
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Group consecutive lines into the same section.
    groups: list[dict] = []
    current = "header"
    buf: list[str] = []
    for ln in lines:
        sec = _section_for(ln, current)
        if sec != current and buf:
            groups.append({"section": current, "page": page, "text": "\n".join(buf)})
            buf = []
            current = sec
        buf.append(ln)
    if buf:
        groups.append({"section": current, "page": page, "text": "\n".join(buf)})

    # Split over-long groups with overlap; keep short groups whole.
    chunks: list[dict] = []
    for g in groups:
        body = g["text"]
        if len(body) <= MAX_CHUNK_CHARS:
            chunks.append({**g, "chunk_id": ""})
        else:
            start = 0
            idx = 0
            while start < len(body):
                piece = body[start:start + MAX_CHUNK_CHARS]
                chunks.append({"section": g["section"], "page": page, "text": piece, "chunk_id": ""})
                idx += 1
                start += MAX_CHUNK_CHARS - OVERLAP_CHARS
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{c['section']}-{i}"
    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Chunk a multi-page document; pages = [{page, text}]."""
    out: list[dict] = []
    for p in pages:
        out.extend(chunk_text(p.get("text", ""), page=p.get("page", 1)))
    # Re-number chunk ids across the whole document.
    for i, c in enumerate(out):
        c["chunk_id"] = f"{c['section']}-{i}"
    return out
