"""Chunking tests — section detection on real receipt/OCR line shapes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import _section_for, chunk_text


def test_totals_written_with_spaces_or_brackets():
    # A real Malaysian receipt: "Sub Total(RM)" and "Total(RM)" — neither
    # contains the word "subtotal", which the old pattern required.
    assert _section_for("(Excluded GST)Sub Total(RM): 313.50", "line_items") == "totals"
    assert _section_for("Total(RM): 332.30", "line_items") == "totals"
    assert _section_for("Rounding (RM) : -0.01", "totals") == "totals"


def test_tax_lines_stay_in_the_tax_block():
    # A GST amount sits inside the totals block; the dedicated tax header
    # and the TIN still get their own section.
    assert _section_for("Total GST(RM): 18.81", "totals") == "totals"
    assert _section_for("GSTSUMMARY", "totals") == "tax"
    assert _section_for("TaxID(TIN):12345678-0001", "header") == "tax"


def test_table_header_and_rows_are_line_items():
    assert _section_for("Item Qty Price Amount", "header") == "line_items"
    assert _section_for("Description", "header") == "line_items"


def test_chunks_cite_meaningful_sections():
    text = (
        "SIMPLIFIED TAX INVOICE\n"
        "Receipt#CS00084670\n"
        "Item Qty Price Amount\n"
        "12133 GA18012 7.50 15.00\n"
        "Sub Total(RM): 313.50\n"
        "Total GST(RM): 18.81\n"
        "Total(RM): 332.30\n"
        "GSTSUMMARY\n"
        "SR 313.50 18.81"
    )
    sections = [c["section"] for c in chunk_text(text)]
    assert "totals" in sections and "tax" in sections
    totals = next(c for c in chunk_text(text) if c["section"] == "totals")
    assert "Sub Total(RM): 313.50" in totals["text"]
    assert "Total(RM): 332.30" in totals["text"]
    # The table must not be polluted by the totals block.
    items = next(c for c in chunk_text(text) if c["section"] == "line_items")
    assert "332.30" not in items["text"]
