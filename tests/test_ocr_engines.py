"""OCR-engine adapter tests — pure logic, no model, no Tesseract, no network.

The RapidOCR adapter is tested with synthetic engine output (the shape
RapidOCR returns: [box, text, score] per detected line), so these tests run
anywhere, including CI without the ONNX models.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ocr import _rapid_to_page, _respace


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_respace_restores_camel_case():
    assert _respace("WestAfricaLogisticsLtd") == "West Africa Logistics Ltd"
    assert _respace("UnitPriceAmount") == "Unit Price Amount"
    assert _respace("PONo:PO-4471") == "PO No:PO-4471"


def test_respace_leaves_serials_alone():
    # Document numbers are letters glued to digits — splitting them would
    # destroy the invoice number the extractor looks for.
    assert _respace("CS00084670") == "CS00084670"
    assert _respace("INV-9912") == "INV-9912"
    assert _respace("12345678-0001") == "12345678-0001"


def test_rapid_output_becomes_page_with_boxes_and_text():
    result = [
        [_box(95, 99, 334, 131), "WestAfricaLogisticsLtd", 0.99],
        [_box(95, 179, 306, 209), "InvoiceNo:INV-9912", 0.98],
        [_box(101, 456, 213, 487), "Steel pipes", 0.96],
        [_box(268, 455, 298, 480), "10", 1.00],
        [_box(318, 454, 399, 478), "100,000", 0.98],
        [_box(428, 448, 531, 477), "1,000,000", 0.95],
    ]
    page = _rapid_to_page(result, 1)
    assert page.method == "camera-scan-ocr"
    assert len(page.words) == 6
    # Confidence is reported on the same 0-100 scale as Tesseract's.
    assert 90 <= page.confidence <= 100
    # Line 1 = header row, line 2 = the item row rebuilt from its columns.
    assert page.text.splitlines()[0] == "West Africa Logistics Ltd"
    assert "Steel pipes 10 100,000 1,000,000" in page.text


def test_rapid_boxes_feed_the_column_item_parser():
    # The table parser must work off RapidOCR boxes exactly as it does off
    # Tesseract words: qty / unit / amount by x position, not line order.
    from app.extract import extract_fields

    result = [
        [_box(101, 456, 213, 487), "Steel pipes", 0.96],
        [_box(268, 455, 298, 480), "10", 1.00],
        [_box(318, 454, 399, 478), "100,000", 0.98],
        [_box(428, 448, 531, 477), "1,000,000", 0.95],
        [_box(101, 499, 172, 524), "Valves", 0.99],
        [_box(257, 497, 275, 521), "5", 1.00],
        [_box(300, 494, 369, 519), "50,000", 0.98],
        [_box(405, 488, 490, 518), "250,000", 0.98],
    ]
    page = _rapid_to_page(result, 1)
    f = extract_fields(page.text, page.confidence, word_pages=[page.words])
    assert [i["qty"] for i in f["line_items"]] == [10.0, 5.0]
    assert [i["amount"] for i in f["line_items"]] == [1_000_000.0, 250_000.0]


def test_empty_engine_output_is_empty_not_invented():
    page = _rapid_to_page([], 1)
    assert page.text == ""
    assert page.words == []
    assert page.confidence is None
