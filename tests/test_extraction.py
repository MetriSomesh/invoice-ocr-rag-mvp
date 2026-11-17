"""Pure-logic tests — no server, no API key, no Tesseract needed.
Run: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import chunk_text
from app.compliance import run_checks
from app.extract import extract_fields

CLEAN_TEXT = """
West Africa Logistics Ltd
TAX INVOICE
Invoice No: INV-9912
Date: 12/11/2026
Due Date: 12/12/2026
PO No: PO-4471
Tax ID (TIN): 12345678-0001

Description          Qty   Unit Price   Amount
Steel pipes          10    100,000      1,000,000
Valves               5     50,000       250,000

Subtotal: 1,250,000
VAT (7.5%): 93,750
Total: 1,343,750
Bank: First Bank Nigeria, Acct 2034567890
"""


def test_extract_clean_text():
    f = extract_fields(CLEAN_TEXT)
    assert f["vendor"] == "West Africa Logistics Ltd"
    assert f["invoice_number"] == "INV-9912"
    assert f["po_number"] == "PO-4471"
    assert f["tin"] == "12345678-0001"
    assert f["subtotal"] == 1_250_000
    assert f["vat_amount"] == 93_750
    assert f["total"] == 1_343_750
    assert len(f["line_items"]) == 2
    assert f["missing_fields"] == []


def test_extract_missing_fields_are_none_not_invented():
    f = extract_fields("Some random text with no invoice data at all.")
    assert f["invoice_number"] is None
    assert f["total"] is None
    assert "invoice_number" in f["missing_fields"]
    assert "total" in f["missing_fields"]


def test_extract_tin_colon_format():
    f = extract_fields("Vendor Ltd\nTIN: 98765432-0001\nTotal: 5000")
    assert f["tin"] == "98765432-0001"


def test_chunking_sections():
    chunks = chunk_text(CLEAN_TEXT)
    sections = {c["section"] for c in chunks}
    assert "totals" in sections
    assert "tax" in sections or "totals" in sections
    assert all(c["text"].strip() for c in chunks)
    assert all(c["chunk_id"] for c in chunks)


def test_compliance_pass_on_clean():
    f = extract_fields(CLEAN_TEXT)
    r = run_checks(f, expected_po_total=1_343_750, existing_invoice_numbers=[])
    assert r["verdict"] == "APPROVED"
    assert r["risk_score"] == 0
    assert r["amount_at_risk"] == 0
    by_name = {c["name"]: c["status"] for c in r["checks"]}
    assert by_name["maths_subtotal_vat_equals_total"] == "PASS"
    assert by_name["tin_vat_present"] == "PASS"


def test_compliance_po_variance_hold():
    f = extract_fields(CLEAN_TEXT)
    r = run_checks(f, expected_po_total=1_000_000, existing_invoice_numbers=[])
    assert r["verdict"] == "HOLD"
    assert r["risk_score"] >= 20
    assert r["amount_at_risk"] == 343_750


def test_compliance_missing_tin_hold():
    f = extract_fields(CLEAN_TEXT.replace("Tax ID (TIN): 12345678-0001", ""))
    r = run_checks(f, expected_po_total=1_343_750, existing_invoice_numbers=[])
    assert r["verdict"] == "HOLD"
    by_name = {c["name"]: c["status"] for c in r["checks"]}
    assert by_name["tin_vat_present"] == "FAIL"


def test_compliance_duplicate():
    f = extract_fields(CLEAN_TEXT)
    r = run_checks(f, expected_po_total=1_343_750, existing_invoice_numbers=["INV-9912"])
    by_name = {c["name"]: c["status"] for c in r["checks"]}
    assert by_name["duplicate_invoice"] == "FAIL"


# --- SROIE real receipt (Hugging Face jsdnrs/ICDAR2019-SROIE, key
# X51005433533) — ground truth from the dataset. This caught a real bug:
# the old total fallback picked a 13-digit barcode and returned
# ~9.9 trillion as the total, the invoice number was "Not found"
# ("RECEIPT #:" was not recognised), GST NO was not mapped to the tax
# id, and the currency was shown as ₦. OCR-like text as Tesseract reads it.
SROIE_TEXT = """
PERNIAGAAN ZHENG HUI
JM0325955-V
NO.59 JALAN PERMAS 9/5
BANDAR BARU PERMAS JAYA
81750 JOHOR BAHRU
TEL : 07-386 7524 FAX : 07-386 3793
GST NO : 000800589824
SIMPLIFIED TAX INVOICE
GOGIANT ENGINEERING (M) SDN BHD
RECEIPT #: CS00084670
SALESPERSON:
DATE: 09/03/2018
CASHIER: USER
TIME: 08:49:00
ITEM QTY RSP AMOUNT
10273 1 170.00 170.00
SR: 14" TRU-CUT CUTTING DISC (1 PC)
2547 1 35.00 35.00
SR: KOBELCO G-12#(2.6MM) WELDING
12133 2 7.50 15.00
SR: CS 3" (750) PAINT BRUSH
9555752615725 1 25.00 25.00
SR: 2" LAMBWOOL YANDA ROLLER REFIL - 20 PCS
2226 1 6.00 6.00
SR: DIAMOND COTTON GLOVE 8104
3679 5 1.00 5.00
SR: 10# WELDING GLASS (BLACK) 1PCS
3680 10 0.30 3.00
SR: WELDING GLASS (CLEAR) 1 PC
9555058800696 2 8.00 16.00
SR: AEC BROOM NO.5110# C/W IRON HANDLE
6891 2 12.00 24.00
SR: S/STEEL DUST PANL C/W HANDLE 23 X 28 X
TOT QTY: 28 313.50
(Excluded GST) Sub Total (RM) : 313.50
Discount (RM) : 0.00
Total GST (RM) : 18.81
Rounding (RM) : -0.01
TOTAL (RM) : 332.30
CASH : 332.30
Change (RM) : 0.00
GST SUMMARY
Tax Code % Amount GST
SR 6 313.50 18.81
Total : 313.50 18.81
GOODS SOLD ARE NOT RETURNABLE.
THANK YOU.
"""


def test_extract_sroie_receipt():
    f = extract_fields(SROIE_TEXT)
    assert "PERNIAGAAN ZHENG HUI" in (f["vendor"] or "")
    assert f["invoice_number"] == "CS00084670"
    assert f["invoice_date"] == "09/03/2018"
    assert f["tin"] == "000800589824"
    assert f["currency"] == "MYR"
    assert f["subtotal"] == 313.50
    assert f["vat_amount"] == 18.81
    assert f["total"] == 332.30
    assert "invoice_number" not in f["missing_fields"]
    assert "tin" not in f["missing_fields"]


def test_extract_sroie_maths_passes():
    f = extract_fields(SROIE_TEXT)
    r = run_checks(f, expected_po_total=None, existing_invoice_numbers=[])
    by_name = {c["name"]: c["status"] for c in r["checks"]}
    # 313.50 + 18.81 = 332.31 vs total 332.30 (rounding -0.01) — within ₦1.
    assert by_name["maths_subtotal_vat_equals_total"] == "PASS"
    assert by_name["tin_vat_present"] == "PASS"


def test_total_fallback_ignores_barcodes():
    # No "Total" label at all: the fallback must pick the decimal money
    # amount, never a 13-digit barcode or a tax number.
    f = extract_fields(
        "Shop Ltd\nBarcode 9555752615725\nGST NO : 000800589824\n"
        "Item 2 10.00 20.00\nGrand 45.60\n"
    )
    assert f["total"] == 45.60


def test_receipt_number_is_invoice_number():
    f = extract_fields("Some Shop\nRECEIPT #: CS00084670\nTotal: 100.00")
    assert f["invoice_number"] == "CS00084670"


def test_gst_no_maps_to_tin():
    f = extract_fields("Some Shop\nGST NO : 000800589824\nTotal: 100.00")
    assert f["tin"] == "000800589824"


def test_receipt_hash_no_space():
    # OCR reads "Receipt#: CS00084670" with no space before '#'.
    f = extract_fields("Some Shop\nReceipt#: CS00084670\nTotal: 100.00")
    assert f["invoice_number"] == "CS00084670"


def test_date_misread_as_data():
    # Tesseract reads "Date:" as "Data:" on noisy scans.
    f = extract_fields("Some Shop\nSalesperson: Data: 09/03/2018\nTotal: 100.00")
    assert f["invoice_date"] == "09/03/2018"


def test_subtotal_mangled_bracket():
    # OCR reads "(RM)" as "(RM}" — the amount must still be found.
    f = extract_fields("Some Shop\nSub Total(RM} 313.50\nTotal (RM) : 332.30")
    assert f["subtotal"] == 313.50
    assert f["total"] == 332.30


def test_vendor_skips_handwritten_scribble():
    # A handwritten annotation at the top of the scan ("A03041" read as
    # "Aio3e q H") must not become the vendor.
    f = extract_fields(". A .\nAio3e q H .\nPERNIAGAAN ZHENG HU!\nGST NO : 000800589824\nTotal: 100.00")
    assert "PERNIAGAAN ZHENG HU" in (f["vendor"] or "")


def test_receipt_number_with_junk_before_value():
    # RapidOCR turned the ':' into '???' on a receipt: "Receipt#???CS00084670".
    f = extract_fields("Some Shop\nReceipt#???CS00084670\nTotal: 100.00")
    assert f["invoice_number"] == "CS00084670"


def test_money_column_multi_letter_misread():
    # Trailing zeros are often read as C: "35.CC" is 35.00, "170.0C" is 170.00.
    assert extract_fields("Some Shop\nTotal {RM) : 35.CC\n")["total"] == 35.0
    assert extract_fields("Some Shop\nTotal (RM) : 170.0C")["total"] == 170.0


def test_table_stop_tolerates_missing_spaces():
    # PP-OCR merges words: "TOTQTY28" must still end the item table.
    page = _word_page([
        [("Item", 40), ("1", 180), ("10.00", 240), ("10.00", 320)],
        [("TOTQTY28", 40), ("10.00", 240)],
        [("SR", 40), ("6", 70), ("10.00", 120), ("0.60", 200)],
    ])
    f = extract_fields("", word_pages=[page])
    assert len(f["line_items"]) == 1


def test_amount_with_trailing_ocr_dots():    # Live OCR reads "Total GST (RM) : 18.84 ." — the amount pattern
    # swallows the trailing dots and float() used to reject the value,
    # so VAT came back None even though the label matched.
    f = extract_fields("Some Shop\nTotal GST (RM) : 18.84 .\nTotal (RM) : 332.30")
    assert f["vat_amount"] == 18.84
    assert f["total"] == 332.30


def test_gst_summary_rows_are_not_line_items():
    f = extract_fields(
        "Some Shop\nItem 1 10.00 10.00\nGST SUMMARY\nSR 6 313.50 18.81\nTotal: 100.00"
    )
    assert len(f["line_items"]) == 1
    assert f["line_items"][0]["amount"] == 10.00


# --- Regression: the failures seen on a real 200 DPI phone scan -----------

def test_total_with_brace_bracket_is_not_guessed():
    # Live OCR reads "Total {RM) : 332.30" (brace opener, paren closer).
    # The label used to miss, so the total fell through to a "largest
    # number" guess and came back as 276015 instead of 332.30.
    f = extract_fields("Some Shop\nTotal {RM) : 332.30\nCASH : 332.30")
    assert f["total"] == 332.30


def test_amount_never_spans_a_newline():
    # The amount pattern used \s, which includes newlines, so digits from
    # two different lines were concatenated into one huge "amount".
    f = extract_fields("Some Shop\nTotal {RM) : 27\n6015\n")
    assert f["total"] != 276015.0


def test_unlabeled_total_can_stay_none():
    # A barcode and a tax number are not totals. With no total-shaped
    # money token the field must stay None (compliance then WARNs).
    f = extract_fields("Shop Ltd\nBarcode 9555752615725\nGST NO : 000800589824")
    assert f["total"] is None
    assert "total" in f["missing_fields"]


def _word_page(rows):
    """Build Tesseract-shaped word boxes from (text, x) rows of one page."""
    words = []
    for line_no, row in enumerate(rows):
        for text, left in row:
            words.append({
                "text": text, "left": left, "top": line_no * 20,
                "width": max(8, 9 * len(text)), "height": 14, "conf": 88.0,
            })
    return words


def test_line_items_parsed_by_column_position():
    # Real receipt layout: code | qty | unit | amount on one line, the
    # description on the NEXT line. A line-order regex can only read the
    # odd row whose description shares the line; the word boxes carry the
    # whole table, so all three rows must parse.
    page = _word_page([
        [("RECEIPT", 40), ("#:", 140), ("CS00084670", 200)],
        [("ITEM", 40), ("QTY", 160), ("RSP", 240), ("AMOUNT", 320)],
        [("10273", 40), ("GA", 100), ("1", 180), ("170.00", 240), ("170.0C", 320)],
        [("SR:", 40), ('14"', 70), ("TRU-CUT", 100), ("CUTTING", 160), ("DISC", 220)],
        [("2547", 40), ("Goi", 100), ("1", 180), ("35.00", 240), ("36.00", 320)],
        [("SR:", 40), ("KOBELCO", 70), ("G-12#", 140), ("WELDING", 200)],
        [("12133", 40), ("GAts", 100), ("2", 180), ("7.60", 240), ("15.00", 320)],
        [("SR:", 40), ("CS", 70), ("3", 100), ("PAINT", 130), ("BRUSH", 190)],
        [("TOT", 40), ("QTY:", 80), ("28", 160), ("313.50", 240)],
        [("(Excluded", 40), ("GST)", 110), ("Sub", 170), ("Total", 200), ("(RM)", 250), (":", 300), ("313.50", 330)],
        [("Total", 40), ("GST", 90), ("(RM)", 130), (":", 170), ("18.84", 200)],
        [("Total", 40), ("{RM)", 90), (":", 130), ("332.30", 160)],
        [("GST", 40), ("SUMMARY", 90)],
        [("SR", 40), ("6", 70), ("313.50", 120), ("18.84", 200)],
    ])
    f = extract_fields("", word_pages=[page])
    assert len(f["line_items"]) == 3
    assert [i["amount"] for i in f["line_items"]] == [170.0, 36.0, 15.0]
    assert [i["qty"] for i in f["line_items"]] == [1.0, 1.0, 2.0]
    # The description printed under the row is attached to that row.
    assert "CUTTING" in f["line_items"][0]["description"]
    assert "KOBELCO" in f["line_items"][1]["description"]


def test_word_boxes_ignore_totals_and_barcodes():    # A 13-digit item code and the totals block must not become line items.
    page = _word_page([
        [("9555752615725", 40), ("1", 180), ("25.00", 240), ("25.00", 320)],
        [("Sub", 40), ("Total", 90), ("(RM)", 130), (":", 170), ("313.50", 200)],
        [("Cash", 40), (":", 90), ("332.30", 130)],
    ])
    f = extract_fields("", word_pages=[page])
    assert len(f["line_items"]) == 1
    assert f["line_items"][0]["amount"] == 25.0


def test_lost_decimal_row_is_excluded_from_sum_not_failed():
    # Live OCR read "24.00" as "2400" on one row. Keeping it in the sum
    # turned a clean receipt into a FAIL with a 2,400 phantom difference.
    # The row stays in the item list (as read) but is excluded and named.
    page = _word_page([
        [("12131", 40), ("1", 180), ("3.60", 240), ("3.60", 320)],
        [("SR:", 40), ("CS", 70), ("PAINT", 120), ("BRUSH", 180)],
        [("6894", 40), ("2", 180), ("1200", 240), ("2400", 320)],
        [("SR:", 40), ("S/STEEL", 70), ("DUST", 140), ("PAN", 190)],
        [("Total", 40), ("(RM)", 90), (":", 130), ("27.60", 160)],
    ])
    f = extract_fields("(Excluded GST) Sub Total (RM) : 3.60\nTotal (RM) : 27.60",
                       word_pages=[page])
    assert len(f["line_items"]) == 2
    assert f["line_items"][1]["amount"] == 2400.0
    assert f["line_items"][1]["amount_uncertain"] is True
    assert f["subtotal"] == 3.60
    # The check must not FAIL because of the excluded row.
    c = run_checks(f, expected_po_total=None, existing_invoice_numbers=[])
    by_name = {x["name"]: x for x in c["checks"]}
    assert by_name["line_item_sum_matches_subtotal"]["status"] == "WARN"
    assert "excluded" in by_name["line_item_sum_matches_subtotal"]["detail"]
    assert "2,400.00" in by_name["line_item_sum_matches_subtotal"]["detail"]


def test_whole_number_amounts_are_not_flagged_uncertain():
    # A naira invoice with no cents ("1,000,000") is precisely written —
    # the uncertainty flag must be relative to how the document formats
    # its own amounts, not a hardcoded "must have cents" rule.
    page = _word_page([
        [("Steel", 40), ("pipes", 90), ("10", 200), ("100,000", 250), ("1,000,000", 330)],
        [("Valves", 40), ("5", 200), ("50,000", 250), ("250,000", 330)],
    ])
    f = extract_fields("Subtotal: 1,250,000\nTotal: 1,343,750", word_pages=[page])
    assert [i["amount"] for i in f["line_items"]] == [1_000_000.0, 250_000.0]
    assert all(i["amount_uncertain"] is False for i in f["line_items"])
    c = run_checks(f, expected_po_total=None, existing_invoice_numbers=[])
    by_name = {x["name"]: x for x in c["checks"]}
    assert by_name["line_item_sum_matches_subtotal"]["status"] == "PASS"



