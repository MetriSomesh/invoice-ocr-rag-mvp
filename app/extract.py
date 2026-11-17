"""Field extraction from real OCR text — regex only, no invention.

Every field is Optional: if a pattern does not match, the field is None
and is listed in `missing_fields`. The caller (compliance) treats missing
fields as WARN/FAIL, never fills them with a guess or a sample value.
Tolerant of common OCR confusions in amounts: 'O'/'o' for 0, 'l' for 1,
extra spaces inside numbers, and ',' vs '.' thousand separators.
"""
from __future__ import annotations

import re
from typing import Any


def _clean_amount(raw: str) -> float | None:
    """'1,250,000.00' / '₦ 1 250 000' / '1.250.000,00' -> float. None on failure."""
    if not raw:
        return None
    s = raw.strip().replace("₦", "").replace("NGN", "").replace("RM", "").replace("MYR", "").replace("$", "").strip()
    s = s.replace("O", "0").replace("o", "0").replace("l", "1")
    # A trailing letter where a final '0' should be is a common OCR misread
    # in money columns: "170.0C" -> 170.00, "35.CC" -> 35.00.
    s = re.sub(r"(?<=[\d.])[CQDG]{1,3}$",
               lambda m: "0" * len(m.group(0)), s)
    s = re.sub(r"(?<=\d)[CQDG]$", "0", s)
    # The amount pattern allows spaces inside numbers ("1 250 000"), so it
    # also swallows trailing OCR noise ("18.84 .\n. " -> "18.84.."), which
    # float() then rejects. Remove ALL whitespace and strip trailing
    # dots/commas left by that noise.
    s = re.sub(r"\s+", "", s).rstrip(".,")
    # European style 1.250.000,00 -> 1250000.00
    if re.search(r"\.\d{3},\d{2}$", s):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


_AMOUNT = r"[\d][\d,. \tOolCQDG]{1,20}"
_MONEY_RE = re.compile(r"(?:₦|NGN|N|USD|\$)?\s*(" + _AMOUNT + r")")

# Shape-strict money for the unlabeled fallback: thousands groups or a
# 1-2 digit decimal part. A bare 13-digit barcode never matches.
_STRICT_MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+\.\d{2}")

# Label-aware amount search: 'Total: ₦ 1,343,750' beats a bare number.
# Labels may carry a parenthetical currency/code between label and amount,
# e.g. "TOTAL (RM) : 332.30", "(Excluded GST) Sub Total (RM) : 313.50",
# "Total GST (RM) : 18.81". OCR often mangles the bracket — "(RM}" with a
# brace instead of a paren, "{RM)" with a brace instead of an opener — so
# both bracket shapes are accepted on either side.
_CURR = r"(?:₦|\$|RM|MYR|NGN)?"
_PAREN = r"(?:\s*[\(\{][A-Za-z]{1,6}[\)\}]|\s*[\(\{]\d[\d\.]*%?[\)\}]?)?"

_LABEL_AMOUNTS = [
    # (?<!sub\s)(?<!sub) before "total" matters: without it "Subtotal" /
    # "Sub Total" matches the total pattern first and returns the subtotal
    # as the total. "Total GST" is not a total either — after "Total" the
    # next token must not be GST (the amount group then fails to match).
    ("total", re.compile(r"(?<!sub\s)(?<!sub)(?:grand\s+)?\btotal" + _PAREN + r"\s*(?:due|amount|payable)?\s*[:\-]?\s*" + _CURR + r"\s*(" + _AMOUNT + r")", re.I)),
    ("subtotal", re.compile(r"sub[-\s]?total" + _PAREN + r"\s*[:\-]?\s*" + _CURR + r"\s*(" + _AMOUNT + r")", re.I)),
    # GST is a VAT-equivalent tax id/amount: "Total GST (RM) : 18.81".
    # Bare "GST" must not match "GST NO : 000800589824" (a tax number, not
    # an amount) — the (?!\s*NO\b) guard blocks that.
    ("vat", re.compile(r"(?:VAT|Value Added Tax|Total\s+GST|GST(?!\s*NO\b))" + _PAREN + r"\s*[:\-]?\s*" + _CURR + r"\s*(" + _AMOUNT + r")", re.I)),
]


def _label_amount(text: str, key: str) -> float | None:
    for name, rx in _LABEL_AMOUNTS:
        if name == key:
            m = rx.search(text)
            if m:
                return _clean_amount(m.group(1))
    return None


def _first(rx: re.Pattern, text: str, group: int = 1) -> str | None:
    m = rx.search(text)
    return m.group(group).strip() if m and m.group(group) else None


# --- Table parsing from word boxes ---------------------------------------
# Line-item rows on a real receipt are a *layout*: code | qty | unit | amount,
# with the description printed on the line below. Flattened OCR text loses
# that structure, so a line-regex can only parse rows whose description
# happens to sit on the same line (4-5 of 12 on the SROIE receipt). Using
# Tesseract's word boxes, every row is parsed by column position instead.
_BOX_MONEY = re.compile(r"^\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?$|^\d{1,4}$")
_TABLE_STOP = re.compile(
    r"gst\s*summary|tax\s*summary|totqty|\b(?:sub[\s-]*total|total|cash|"
    r"change|rounding|discount|tot\s*qty|amount\s*due|balance)\b", re.I)


def _norm_box_token(token: str) -> str:
    """Clean a single word from a money column ('170.0C' -> '170.00')."""
    t = token.strip().strip(".,;:!*'\"()[]{}")
    t = t.replace("O", "0").replace("o", "0")
    t = re.sub(r"(?<=[\d.])[CQDG]{1,3}$",
               lambda m: "0" * len(m.group(0)), t)
    t = re.sub(r"(?<=\d)[CQDG]$", "0", t)
    t = re.sub(r"(?<=\d)[lI|]$", "1", t)
    return t


def _group_word_lines(words: list[dict]) -> list[list[dict]]:
    """Group word boxes into visual lines by vertical position."""
    ws = [w for w in words if str(w.get("text", "")).strip()]
    if not ws:
        return []
    heights = sorted((w.get("height") or 0) for w in ws)
    med_h = heights[len(heights) // 2] or 10
    tol = max(4, 0.6 * med_h)
    ws.sort(key=lambda w: (w.get("top", 0), w.get("left", 0)))
    lines: list[list[dict]] = []
    current: list[dict] = []
    current_top = None
    for w in ws:
        top = w.get("top", 0)
        if current_top is not None and abs(top - current_top) > tol:
            lines.append(sorted(current, key=lambda x: x.get("left", 0)))
            current = []
        if not current:
            current_top = top
        current.append(w)
    if current:
        lines.append(sorted(current, key=lambda x: x.get("left", 0)))
    return lines


def _looks_precise(raw: str) -> bool:
    """True for money written with cents or thousands separators.

    '1,000,000' and '24.00' are precise; a bare '2400' is not — but a bare
    integer is only suspicious when the rest of the document is written
    with separators/cents (see _items_from_word_pages).
    """
    return bool(re.search(r"[.,]\d{1,2}$|[.,]\d{3}", raw))


def _items_from_word_pages(word_pages: list[list[dict]] | None) -> list[dict[str, Any]]:
    """Parse line items as a table: qty | unit price | amount columns.

    Only rows that are arithmetically plausible or carry a decimal amount
    are accepted, so header/footer noise ("TEL: 07-386 7524 FAX: ...") and
    item-code columns (13-digit barcodes) are never mistaken for amounts.
    """
    items: list[dict[str, Any]] = []
    amount_styles: list[tuple[dict[str, Any], str]] = []
    for words in word_pages or []:
        for line in _group_word_lines(words):
            tokens = [str(w["text"]).strip() for w in line]
            joined = " ".join(tokens)
            if not joined:
                continue
            if _TABLE_STOP.search(joined):
                break
            nums: list[str] = []
            for tok in tokens:
                norm = _norm_box_token(tok)
                if _BOX_MONEY.match(norm):
                    nums.append(norm)
            if len(nums) < 2:
                # Description line (or a code-only line): attach it to the
                # previous item so the row keeps its description.
                text = " ".join(t for t in tokens if not _BOX_MONEY.match(_norm_box_token(t)))
                text = re.sub(r"\s+", " ", text).strip()
                if items and text:
                    prev = items[-1]
                    prev["description"] = (prev["description"] + " " + text).strip()[:200]
                continue

            amount_raw = nums[-1]
            unit_raw = nums[-2] if len(nums) >= 2 else None
            qty_raw = nums[-3] if len(nums) >= 3 else None
            amount = _clean_amount(amount_raw)
            unit = _clean_amount(unit_raw) if unit_raw else None
            qty = _clean_amount(qty_raw) if qty_raw else None
            if amount is None or amount <= 0 or amount > 1_000_000:
                continue
            qty = qty if qty and 0 < qty <= 999 else None
            # Accept a row when it looks like money (decimal part) or when
            # qty x unit reconciles with the amount. Bare integers that do
            # neither are noise, not line items.
            has_decimal = _looks_precise(amount_raw)
            consistent = (qty is not None and unit is not None
                          and abs(qty * unit - amount) <= max(1.0, 0.02 * amount))
            if not (has_decimal or consistent):
                continue
            desc_bits = [t for t in tokens if not _BOX_MONEY.match(_norm_box_token(t))]
            item = {
                "description": re.sub(r"\s+", " ", " ".join(desc_bits)).strip()[:200],
                "qty": qty,
                "unit_price": unit,
                "amount": amount,
                # Provisional; finalised below against the whole document.
                # "2400" where every other amount has cents is most likely a
                # dropped decimal point: the value is kept as read but
                # flagged so the sum check can exclude it rather than report
                # a confidently wrong total.
                "amount_uncertain": False,
            }
            items.append(item)
            amount_styles.append((item, amount_raw))

    # A bare-integer amount is only uncertain when the document itself uses
    # cents / thousands separators elsewhere. A whole-number invoice must
    # not be flagged just for having no decimal part.
    doc_uses_precision = any(_looks_precise(raw) for _, raw in amount_styles)
    if doc_uses_precision:
        for item, raw in amount_styles:
            item["amount_uncertain"] = not _looks_precise(raw)
    return items



def extract_fields(text: str, ocr_confidence: float | None = None,
                   word_pages: list[list[dict]] | None = None) -> dict[str, Any]:
    """Extract invoice fields from OCR text. Returns dict with None for missing.

    `word_pages` are Tesseract word boxes (per page) when available; they let
    the line-item table be parsed by column position. Digital PDFs have no
    OCR boxes, so they keep the flattened-line path.
    """
    t = text or ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    # --- vendor: first line that looks like a company name (not a label) ---
    # Strict pass first: a real company line has >=2 words made of letters
    # only (length >= 3) and no digits. That skips handwritten scribbles
    # and OCR noise at the top of a scan ("Aio3e q H" from a handwritten
    # "A03041") which the loose rule used to accept as the vendor.
    # Fallback to the loose rule if no line qualifies (single-word vendor).
    def _vendor_candidate(ln: str) -> bool:
        if re.match(r"^(invoice|tax|date|po|phone|email|address)", ln, re.I):
            return False
        words = [w for w in re.split(r"\s+", ln) if w]
        return sum(1 for w in words if re.fullmatch(r"[A-Za-z]{3,}", w)) >= 2

    vendor = None
    for ln in lines[:8]:
        if _vendor_candidate(ln):
            vendor = re.sub(r"\s+", " ", ln)[:200]
            break
    if vendor is None:
        for ln in lines[:6]:
            if re.match(r"^(invoice|tax|date|po|phone|email|address)", ln, re.I):
                continue
            if len(ln) >= 3 and re.search(r"[A-Za-z]{3,}", ln):
                vendor = re.sub(r"\s+", " ", ln)[:200]
                break
    if vendor:
        # Strip trailing OCR noise, e.g. "PERNIAGAAN ZHENG HU} ." -> the
        # name, not the stray punctuation. Only trailing chars that can
        # never end a company name are removed.
        vendor = re.sub(r"[\s\}\{\)\]\.,;:\"'\-!]+$", "", vendor).strip()

    # Require the label (No./number/#) AND a digit in the captured token.
    # Without both, "TAX INVOICE" alone matches and captures the next word
    # ("Invoice") as the invoice number — the bug the unit tests caught.
    # Receipts use "RECEIPT #: CS00084670" / "DOCUMENT NO : TD01167104"
    # instead of an invoice number — accept those as the invoice number.
    # \s* (not \s+) after the keyword: OCR often reads "Receipt#: ..."
    # with no space between the word and the '#'.
    invoice_number = _first(
        re.compile(r"(?:invoice|receipt|document)\s*(?:no\.?|number|#|ref)\s*[^A-Za-z0-9]{0,5}\s*([A-Za-z0-9][A-Za-z0-9\-/]*\d[A-Za-z0-9\-/]*)", re.I), t)
    po_number = _first(
        re.compile(r"(?:P\.?\s?O\.?|purchase\s+order)\s+(?:no\.?|number|#)\s*[:\-#]?\s*([A-Za-z0-9][A-Za-z0-9\-/]*\d[A-Za-z0-9\-/]*)", re.I), t)
    # Tax id: TIN / Tax ID, but also GST No / VAT ID / GSTIN, e.g.
    # "GST NO : 000800589824" on a Malaysian simplified tax invoice.
    tin = _first(
        re.compile(r"(?:GST\s*(?:NO\.?|ID|IN)?|VAT\s*(?:ID|NO\.?)?|GSTIN|TIN|Tax\s*(?:ID|Identification|No\.?)(?:\s*No\.?)?)\s*[:\-]?\s*([0-9][0-9\-]{7,})", re.I), t)
    # TIN written as "Tax ID (TIN): 12345678-0001" — the (TIN) variant above
    # misses the colon form; catch it explicitly.
    if not tin:
        tin = _first(re.compile(r"\(TIN\)\s*[:\-]?\s*([0-9][0-9\-]{7,})", re.I), t)

    def _date(label: str) -> str | None:
        rx = re.compile(label + r"\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})", re.I)
        return _first(rx, t)

    # "data" is a common OCR misread of "date" ("Salesperson: Data:
    # 09/03/2018") — accept it as a date label.
    invoice_date = _date(r"(?:invoice\s+)?(?:date|data)")
    due_date = _date(r"(?:due\s+date|payment\s+due|pay\s+by)")

    # Currency from the document, not assumed. RM/MYR (Malaysian Ringgit)
    # appears as "(RM)" after amounts; ₦/NGN is Naira, $ is USD.
    _upper = t.upper()
    if "₦" in t or "NGN" in _upper:
        currency = "NGN"
    elif re.search(r"\bRM\b", _upper) or "MYR" in _upper:
        currency = "MYR"
    elif "$" in t:
        currency = "USD"
    else:
        currency = None

    subtotal = _label_amount(t, "subtotal")
    vat_amount = _label_amount(t, "vat")
    total = _label_amount(t, "total")
    # Fallback (guarded): only if no "Total" label matched. The old
    # fallback — largest money-like number — picked up barcodes / GST
    # numbers (13-digit integers like 9555752615725) and returned
    # ~₦9.9 trillion as the total. The fallback is now much stricter:
    # a candidate must be *shaped* like money (thousands separators or a
    # 1-2 digit decimal part), sit under 10,000,000, and must not be a
    # bare long integer. If nothing qualifies the total stays None and
    # compliance reports a WARN instead of a guessed number.
    if total is None:
        candidates = []
        for m in _STRICT_MONEY_RE.finditer(t):
            a = _clean_amount(m.group(0))
            if a is not None and 0 < a < 10_000_000:
                candidates.append(a)
        total = max(candidates) if candidates else None


    # --- line items: rows like "Steel pipes  10  25,000  250,000" ---
    # Preferred path: the OCR word boxes, parsed as qty | unit | amount
    # columns (handles the very common "description on the next line"
    # receipt layout). Falls back to the line regex for digital PDFs and
    # any page where no table row was recognised.
    line_items: list[dict[str, Any]] = _items_from_word_pages(word_pages)
    # Tolerant of OCR noise: a "=" where a space should be ("1 = 35.00"),
    # and trailing stray punctuation ("170.0 .").
    item_rx = re.compile(
        r"^(.{3,60}?)\s+(\d+(?:\.\d+)?)\s+(?:x\s+|=+\s+)?([\d,\.Ool]+)\s+([\d,\.Ool]+)[\s\.\,]*$"
    )
    for ln in lines if not line_items else []:
        # A GST/tax summary block at the end of the document is not line
        # items — its rows ("SR 6 313.50 18.81") would otherwise be parsed
        # as items and corrupt the line-item sum check.
        if re.search(r"gst\s+summary|tax\s+summary", ln, re.I):
            break
        m = item_rx.match(ln)
        if m and not re.search(r"total|subtotal|vat|tax|balance|amount due", ln, re.I):
            amount = _clean_amount(m.group(4))
            if amount is not None:
                line_items.append({
                    "description": m.group(1).strip(),
                    "qty": float(m.group(2)),
                    "unit_price": _clean_amount(m.group(3)),
                    "amount": amount,
                    "amount_uncertain": False,
                })

    bank_details = None
    bank_m = re.search(r"(bank.{0,200})", t, re.I | re.S)
    if bank_m:
        bank_details = re.sub(r"\s+", " ", bank_m.group(1)).strip()[:500]

    fields: dict[str, Any] = {
        "vendor": vendor,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "po_number": po_number,
        "tin": tin,
        "currency": currency,
        "subtotal": subtotal,
        "vat_amount": vat_amount,
        "total": total,
        "line_items": line_items,
        "bank_details": bank_details,
    }
    required = ["vendor", "invoice_number", "invoice_date", "po_number", "tin", "total"]
    fields["missing_fields"] = [k for k in required if not fields.get(k)]
    fields["ocr_confidence"] = ocr_confidence
    fields["low_confidence"] = bool(ocr_confidence is not None and ocr_confidence < 60)
    return fields
