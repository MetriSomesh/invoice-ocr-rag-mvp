"""7 compliance checks, computed only from extracted values.

No check ever invents a value: a missing field is WARN/FAIL with the
reason in `detail`. A check that could not run because there was nothing
to check (no PO supplied, no due date printed) is INFO: it is shown to the
user but does not move the risk score or the verdict. Otherwise every
receipt without a purchase order would read REVIEW, and APPROVED would
mean "you filled in the form", not "the document passed".

Risk is a weighted sum of failed checks (0-100).
Verdict: HOLD if any blocker FAILs, REVIEW if any WARN, else APPROVED.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

# (name, weight towards risk score, is_blocker)
CHECKS = [
    ("maths_subtotal_vat_equals_total", 25, True),
    ("line_item_sum_matches_subtotal", 15, False),
    ("tin_vat_present", 20, True),
    ("po_variance", 20, True),
    ("duplicate_invoice", 10, False),
    ("completeness", 5, False),
    ("due_date_valid", 5, False),
]

# Fields whose absence actually makes an invoice incomplete. `po_number` is
# deliberately absent: many receipts, utility bills and simplified tax
# invoices carry no PO, so its absence is reported (in missing_fields) but
# is not a finding.
CORE_FIELDS = ("vendor", "invoice_number", "invoice_date", "tin", "total")


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def run_checks(fields: dict[str, Any], expected_po_total: float | None = None,
               existing_invoice_numbers: list[str] | None = None) -> dict[str, Any]:
    """fields = extract_fields() output. Returns {checks, risk_score, amount_at_risk, verdict}."""
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str, amount: float | None = None):
        checks.append({"name": name, "status": status, "detail": detail, "amount": amount})

    subtotal, vat, total = fields.get("subtotal"), fields.get("vat_amount"), fields.get("total")
    items = fields.get("line_items") or []

    # 1. Maths: subtotal + VAT == total (tolerance ₦1 for OCR rounding)
    if subtotal is not None and vat is not None and total is not None:
        diff = abs((subtotal + vat) - total)
        add("maths_subtotal_vat_equals_total",
            "PASS" if diff <= 1 else "FAIL",
            f"Subtotal {subtotal:,.2f} + VAT {vat:,.2f} = {subtotal + vat:,.2f} vs total {total:,.2f} (diff {diff:,.2f})",
            diff if diff > 1 else 0.0)
    else:
        add("maths_subtotal_vat_equals_total", "WARN",
            "Cannot verify maths — subtotal, VAT or total missing from OCR text.")

    # 2. Line-item sum vs subtotal
    # Rows whose amount was read without a decimal point ("2400" for 24.00)
    # are excluded from the sum and named in the detail — the value is not
    # silently repaired, and a single lost decimal cannot turn a clean
    # receipt into a FAIL with a huge phantom difference.
    if items and subtotal is not None:
        certain = [i for i in items if not i.get("amount_uncertain")]
        uncertain = [i for i in items if i.get("amount_uncertain")]
        s = sum(i.get("amount") or 0 for i in certain)
        diff = abs(s - subtotal)
        note = ""
        if uncertain:
            listed = ", ".join(f"{i.get('amount'):,.2f}" for i in uncertain[:4])
            note = (f" {len(uncertain)} row(s) excluded (amount read without a "
                    f"decimal point: {listed}).")
        add("line_item_sum_matches_subtotal",
            "PASS" if (diff <= 1 and not uncertain) else ("WARN" if uncertain else "FAIL"),
            f"Line items sum {s:,.2f} vs subtotal {subtotal:,.2f} (diff {diff:,.2f}).{note}",
            diff if diff > 1 else 0.0)
    else:
        add("line_item_sum_matches_subtotal", "WARN",
            "No line items or no subtotal extracted — sum cannot be verified.")

    # 3. TIN / VAT present
    if fields.get("tin"):
        add("tin_vat_present", "PASS", f"TIN present: {fields['tin']}")
    else:
        add("tin_vat_present", "FAIL", "No TIN / Tax ID found in the document — compliance blocker.")

    # 4. PO variance — only when the user supplied an expected PO total
    if expected_po_total is not None and total is not None:
        diff = abs(total - expected_po_total)
        add("po_variance",
            "PASS" if diff <= 1 else "FAIL",
            f"Invoice total {total:,.2f} vs expected PO total {expected_po_total:,.2f} (variance {diff:,.2f})",
            diff if diff > 1 else 0.0)
    elif expected_po_total is None:
        # Not a finding: the check was never requested. INFO, not WARN.
        add("po_variance", "INFO", "No expected PO total supplied — PO match not checked.")
    else:
        add("po_variance", "WARN", "No invoice total extracted — PO match cannot be checked.")

    # 5. Duplicate vs invoices already in the DB
    inv_no = fields.get("invoice_number")
    existing = existing_invoice_numbers or []
    if inv_no and inv_no in existing:
        add("duplicate_invoice", "FAIL", f"Invoice number {inv_no} already exists in the database.")
    elif inv_no:
        add("duplicate_invoice", "PASS", f"Invoice number {inv_no} is new.")
    else:
        add("duplicate_invoice", "WARN", "No invoice number extracted — duplicate check skipped.")

    # 6. Completeness — only CORE_FIELDS are required; a missing PO number
    # is reported in missing_fields but is not a finding (see CORE_FIELDS).
    core_missing = [m for m in missing if m in CORE_FIELDS]
    optional_missing = [m for m in missing if m not in CORE_FIELDS]
    if core_missing:
        add("completeness", "WARN",
            "Missing required fields: " + ", ".join(core_missing))
    else:
        detail = "All required fields present."
        if optional_missing:
            detail += " Not printed (optional): " + ", ".join(optional_missing) + "."
        add("completeness", "PASS", detail)

    # 7. Due date valid. A missing due date is INFO, not a warning: a cash
    # receipt legitimately has none. An unparseable one is a real finding.
    due = _parse_date(fields.get("due_date"))
    if fields.get("due_date") and due is None:
        add("due_date_valid", "WARN",
            f"Due date {fields.get('due_date')!r} could not be parsed.")
    elif due is None:
        add("due_date_valid", "INFO", "No due date printed — nothing to check.")
    elif due < date.today():
        add("due_date_valid", "FAIL", f"Due date {due.isoformat()} is in the past.")
    else:
        add("due_date_valid", "PASS", f"Due date {due.isoformat()} is valid.")

    # Risk: weighted sum of FAILs (full weight) + WARNs (half weight, capped).
    weights = dict((n, (w, b)) for n, w, b in CHECKS)
    risk = 0
    for c in checks:
        w, _ = weights[c["name"]]
        if c["status"] == "FAIL":
            risk += w
        elif c["status"] == "WARN":
            risk += w // 2
        # INFO contributes 0 — shown, not scored.
    risk = min(100, risk)
    if fields.get("low_confidence"):
        risk = min(100, risk + 10)

    amount_at_risk = sum((c.get("amount") or 0) for c in checks if c["status"] == "FAIL")
    blockers = [c for c in checks if c["status"] == "FAIL" and weights[c["name"]][1]]
    if blockers:
        verdict = "HOLD"
    elif any(c["status"] in ("FAIL", "WARN") for c in checks):
        verdict = "REVIEW"
    else:
        verdict = "APPROVED"
    return {"checks": checks, "risk_score": risk,
            "amount_at_risk": round(amount_at_risk, 2), "verdict": verdict}
