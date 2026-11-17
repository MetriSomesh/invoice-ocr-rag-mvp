"""Generate a sample invoice PDF for local testing (no external data).

Creates samples/sample-invoice-INV-9912.pdf — a DIGITAL PDF with a text
layer (so the text-layer path is exercised). To test camera-scan OCR,
print it and photograph it, or screenshot it as a JPG and upload that.

Run: python scripts/generate_sample_invoice.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "samples" / "sample-invoice-INV-9912.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

LINES = [
    "West Africa Logistics Ltd",
    "TAX INVOICE",
    "Invoice No: INV-9912",
    "Date: 12/11/2026",
    "Due Date: 12/12/2026",
    "PO No: PO-4471",
    "Tax ID (TIN): 12345678-0001",
    "",
    "Description          Qty   Unit Price   Amount",
    "Steel pipes          10    100,000      1,000,000",
    "Valves               5     50,000       250,000",
    "",
    "Subtotal: 1,250,000",
    "VAT (7.5%): 93,750",
    "Total: 1,343,750",
    "Bank: First Bank Nigeria, Acct 2034567890",
]

c = canvas.Canvas(str(OUT), pagesize=A4)
c.setFont("Helvetica", 12)
y = 780
for line in LINES:
    c.drawString(60, y, line)
    y -= 22
c.save()
print(f"Wrote {OUT}")
