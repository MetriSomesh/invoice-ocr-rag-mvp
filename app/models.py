"""SQLAlchemy models. Nothing here is seeded with fake invoices —
an empty database is a real empty state (the UI shows 'No invoices yet').
"""
from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text, func

from .database import Base


class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(512), nullable=False)
    file_type = Column(String(32), nullable=False)  # pdf / jpg / png
    ocr_method = Column(String(64), nullable=False, default="unknown")
    # e.g. "digital-pdf-text" or "camera-scan-ocr"
    ocr_confidence = Column(Float, nullable=True)  # 0-100, mean word confidence
    raw_text = Column(Text, nullable=False, default="")

    # Extracted fields (None = not found in the OCR text — never invented)
    vendor = Column(String(512), nullable=True)
    invoice_number = Column(String(128), nullable=True, index=True)
    invoice_date = Column(String(32), nullable=True)
    due_date = Column(String(32), nullable=True)
    po_number = Column(String(128), nullable=True)
    tin = Column(String(64), nullable=True)
    currency = Column(String(16), nullable=True)
    subtotal = Column(Float, nullable=True)
    vat_amount = Column(Float, nullable=True)
    total = Column(Float, nullable=True)
    expected_po_total = Column(Float, nullable=True)
    line_items = Column(JSON, nullable=True)  # list of {description, qty, unit_price, amount}
    bank_details = Column(Text, nullable=True)
    missing_fields = Column(JSON, nullable=True)

    risk_score = Column(Integer, nullable=True)  # 0-100
    amount_at_risk = Column(Float, nullable=True)
    verdict = Column(String(32), nullable=True)  # APPROVED / REVIEW / HOLD

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class InvoiceChunk(Base):
    __tablename__ = "invoice_chunks"

    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, nullable=False, index=True)
    chunk_id = Column(String(128), nullable=False, index=True)  # e.g. inv-12-totals-0
    section = Column(String(64), nullable=False)  # header / line_items / totals / tax / bank / terms
    page = Column(Integer, nullable=False, default=1)
    text = Column(Text, nullable=False)


class ComplianceCheck(Base):
    __tablename__ = "compliance_checks"

    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    status = Column(String(16), nullable=False)  # PASS / FAIL / WARN
    detail = Column(Text, nullable=False, default="")
    amount = Column(Float, nullable=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, nullable=False, index=True)
    event = Column(String(128), nullable=False)  # uploaded / ocr_done / extracted / checked / asked
    detail = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
