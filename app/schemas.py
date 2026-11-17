"""Pydantic schemas for API requests / responses."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = ""
    qty: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class ComplianceCheckOut(BaseModel):
    name: str
    status: str
    detail: str = ""
    amount: Optional[float] = None


class ChunkOut(BaseModel):
    chunk_id: str
    section: str
    page: int
    text: str


class Citation(BaseModel):
    chunk_id: str
    section: str
    page: int
    score: Optional[float] = None
    snippet: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    model: str  # e.g. "gemini-2.5-flash" or "extractive-fallback"
    is_fallback: bool = False
    warning: Optional[str] = None


class InvoiceOut(BaseModel):
    id: int
    filename: str
    file_type: str
    ocr_method: str
    ocr_confidence: Optional[float] = None
    vendor: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    po_number: Optional[str] = None
    tin: Optional[str] = None
    currency: Optional[str] = None
    subtotal: Optional[float] = None
    vat_amount: Optional[float] = None
    total: Optional[float] = None
    expected_po_total: Optional[float] = None
    line_items: list[dict[str, Any]] = []
    bank_details: Optional[str] = None
    missing_fields: list[str] = []
    risk_score: Optional[int] = None
    amount_at_risk: Optional[float] = None
    verdict: Optional[str] = None

    class Config:
        from_attributes = True
