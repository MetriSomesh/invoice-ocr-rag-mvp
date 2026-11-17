"""FastAPI app — EuniServ Invoice MVP.

Run locally (do NOT run here — the user runs it on his own computer):
    uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
    then open http://127.0.0.1:8000

Endpoints:
  GET  /                        UI (static/index.html)
  GET  /health                  {status, gemini_configured, tesseract, poppler}
  POST /upload                  multipart PDF/JPG/PNG (+ optional expected_po_total)
  GET  /invoices                list, newest first (empty list = real empty state)
  GET  /invoices/{id}           invoice + chunks + checks
  GET  /invoices/{id}/raw-text  raw OCR text (verify what was actually read)
  POST /invoices/{id}/ask       {question} -> RAG answer + citations
  GET  /invoices/{id}/audit     JSON audit trail download
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from . import chunking, compliance, extract, ocr, rag
from .config import settings
from .database import get_db, init_db
from .models import AuditEvent, ComplianceCheck, Invoice, InvoiceChunk
from .schemas import AskRequest

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="EuniServ Invoice MVP", version="0.1.0")
settings.upload_dir.mkdir(parents=True, exist_ok=True)

ALLOWED = {".pdf": "pdf", ".jpg": "jpg", ".jpeg": "jpg", ".png": "png", ".txt": "txt"}


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": settings.has_gemini_key,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "ocr_engine": settings.ocr_engine,
        "rapidocr": ocr._rapidocr_available(),
        "tesseract": bool(shutil.which("tesseract") or settings.tesseract_cmd),
        "poppler": bool(shutil.which("pdftoppm") or settings.poppler_path),
        "chroma_dir": settings.chroma_dir,
    }


@app.post("/upload")
async def upload_invoice(
    file: UploadFile = File(...),
    expected_po_total: float | None = Form(default=None),
    db: Session = Depends(get_db),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type {suffix!r}. Upload PDF, JPG or PNG.")
    # Save the real uploaded file — never substitute a sample.
    stored = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    stored.write_bytes(await file.read())

    # 1. OCR (real — digital text layer first, camera-scan OCR otherwise)
    try:
        result = ocr.process_file(stored)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCR failed: {exc}")

    # 2. Extract fields from the actual OCR text (+ word boxes when OCR ran,
    #    so the line-item table is parsed by column position, not line order)
    word_pages = [p.words for p in result.pages if p.words]
    fields = extract.extract_fields(
        result.full_text,
        ocr_confidence=result.avg_confidence,
        word_pages=word_pages or None,
    )

    # 3. Layout-aware chunks
    pages = [{"page": p.page, "text": p.text} for p in result.pages]
    chunks = chunking.chunk_pages(pages)

    # 4. Compliance (duplicate check against invoices already in the DB)
    existing = [r[0] for r in db.query(Invoice.invoice_number).filter(Invoice.invoice_number.isnot(None)).all()]
    comp = compliance.run_checks(fields, expected_po_total=expected_po_total, existing_invoice_numbers=existing)

    # 5. Persist invoice + chunks + checks + audit
    inv = Invoice(
        filename=file.filename or stored.name, file_type=ALLOWED[suffix],
        ocr_method=result.method, ocr_confidence=result.avg_confidence,
        raw_text=result.full_text,
        vendor=fields["vendor"], invoice_number=fields["invoice_number"],
        invoice_date=fields["invoice_date"], due_date=fields["due_date"],
        po_number=fields["po_number"], tin=fields["tin"], currency=fields["currency"],
        subtotal=fields["subtotal"], vat_amount=fields["vat_amount"], total=fields["total"],
        expected_po_total=expected_po_total, line_items=fields["line_items"],
        bank_details=fields["bank_details"], missing_fields=fields["missing_fields"],
        risk_score=comp["risk_score"], amount_at_risk=comp["amount_at_risk"], verdict=comp["verdict"],
    )
    db.add(inv)
    db.commit()
    db.refresh(inv)
    for c in chunks:
        db.add(InvoiceChunk(invoice_id=inv.id, chunk_id=f"inv-{inv.id}-{c['chunk_id']}",
                            section=c["section"], page=c["page"], text=c["text"]))
    for c in comp["checks"]:
        db.add(ComplianceCheck(invoice_id=inv.id, name=c["name"], status=c["status"],
                               detail=c["detail"], amount=c.get("amount")))
    db.add(AuditEvent(invoice_id=inv.id, event="uploaded",
                      detail=f"{file.filename} via {result.method}, confidence={result.avg_confidence}"))
    db.add(AuditEvent(invoice_id=inv.id, event="ocr_done", detail=f"method={result.method}"))
    db.add(AuditEvent(invoice_id=inv.id, event="checked",
                      detail=f"verdict={comp['verdict']} risk={comp['risk_score']}"))
    db.commit()

    # 6. Index chunks in Chroma (embeddings need a key or sentence-transformers;
    #    failure here does NOT block the upload — RAG falls back to BM25).
    indexed, index_warning = 0, None
    try:
        indexed = rag.index_chunks(inv.id, [
            {**c, "chunk_id": f"inv-{inv.id}-{c['chunk_id']}"} for c in chunks
        ])
    except Exception as exc:  # noqa: BLE001
        index_warning = f"Vector indexing skipped: {exc}. RAG will use BM25 fallback."

    return {
        "id": inv.id, "filename": inv.filename, "ocr_method": result.method,
        "ocr_engine": result.engine or None,
        "ocr_confidence": result.avg_confidence,
        "low_confidence": fields.get("low_confidence"),
        "vendor": inv.vendor, "invoice_number": inv.invoice_number,
        "total": inv.total, "currency": inv.currency,
        "missing_fields": inv.missing_fields, "risk_score": inv.risk_score,
        "amount_at_risk": inv.amount_at_risk, "verdict": inv.verdict,
        "checks": comp["checks"], "chunks_indexed": indexed,
        "index_warning": index_warning,
    }


@app.get("/invoices")
def list_invoices(db: Session = Depends(get_db)):
    rows = db.query(Invoice).order_by(Invoice.id.desc()).all()
    return [{
        "id": r.id, "filename": r.filename, "vendor": r.vendor,
        "invoice_number": r.invoice_number, "total": r.total, "currency": r.currency,
        "verdict": r.verdict, "risk_score": r.risk_score,
        "ocr_method": r.ocr_method, "ocr_confidence": r.ocr_confidence,
    } for r in rows]


@app.get("/invoices/{invoice_id}")
def get_invoice(invoice_id: int, db: Session = Depends(get_db)):
    inv = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not inv:
        raise HTTPException(404, "Invoice not found.")
    chunks = db.query(InvoiceChunk).filter(InvoiceChunk.invoice_id == invoice_id).all()
    checks = db.query(ComplianceCheck).filter(ComplianceCheck.invoice_id == invoice_id).all()
    return {
        "id": inv.id, "filename": inv.filename, "file_type": inv.file_type,
        "ocr_method": inv.ocr_method, "ocr_confidence": inv.ocr_confidence,
        "vendor": inv.vendor, "invoice_number": inv.invoice_number,
        "invoice_date": inv.invoice_date, "due_date": inv.due_date,
        "po_number": inv.po_number, "tin": inv.tin, "currency": inv.currency,
        "subtotal": inv.subtotal, "vat_amount": inv.vat_amount, "total": inv.total,
        "expected_po_total": inv.expected_po_total, "line_items": inv.line_items or [],
        "bank_details": inv.bank_details, "missing_fields": inv.missing_fields or [],
        "risk_score": inv.risk_score, "amount_at_risk": inv.amount_at_risk,
        "verdict": inv.verdict,
        "chunks": [{"chunk_id": c.chunk_id, "section": c.section, "page": c.page, "text": c.text} for c in chunks],
        "checks": [{"name": c.name, "status": c.status, "detail": c.detail, "amount": c.amount} for c in checks],
    }


@app.get("/invoices/{invoice_id}/raw-text")
def raw_text(invoice_id: int, db: Session = Depends(get_db)):
    inv = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not inv:
        raise HTTPException(404, "Invoice not found.")
    return {"invoice_id": inv.id, "ocr_method": inv.ocr_method,
            "ocr_confidence": inv.ocr_confidence, "raw_text": inv.raw_text}


@app.post("/invoices/{invoice_id}/ask")
def ask(invoice_id: int, body: AskRequest, db: Session = Depends(get_db)):
    inv = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not inv:
        raise HTTPException(404, "Invoice not found.")
    chunks = db.query(InvoiceChunk).filter(InvoiceChunk.invoice_id == invoice_id).all()
    chunk_dicts = [{"chunk_id": c.chunk_id, "section": c.section, "page": c.page, "text": c.text} for c in chunks]
    result = rag.ask_question(invoice_id, body.question, chunk_dicts)
    db.add(AuditEvent(invoice_id=invoice_id, event="asked", detail=body.question[:300]))
    db.commit()
    return result


@app.get("/invoices/{invoice_id}/audit")
def audit(invoice_id: int, db: Session = Depends(get_db)):
    inv = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not inv:
        raise HTTPException(404, "Invoice not found.")
    events = db.query(AuditEvent).filter(AuditEvent.invoice_id == invoice_id).order_by(AuditEvent.id).all()
    payload = {
        "invoice_id": inv.id, "filename": inv.filename, "ocr_method": inv.ocr_method,
        "ocr_confidence": inv.ocr_confidence, "vendor": inv.vendor,
        "invoice_number": inv.invoice_number, "total": inv.total,
        "verdict": inv.verdict, "risk_score": inv.risk_score,
        "amount_at_risk": inv.amount_at_risk,
        "events": [{"event": e.event, "detail": e.detail, "at": str(e.created_at)} for e in events],
    }
    return JSONResponse(payload, headers={
        "Content-Disposition": f'attachment; filename="invoice-{invoice_id}-audit.json"'})


# UI last: serves app/static/index.html at / and static assets under /static
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))
