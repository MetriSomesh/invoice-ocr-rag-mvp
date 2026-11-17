# EuniServ Invoice MVP — full local Python backend

Real upload → OCR → extraction → compliance → LangChain RAG. No simulation:
an uploaded file is actually read, and missing fields stay missing.

**For the Thursday 19 Nov call with Raymond Kalaiyo (EuniServ):** this is the
2-week prototype core — OCR + layout-aware chunking + RAG — running locally
on your own computer. Show the upload of a camera-scanned invoice live.

## Tech stack (all actually wired)

| Layer | Tech |
|---|---|
| API | Python 3.11+, FastAPI + Uvicorn (`app/main.py`) |
| RAG | LangChain (`langchain`, `langchain-community`, `langchain-chroma`, `langchain-google-genai`): layout-aware chunking, Chroma vector store (`./chroma_db`), Gemini LLM answers with citations |
| Embeddings | Google Generative AI `models/text-embedding-004` (primary); `sentence-transformers` local fallback if no key |
| LLM | Google Gemini `gemini-2.5-flash` (configurable via `LLM_MODEL` in `.env`) |
| Database | SQLite + SQLAlchemy (`euniserv.db`): invoices, extracted fields, chunks, compliance checks, audit trail |
| OCR — camera scans | **RapidOCR** (PP-OCRv4 det+rec via ONNX, bundled models) by default; Tesseract 5 + OpenCV fallback (`OCR_ENGINE=auto\|rapidocr\|tesseract`), OpenCV preprocessing, Pillow |
| OCR — digital PDFs | PyMuPDF text layer first; OCR only if the text layer is empty/short |
| Extraction | Regex + layout-aware chunking (`app/extract.py`, `app/chunking.py`) |
| Compliance | 7 checks in `app/compliance.py`, risk 0–100, verdict APPROVED / REVIEW / HOLD |
| UI | FastAPI-served static page (`app/static/`) — upload, results, RAG ask, raw OCR text, audit download |

## 1. Install system dependencies (Tesseract + Poppler)

**Only needed for the Tesseract fallback.** The default `OCR_ENGINE=auto`
uses RapidOCR, which ships its models in the pip package and needs no
system binaries (PDF pages are then rendered by PyMuPDF).

**macOS (Homebrew):**
```bash
brew install tesseract poppler
```

**Windows:**
1. Tesseract: https://github.com/UB-Mannheim/tesseract/wiki — install, then add its folder to PATH (or set `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` in `.env`).
2. Poppler: https://github.com/oschwartz10612/poppler-windows/releases — unzip, add its `bin\` folder to PATH (or set `POPPLER_PATH=...` in `.env`).

Verify: `tesseract --version` and `pdftoppm -v` should both print a version.

## 2. Run it

```bash
cd euniserv-invoice-mvp
cp .env.example .env
# Edit .env and add your Gemini key:
#   GEMINI_API_KEY=AIza...
# Get one free at https://aistudio.google.com/app/apikey

# macOS / Linux:
bash run.sh
# Windows:
run.bat
# …or manually:
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open **http://127.0.0.1:8000** — check `/health` first: it reports whether
Gemini, Tesseract and Poppler were detected.

> **Do not run the server on a shared/public host with real invoices.**
> It is a local demo: no auth, files are stored in `./uploads/`.

## 3. Upload a camera-scanned PDF (the real test)

1. Photograph / scan a paper invoice with your phone camera — straight-on,
   good light, whole page in frame. Save/export as **PDF** (or JPG/PNG).
2. In the UI, drop the file (or Browse), optionally enter the **expected PO
   total** in ₦ (this enables the PO-variance check), then **Process invoice**.
3. Camera scans take 10–30 s: each page is rendered at 300 DPI and OCR'd.
4. Check the result:
   - **OCR method** shows `camera-scan-ocr` (not `digital-pdf-text`) and a
     confidence % — that is Tesseract's mean per-word confidence.
   - **Show raw OCR text** — verify exactly what was read. If a field is
     wrong in the raw text, the extracted field will be wrong too; that is
     honest, not a bug to hide.
   - **Missing fields** are listed, never invented. Low confidence (<60%)
     adds +10 to the risk score.
   - **Ask the invoice**: type a question, e.g. *"What is the VAT amount
     and is the PO total matched?"* — the answer cites the chunks used.

A test file is included: generate it with
`python scripts/generate_sample_invoice.py` → `samples/sample-invoice-INV-9912.pdf`
(digital PDF — good for the text-layer path; print + photograph it to test
the camera-scan path).

## 4. How OCR accuracy is reported

There is **no single claimed accuracy %** for this MVP — accuracy depends
on the scan. What you get per upload:

- **Digital PDF:** text layer, effectively exact (no OCR error).
- **Camera scan:** per-line confidence (RapidOCR) or per-word confidence
  (Tesseract), averaged and shown on the result with the engine that ran.
  Clean, straight, well-lit print typically reads well; blur, tilt, shadows,
  handwriting and crumpled paper drop confidence and lose fields — the app
  shows that instead of guessing.
- **Which engine?** `OCR_ENGINE=auto` (default) uses **RapidOCR** when it is
  installed and falls back to Tesseract otherwise. On a 200 DPI phone photo
  of the sample invoice, RapidOCR read every field correctly and returned
  APPROVED, while Tesseract lost the totals block (`total` came back as 250
  instead of 1,343,750) and returned HOLD. On a clean 300 DPI scan both are
  correct. RapidOCR's confidence scale runs higher than Tesseract's, so
  compare engines on extracted fields, never on confidence alone.
- **Preprocessing (Tesseract path):** grayscale → deskew → **global Otsu**
  threshold. Deliberate: on a 200 DPI phone scan adaptive thresholding
  eroded glyph strokes (`18.81`→`18.84`, `CS00084670`→`0500084670`) while
  Otsu read both correctly.
- **Line items:** parsed from Tesseract word boxes by column position
  (qty / unit price / amount), so a receipt that prints the description on
  the line *below* the numbers still parses. An amount read without a
  decimal point (`2400` where the rest of the document uses `24.00`) is kept
  as read, but excluded from the line-item sum and named in the check detail
  rather than turning a clean receipt into a FAIL.
- **How to measure it yourself:** upload a scan whose true text you know,
  open *Show raw OCR text*, and compare character-by-character. The
  extraction unit tests (`tests/test_extraction.py`) verify the downstream
  logic on clean text: subtotal 1,250,000 + VAT 93,750 = total 1,343,750,
  PO variance, TIN, duplicate detection — run them with
  `python -m pytest tests/ -v`.

To improve a bad scan: re-photograph it straighter / brighter, or raise
`OCR_DPI=300` to `400` in `.env`. Handwriting is out of scope for Tesseract
in this MVP — flag that on the call if Raymond asks.

## 5. API (for the call / for curl)

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/health` | Gemini / Tesseract / Poppler status |
| POST | `/upload` | multipart `file` + optional `expected_po_total` |
| GET | `/invoices` | list, newest first (empty = real empty state) |
| GET | `/invoices/{id}` | invoice + chunks + checks |
| GET | `/invoices/{id}/raw-text` | raw OCR text |
| POST | `/invoices/{id}/ask` | `{"question": "..."}` → answer + citations |
| GET | `/invoices/{id}/audit` | JSON audit-trail download |

## 6. What needs your key / installs, and what works without them

- **Without `GEMINI_API_KEY`:** the app still runs. Embeddings fall back
  to `sentence-transformers` if installed, otherwise retrieval uses BM25;
  LLM answers fall back to an *extractive* answer (the best-matching OCR
  lines, verbatim, with their chunk ids — or an explicit "no line matches"
  when nothing does), labelled `extractive-fallback` — never a fake LLM answer.
- **Without Tesseract/Poppler:** camera scans still work through RapidOCR
  (PyMuPDF renders the PDF); digital PDFs use the text layer. Set
  `OCR_ENGINE=tesseract` only if you want the Tesseract path.
- **No key is ever hard-coded.** `.env` is gitignored; only `.env.example`
  is in the repo.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Tesseract not found` | Only when `OCR_ENGINE=tesseract`: install Tesseract (step 1) or set `TESSERACT_CMD`. With `OCR_ENGINE=auto` RapidOCR is used instead |
| `Poppler (pdftoppm) not found` | Set `POPPLER_PATH`, or ignore it — PyMuPDF renders the PDF when Poppler is absent |
| OCR returns empty text | Scan is too dark/blurry — re-photograph; check *raw OCR text* |
| `Gemini call failed` on Ask | Check `GEMINI_API_KEY` in `.env` and that the key is enabled for Gemini API; the UI falls back to extractive answers |
| `pip install` fails on `chromadb` / `sentence-transformers` | Use Python 3.11 or 3.12; on Apple Silicon make sure pip is up to date (`pip install -U pip`). `sentence-transformers` is heavy — you can comment it out of `requirements.txt` if you always use a Gemini key |
| Port 8000 busy | `uvicorn app.main:app --port 8001` |

## Project layout

```
app/main.py        FastAPI app + all endpoints + UI serving
app/config.py      .env settings (no secrets in code)
app/database.py    SQLite engine / session
app/models.py      Invoice / InvoiceChunk / ComplianceCheck / AuditEvent
app/ocr.py         digital text layer + OpenCV + Tesseract camera-scan OCR
app/extract.py     regex field extraction (None, never invented)
app/chunking.py    layout-aware chunking (header/line_items/totals/tax/bank/terms)
app/compliance.py  7 checks, risk score, verdict
app/rag.py         LangChain Chroma + Gemini, BM25 + extractive fallbacks
app/schemas.py     Pydantic request/response models
app/static/        index.html / app.js / styles.css (UI)
scripts/generate_sample_invoice.py   sample PDF generator (reportlab)
tests/test_extraction.py             pure-logic unit tests (offline)
samples/  uploads/  chroma_db/  euniserv.db   created at runtime
```
