"""Real OCR for camera-scanned PDFs and image scans.

Pipeline (in order):
  1. PDF with a text layer (digital PDF) -> PyMuPDF text extraction.
     This is exact, no OCR error. Used only when the layer yields real text.
  2. Otherwise (camera-scanned PDF, or JPG/PNG) -> render each page to an
     image at 300 DPI (pdf2image / Poppler) and OCR it with either engine:
       * RapidOCR (PP-OCR models via ONNX, no system deps) — default when
         installed, because it is markedly better on photos of small print;
       * Tesseract 5 + OpenCV (grayscale -> deskew -> Otsu).
     `OCR_ENGINE=auto|rapidocr|tesseract` in .env chooses.
  3. Confidence is the mean per-word (Tesseract) or per-line (RapidOCR)
     score, 0-100. The two scales are not identical — RapidOCR's recognition
     scores run higher — so the engine is reported alongside the number and
     a cross-engine comparison is never made on confidence alone.
     Both engines return bounding boxes so extraction can parse the
     line-item table by column position.

Tesseract needs system binaries (see README): `tesseract` and Poppler
(`pdftoppm`) for PDF rendering. RapidOCR needs neither (PDF pages are
rendered by PyMuPDF as a fallback when Poppler is absent).
"""
from __future__ import annotations

import io
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from .config import settings

if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


@dataclass
class OCRPage:
    page: int
    text: str
    confidence: float | None  # mean word confidence, 0-100, or None for text-layer
    method: str  # "digital-pdf-text" | "camera-scan-ocr"
    # Word boxes for layout-aware extraction: {text, left, top, width, height, conf}.
    # Empty for text-layer pages (no OCR was run).
    words: list[dict] = field(default_factory=list)
    engine: str = ""  # "rapidocr" | "tesseract" | "" (text layer)


@dataclass
class OCRResult:
    full_text: str
    pages: list[OCRPage] = field(default_factory=list)
    method: str = "unknown"
    avg_confidence: float | None = None
    engine: str = ""  # which OCR engine actually ran

    @property
    def is_low_confidence(self) -> bool:
        return self.avg_confidence is not None and self.avg_confidence < 60


def _rapidocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


_RAPID_ENGINE = None


def _rapidocr_engine():
    """Lazily build the RapidOCR engine (model load is ~1-2 s, once)."""
    global _RAPID_ENGINE
    if _RAPID_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _RAPID_ENGINE = RapidOCR()
    return _RAPID_ENGINE


def _respace(text: str) -> str:
    """Restore word spaces that PP-OCR recognition drops.

    RapidOCR returns 'WestAfricaLogisticsLtd' / 'PO No' as one run. Two
    camel-case rules put the boundaries back without touching serials
    ('CS00084670', 'INV-9912') — letters next to digits are left alone
    because that is exactly how document numbers look.
    """
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    t = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def _rapid_to_page(result, page_no: int) -> OCRPage:
    """Convert RapidOCR output to boxes + text + confidence.

    Each detected box is one token for the table parser (PP-OCR already
    separates qty / unit price / amount into their own boxes), so the same
    column-based line-item parsing applies as for Tesseract words.
    """
    words: list[dict] = []
    confs: list[float] = []
    for entry in result or []:
        try:
            box, text, score = entry[0], entry[1], entry[2]
        except (TypeError, IndexError):
            continue
        token = _respace(str(text).strip())
        if not token:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        try:
            conf = float(score) * 100.0
        except (TypeError, ValueError):
            conf = -1.0
        if conf >= 0:
            confs.append(conf)
        words.append({
            "text": token,
            "left": int(min(xs)),
            "top": int(min(ys)),
            "width": int(max(xs) - min(xs)),
            "height": int(max(ys) - min(ys)),
            "conf": conf,
        })

    # Rebuild text line by line from the boxes (top -> left ordering).
    words.sort(key=lambda w: (w["top"], w["left"]))
    heights = sorted(w["height"] or 0 for w in words)
    med_h = heights[len(heights) // 2] if heights else 10
    tol = max(4, 0.6 * (med_h or 10))
    lines: list[str] = []
    current: list[dict] = []
    current_top = None
    for w in words:
        if current_top is not None and abs(w["top"] - current_top) > tol:
            lines.append(" ".join(x["text"] for x in sorted(current, key=lambda x: x["left"])))
            current = []
        if not current:
            current_top = w["top"]
        current.append(w)
    if current:
        lines.append(" ".join(x["text"] for x in sorted(current, key=lambda x: x["left"])))

    avg = round(sum(confs) / len(confs), 1) if confs else None
    return OCRPage(page=page_no, text="\n".join(lines).strip(), confidence=avg,
                   method="camera-scan-ocr", words=words)


def _ocr_rapidocr(pil_img: Image.Image, page_no: int) -> OCRPage:
    engine = _rapidocr_engine()
    arr = np.array(pil_img.convert("RGB"))
    result, _elapse = engine(arr)
    page = _rapid_to_page(result, page_no)
    page.engine = "rapidocr"
    return page



def _check_binaries() -> None:
    if shutil.which("tesseract") is None and not settings.tesseract_cmd:
        raise RuntimeError(
            "Tesseract not found on PATH. Install it first — macOS: 'brew install tesseract', "
            "Windows: install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and set TESSERACT_CMD in .env. See README.md."
        )


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct small rotation from text blobs, not from every dark pixel.

    The old version took minAreaRect over *all* dark pixels, so page rules,
    borders or a barcode tilted the whole page. Here only small, text-sized
    connected components vote, the median angle wins, and nothing is
    rotated for tiny (<0.5 deg) or implausible (>7 deg) results.
    """
    try:
        h, w = gray.shape[:2]
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
        angles: list[float] = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if not (30 <= area <= 2000) or ch > h * 0.05 or cw > w * 0.2:
                continue
            pts = np.column_stack(np.where(labels[y:y + ch, x:x + cw] == i))
            if len(pts) < 10:
                continue
            a = float(cv2.minAreaRect(pts)[-1])
            a = -(90 + a) if a < -45 else -a
            if abs(a) < 20:
                angles.append(a)
        if len(angles) < 20:
            return gray
        angle = float(np.median(angles))
        if abs(angle) < 0.5 or abs(angle) > 7:
            return gray
        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


def _preprocess(pil_img: Image.Image) -> np.ndarray:
    """Camera-scan cleanup: grey -> deskew -> global Otsu threshold.

    Global Otsu keeps stroke edges intact; adaptive thresholding was
    measurably worse on a 200 DPI phone scan (see module docstring), and
    the heavy `fastNlMeansDenoising` pass only blurred thin print.
    """
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = _deskew(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, clean = cv2.threshold(blurred, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return clean


def _ocr_tesseract(pil_img: Image.Image, page_no: int) -> OCRPage:
    """One Tesseract pass: word boxes + text + confidence, all from the same run.

    The old version ran Tesseract twice (image_to_data for confidence, then
    image_to_string for the text), so the reported confidence could belong
    to a different reading than the text shown. Now the text is rebuilt
    from the same word boxes, which also gives extraction the column
    positions it needs to parse line items as a table.
    """
    _check_binaries()
    clean = _preprocess(pil_img)
    data = pytesseract.image_to_data(
        clean, lang=settings.ocr_lang, config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )

    words: list[dict] = []
    lines: dict[tuple[int, int, int], list[int]] = {}
    confs: list[float] = []
    for i, text in enumerate(data.get("text", [])):
        token = str(text).strip()
        if not token:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError, IndexError):
            conf = -1.0
        if conf >= 0:
            confs.append(conf)
        words.append({
            "text": token,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "conf": conf,
        })
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(len(words) - 1)

    text_lines: list[str] = []
    for key in sorted(lines):
        idx = sorted(lines[key], key=lambda j: words[j]["left"])
        line = " ".join(words[j]["text"] for j in idx).strip()
        if line:
            text_lines.append(line)

    avg = round(sum(confs) / len(confs), 1) if confs else None
    return OCRPage(page=page_no, text="\n".join(text_lines).strip(),
                   confidence=avg, method="camera-scan-ocr", words=words,
                   engine="tesseract")


def _resolve_engine() -> str:
    """Pick the OCR engine from OCR_ENGINE + what is actually installed."""
    choice = (settings.ocr_engine or "auto").lower()
    if choice == "rapidocr":
        if not _rapidocr_available():
            raise RuntimeError(
                "OCR_ENGINE=rapidocr but rapidocr-onnxruntime is not installed. "
                "Run: pip install rapidocr-onnxruntime (or set OCR_ENGINE=auto)."
            )
        return "rapidocr"
    if choice == "tesseract":
        return "tesseract"
    return "rapidocr" if _rapidocr_available() else "tesseract"


def _ocr_page(pil_img: Image.Image, page_no: int) -> OCRPage:
    """OCR one page with the configured engine, degrading to Tesseract."""
    engine = _resolve_engine()
    if engine == "rapidocr":
        try:
            return _ocr_rapidocr(pil_img, page_no)
        except Exception as exc:  # noqa: BLE001 — never lose the page
            if (settings.ocr_engine or "auto").lower() == "rapidocr":
                raise RuntimeError(f"RapidOCR failed: {exc}") from exc
            page = _ocr_tesseract(pil_img, page_no)
            page.engine = f"tesseract (rapidocr failed: {exc})"
            return page
    return _ocr_tesseract(pil_img, page_no)


def _render_pdf_pages(p) -> list[Image.Image]:
    """Render PDF pages to images: Poppler if present, else PyMuPDF.

    PyMuPDF is already a dependency, so a camera-scan PDF still works on a
    machine without Poppler — it just uses the other renderer.
    """
    if shutil.which("pdftoppm") or settings.poppler_path:
        return convert_from_path(str(p), dpi=settings.ocr_dpi,
                                 poppler_path=settings.poppler_path)
    images: list[Image.Image] = []
    doc = fitz.open(p)
    for page in doc:
        pix = page.get_pixmap(dpi=settings.ocr_dpi)
        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return images



def _digital_pdf_text(path: Path) -> list[OCRPage] | None:
    """Return text-layer pages if the PDF has a usable text layer, else None.

    'Usable' = at least 50 non-space characters across the document.
    Camera scans have an empty / near-empty layer, so they fall through to OCR.
    """
    try:
        doc = fitz.open(path)
        pages: list[OCRPage] = []
        total = 0
        for i, page in enumerate(doc, start=1):
            text = (page.get_text() or "").strip()
            total += len(text)
            pages.append(OCRPage(page=i, text=text, confidence=None, method="digital-pdf-text"))
        doc.close()
        return pages if total >= 50 else None
    except Exception:
        return None


def process_file(path: str | Path) -> OCRResult:
    """Read a real file and return its text + per-page confidence.

    Accepts .pdf, .jpg/.jpeg, .png. No file content is invented: if OCR
    reads nothing, full_text is empty and confidence is None.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        text_pages = _digital_pdf_text(p)
        if text_pages is not None:
            full = "\n".join(pg.text for pg in text_pages if pg.text)
            return OCRResult(full_text=full, pages=text_pages,
                             method="digital-pdf-text", avg_confidence=None)
        # Camera-scanned PDF: render pages, then OCR each one.
        images = _render_pdf_pages(p)
        pages = [_ocr_page(img, i + 1) for i, img in enumerate(images)]
        confs = [pg.confidence for pg in pages if pg.confidence is not None]
        avg = round(sum(confs) / len(confs), 1) if confs else None
        full = "\n".join(pg.text for pg in pages if pg.text)
        method = pages[0].method if pages else "camera-scan-ocr"
        return OCRResult(full_text=full, pages=pages, method=method,
                         avg_confidence=avg,
                         engine=next((pg.engine for pg in pages if pg.engine), ""))

    if suffix in {".jpg", ".jpeg", ".png"}:
        pages = [_ocr_page(Image.open(p), 1)]
        return OCRResult(full_text=pages[0].text, pages=pages,
                         method=pages[0].method, avg_confidence=pages[0].confidence,
                         engine=pages[0].engine)

    if suffix == ".txt":
        text = p.read_text(encoding="utf-8", errors="ignore")
        return OCRResult(full_text=text.strip(),
                         pages=[OCRPage(page=1, text=text.strip(), confidence=None, method="text-file")],
                         method="text-file", avg_confidence=None)

    raise ValueError(f"Unsupported file type: {suffix} — upload PDF, JPG or PNG.")
