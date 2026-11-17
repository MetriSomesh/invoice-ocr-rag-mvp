"""Settings — loaded from .env, never hard-code secrets here.

GEMINI_API_KEY is read only from the environment / .env file.
If it is missing the app still starts; RAG answers use a clearly
labelled extractive fallback (see app/rag.py) and never fake an LLM answer.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash").strip()
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "models/text-embedding-004"
    ).strip()
    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'euniserv.db'}"
    )
    chroma_dir: str = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "").strip()
    poppler_path: str | None = os.getenv("POPPLER_PATH") or None
    ocr_dpi: int = int(os.getenv("OCR_DPI", "300"))
    ocr_lang: str = os.getenv("OCR_LANG", "eng")
    # OCR engine for camera scans: "auto" (RapidOCR if installed, else
    # Tesseract), "rapidocr" (force, error if missing) or "tesseract".
    ocr_engine: str = os.getenv("OCR_ENGINE", "auto").strip().lower()

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()
