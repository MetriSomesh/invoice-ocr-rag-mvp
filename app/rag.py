"""LangChain RAG: Chroma vector store + Gemini LLM, with honest fallbacks.

Primary path (needs GEMINI_API_KEY in .env):
  GoogleGenerativeAIEmbeddings (text-embedding-004) -> langchain-chroma
  Chroma (persistent, ./chroma_db) -> retriever (top-k=4)
  ChatGoogleGenerativeAI (gemini-2.5-flash) answers ONLY from the
  retrieved chunks; the prompt forbids using outside knowledge, and the
  answer is returned with citations (chunk_id, section, page, score).

Fallback paths (no key, or no sentence-transformers):
  * Embeddings: sentence-transformers (all-MiniLM-L6-v2) if installed,
    otherwise Chroma's default embedding is unavailable, so retrieval
    falls back to in-process BM25 over the stored chunks.
  * LLM: no key -> extractive answer: the best-matching OCR lines from the
    retrieved chunks, verbatim with their chunk ids (or an explicit
    "no line matches" when none do), labelled model="extractive-fallback"
    and is_fallback=True. Never a fake LLM answer.

All functions are lazy — importing this module never calls an API or
loads a model. That keeps `pytest` and `py_compile` offline-safe.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .config import settings

COLLECTION_PREFIX = "invoice"


def _gemini_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=settings.gemini_api_key
    )


def _local_embeddings():
    """sentence-transformers fallback. Raises ImportError if not installed."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_embedding_function():
    """Gemini first, sentence-transformers second. Raises if neither works."""
    if settings.has_gemini_key:
        return _gemini_embeddings()
    try:
        return _local_embeddings()
    except Exception as exc:  # noqa: BLE001 — fall through to BM25
        raise RuntimeError(
            "No embedding model available: set GEMINI_API_KEY in .env, or "
            "install sentence-transformers (pip install sentence-transformers)."
        ) from exc


def get_vectorstore(invoice_id: int):
    """LangChain Chroma vector store for one invoice's chunks."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=f"{COLLECTION_PREFIX}-{invoice_id}",
        embedding_function=get_embedding_function(),
        persist_directory=settings.chroma_dir,
    )


def index_chunks(invoice_id: int, chunks: list[dict]) -> int:
    """Embed + store chunks. Returns number indexed. Raises on no embeddings."""
    from langchain_core.documents import Document

    vs = get_vectorstore(invoice_id)
    docs = [
        Document(page_content=c["text"], metadata={
            "chunk_id": c["chunk_id"], "section": c["section"], "page": c["page"],
            "invoice_id": invoice_id,
        })
        for c in chunks
    ]
    if docs:
        vs.add_documents(docs)
    return len(docs)


# --- BM25 fallback (no embeddings at all) ---------------------------------

def _bm25(query: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
    """Small BM25 over chunk texts. Returns [{chunk, score}] sorted desc."""
    def tokens(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    docs = [tokens(c["text"]) for c in chunks]
    n = len(docs)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in docs) / n
    df = Counter()
    for d in docs:
        for tok in set(d):
            df[tok] += 1
    q = tokens(query)
    k1, b = 1.5, 0.75
    scored = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        score = 0.0
        for tok in q:
            if tok not in tf:
                continue
            idf = math.log(1 + (n - df[tok] + 0.5) / (df[tok] + 0.5))
            score += idf * (tf[tok] * (k1 + 1)) / (tf[tok] + k1 * (1 - b + b * len(d) / avgdl))
        scored.append({"chunk": chunks[i], "score": round(score, 3)})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


# --- extractive answer (no LLM configured) --------------------------------

_STOPWORDS = {
    "what", "which", "who", "when", "where", "why", "how", "is", "are", "was",
    "were", "be", "been", "the", "a", "an", "of", "for", "on", "in", "at", "to",
    "and", "or", "did", "do", "does", "this", "that", "these", "those", "it",
    "its", "there", "any", "all", "much", "many",
}


def _query_terms(question: str) -> list[str]:
    """Content words of the question, for line-level matching."""
    toks = re.findall(r"[a-z0-9][a-z0-9.\-/]*", question.lower())
    return [t.strip(".-/") for t in toks
            if t.strip(".-/") not in _STOPWORDS and len(t.strip(".-/")) > 1]


# One field, several printed names. A Malaysian receipt labels the TIN "GSTNO"
# and the tax "GST"; matching those spellings is terminology lookup, not
# guessing — the answer is still the line as printed.
_TERM_ALIASES = {
    "tin": ("gstno", "gst no", "taxid", "tax id", "vat no", "vatno"),
    "gst": ("vat", "sales tax"),
    "vat": ("gst",),
    "subtotal": ("sub total",),
    "total": ("grand total", "amount due", "balance due"),
    "po": ("purchase order",),
    "vendor": ("supplier", "company", "sdn bhd"),
}


def _score_line(line: str, terms: list[str]) -> float:
    """Score one OCR line against the question terms.

    An exact token match beats a substring match, because OCR merges words
    ("SIMPLIFIEDTAXINVOICE") — substring still recovers those, without
    letting a short number match unrelated digits.
    """
    low = line.lower()
    tokens = set(re.findall(r"[a-z0-9][a-z0-9.\-/]*", low))
    score = 0.0
    for t in terms:
        candidates = (t, *_TERM_ALIASES.get(t, ()))
        if any(c in tokens for c in candidates):
            score += 3.0
        elif any(len(c) >= 4 and c in low for c in candidates):
            score += 1.0
    return score


def _extractive_answer(question: str, retrieved: list[dict],
                       max_lines: int = 6) -> str:
    """Build a verbatim answer from the best-matching lines.

    Returning one whole chunk hides the answer inside a page of OCR text;
    returning the matching lines is still verbatim (nothing is invented or
    paraphrased) but actually answers the question. Falls back to the top
    chunk when no line matches any query term.
    """
    terms = _query_terms(question)
    if not terms:
        return retrieved[0]["chunk"]["text"]
    scored: list[tuple[float, int, str, str]] = []
    seen: set[str] = set()
    for order, r in enumerate(retrieved):
        chunk = r["chunk"]
        for raw in chunk["text"].splitlines():
            line = raw.strip()
            if not line or line.lower() in seen:
                continue
            sc = _score_line(line, terms)
            if sc <= 0:
                continue
            seen.add(line.lower())
            scored.append((sc, order, line, chunk["chunk_id"]))

    if not scored:
        return (f"No line in this invoice matches those terms "
                f"({', '.join(terms)}). Closest text, verbatim:\n\n"
                f"{retrieved[0]['chunk']['text']}")

    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = scored[:max_lines]
    picked.sort(key=lambda x: x[1])  # keep document order, not score order
    lines = []
    for _sc, _order, line, chunk_id in picked:
        snippet = line if len(line) <= 240 else line[:237] + "..."
        lines.append(f"- {snippet}  [{chunk_id}]")
    return "\n".join(lines)


def _citations(results: list[dict]) -> list[dict]:
    out = []
    for r in results:
        c = r["chunk"]
        out.append({
            "chunk_id": c["chunk_id"], "section": c["section"], "page": c["page"],
            "score": r.get("score"),
            "snippet": c["text"][:300],
        })
    return out


def ask_question(invoice_id: int, question: str, chunks: list[dict],
                 top_k: int = 4) -> dict[str, Any]:
    """Answer a question about one invoice, ONLY from its chunks.

    Returns {answer, citations, model, is_fallback, warning}.
    """
    # 1. Retrieve — Chroma if embeddings work, BM25 otherwise.
    retrieved: list[dict] = []
    used_bm25 = False
    try:
        vs = get_vectorstore(invoice_id)
        docs = vs.similarity_search_with_score(question, k=top_k)
        for doc, score in docs:
            retrieved.append({"chunk": {
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "section": doc.metadata.get("section", ""),
                "page": doc.metadata.get("page", 1),
                "text": doc.page_content,
            }, "score": round(float(score), 3)})
    except Exception:
        retrieved = _bm25(question, chunks, top_k=top_k)
        used_bm25 = True

    if not retrieved:
        return {"answer": "No relevant text was found in this invoice for that question.",
                "citations": [], "model": "extractive-fallback", "is_fallback": True,
                "warning": "No chunks retrieved — the invoice text may be empty or unreadable."}

    context = "\n\n".join(
        f"[{r['chunk']['chunk_id']} | {r['chunk']['section']} | page {r['chunk']['page']}]\n{r['chunk']['text']}"
        for r in retrieved
    )

    # 2. LLM answer (Gemini) — only from retrieved context.
    if settings.has_gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(model=settings.llm_model,
                                         google_api_key=settings.gemini_api_key,
                                         temperature=0)
            prompt = (
                "You are an invoice analyst. Answer the question using ONLY the "
                "invoice excerpts below. If the answer is not in the excerpts, say "
                "so plainly — do not guess or use outside knowledge. "
                "Cite the chunk id for each fact you use.\n\n"
                f"EXCERPTS:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
            )
            answer = llm.invoke(prompt).content
            return {"answer": str(answer), "citations": _citations(retrieved),
                    "model": settings.llm_model, "is_fallback": False,
                    "warning": "Retrieval used BM25 fallback (embeddings unavailable)." if used_bm25 else None}
        except Exception as exc:  # noqa: BLE001 — fall through to extractive
            warning = f"Gemini call failed ({exc}); using extractive fallback."

    else:
        warning = "No GEMINI_API_KEY set — extractive fallback answer (no LLM)."

    # 3. Extractive fallback — matching lines verbatim, clearly labelled.
    body = _extractive_answer(question, retrieved)
    answer = (
        "Based on the retrieved invoice text (extractive answer — no LLM was used):\n\n"
        f"{body}"
    )
    return {"answer": answer, "citations": _citations(retrieved),
            "model": "extractive-fallback", "is_fallback": True,
            "warning": warning if 'warning' in locals() else None}
