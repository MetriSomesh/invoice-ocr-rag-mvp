"""Extractive-fallback answer tests — no LLM, no embeddings, no network.

The fallback must answer from the retrieved OCR lines and nothing else: it
may pick and order lines, but it can never invent a value.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import _extractive_answer, _query_terms, _score_line


def _r(chunk_id, section, text):
    return {"chunk": {"chunk_id": chunk_id, "section": section, "page": 1,
                      "text": text}, "score": 1.0}


RECEIPT = [
    _r("header-0", "header",
       "PERNIAGAAN ZHENG HUI\nSIMPLIFIED TAX INVOICE\n"
       "Receipt#CS00084670\nDate: 09/03/2018"),
    _r("line_items-1", "line_items",
       "Item Qty Price Amount\n"
       "12133 GA18012 7.50 15.00\n"
       "9555752615725 18011 25.00 26.00"),
    _r("totals-2", "totals",
       "Sub Total(RM): 313.50\nTotal GST(RM): 18.81\nTotal(RM): 332.30"),
]


def test_stopwords_are_dropped_from_query_terms():
    assert _query_terms("What is the total?") == ["total"]
    assert _query_terms("How much is the GST?") == ["gst"]


def test_exact_token_scores_higher_than_substring():
    terms = ["invoice"]
    exact = _score_line("simplified tax invoice", terms)
    merged = _score_line("SIMPLIFIEDTAXINVOICE", terms)
    assert exact > merged > 0


def test_answer_picks_matching_lines_not_the_whole_chunk():
    answer = _extractive_answer("What is the total?", RECEIPT)
    assert "Total(RM): 332.30" in answer
    assert "[totals-2]" in answer
    # Only matching lines are returned — not every line of every chunk.
    assert "Item Qty Price Amount" not in answer


def test_answer_can_span_two_chunks():
    # The invoice number lives in the header, the total in the totals block;
    # a two-part question must surface both, with their own citations.
    answer = _extractive_answer("What is the receipt number and the total?", RECEIPT)
    assert "Receipt#CS00084670" in answer
    assert "Total(RM): 332.30" in answer
    assert "[header-0]" in answer and "[totals-2]" in answer


def test_number_search_finds_the_item_row():
    answer = _extractive_answer("Which item cost 25.00?", RECEIPT)
    assert "25.00" in answer
    assert "[line_items-1]" in answer


def test_unmatched_question_says_so_instead_of_dumping_text():
    # Saying "no line matches" is honest; dumping a chunk looks like an
    # answer. The closest text is still shown, verbatim, for verification.
    answer = _extractive_answer("zzz qqq", RECEIPT)
    assert "No line in this invoice matches those terms" in answer
    assert "zzz" in answer
    assert RECEIPT[0]["chunk"]["text"] in answer


def test_never_invents_a_value():
    # Every number the fallback prints must already exist in the retrieved
    # text — a question about a field the document lacks must not conjure one.
    source = " ".join(r["chunk"]["text"] for r in RECEIPT)
    for q in ("What is the shipping cost?", "What is the total VAT rate?",
              "What is the receipt number and total?"):
        answer = _extractive_answer(q, RECEIPT)
        for num in re.findall(r"\d[\d,]*\.?\d*", answer):
            assert num.strip(".,") in source, f"{num} invented for {q!r}"
def test_tin_question_finds_the_gstno_line():
    # Malaysian receipts print the TIN as "GSTNO"; asking for the TIN must
    # find it without the app ever rewriting the printed label.
    receipt = [_r("header-0", "header", "PERNIAGAAN ZHENG HUI\nJM0325955-V"),
               _r("tax-1", "tax", "GSTNO:000800589824\nTotal GST(RM): 18.81")]
    answer = _extractive_answer("What is the vendor TIN?", receipt)
    assert "GSTNO:000800589824" in answer
    assert "[tax-1]" in answer


def test_gst_and_vat_are_treated_as_the_same_tax():
    receipt = [_r("tax-1", "tax", "VAT (7.5%): 93,750")]
    assert "93,750" in _extractive_answer("What is the GST?", receipt)
