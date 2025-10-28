"""
Utilities for extracting clean sentences from PDF annual reports.

The processing flow mirrors the document scanning utilities used in
`bert-document-scan` so we maintain consistent cleaning, segmentation,
and filename handling across projects.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


def clean_extracted_text(text: str) -> str:
    """
    Sanitize common encoding artefacts produced during PDF extraction.
    """
    if not text:
        return text

    cleaned_text = text
    simple_replacements = {
        "We‚Äôre": "We're",
        "We‚Äôve": "We've",
        "‚Äôs": "'s",
        "Äô": "'",
        "Äì": "-",
        "≈ç": "fi",
        "Â": "",
        "‚Äô": "'",  # Apostrophes
        "‚Äì": "-",  # Dashes
        "‚Äù": "'",  # Quote variant
        "‚Äú": "'",  # Quote variant
    }

    for old_char, new_char in simple_replacements.items():
        if old_char in cleaned_text:
            cleaned_text = cleaned_text.replace(old_char, new_char)

    return cleaned_text


def text_to_sentences(text: str) -> str:
    """
    Split text into sentences while protecting common abbreviations.
    """
    if not text or not text.strip():
        return ""

    normalized = re.sub(r"\s+", " ", text).strip()
    protected_text = normalized

    abbreviations = [
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Ltd|Corp|Co|vs|etc|est|approx|govt)\.",
        r"\b(?:U\.S|U\.K|U\.N|E\.U)\.",
        r"\b(?:e\.g|i\.e|cf|viz)\.",
        r"\b[A-Z]\.",
        r"\$\d+\.\d+",
        r"\b\d+\.\d+",
    ]

    for pattern in abbreviations:
        protected_text = re.sub(
            pattern, lambda match: match.group(0).replace(".", "<DOT>"), protected_text, flags=re.IGNORECASE
        )

    sentence_boundaries = re.split(r"(?<=[.!?])\s+(?=[A-Z]|$)", protected_text)
    valid_sentences: List[str] = []
    disqualifying_endings = (
        "with",
        "and",
        "or",
        "but",
        "during",
        "since",
        "until",
        "while",
        "the",
        "a",
        "an",
        "this",
        "that",
        "which",
        "we",
        "they",
    )

    for sentence in sentence_boundaries:
        if not sentence:
            continue

        restored = sentence.replace("<DOT>", ".").strip()
        if (
            len(restored) >= 10
            and re.search(r"[a-zA-Z]", restored)
            and not restored.lower().rstrip(".!?").endswith(disqualifying_endings)
        ):
            valid_sentences.append(restored)

    return "\n".join(valid_sentences)


def fetch_pdf(pdf_source: str) -> str:
    """
    Placeholder hook for retrieving local PDF paths (mirrors reference logic).
    """
    return pdf_source


def extract_sentences_with_pages(pdf_path: str, min_len: int = 30, max_len: int = 600) -> List[dict[str, str | int]]:
    """
    Extract cleaned sentences from a PDF, keeping track of the source page.
    """
    doc = fitz.open(pdf_path)
    sentences: List[dict[str, str | int]] = []

    try:
        for page_index in range(len(doc)):
            text = doc[page_index].get_text("text")
            text = clean_extracted_text(text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\s+\n", "\n", text)
            text = re.sub(r"\n{2,}", "\n", text)

            for sentence in text_to_sentences(text).split("\n"):
                cleaned_sentence = clean_extracted_text(sentence.strip())
                if min_len <= len(cleaned_sentence) <= max_len:
                    sentences.append({"page": page_index + 1, "sentence": cleaned_sentence})
    finally:
        doc.close()

    return sentences


def safe_report_name(path_or_url: str) -> str:
    """
    Generate a standardized report identifier derived from the PDF path.
    """
    name = Path(path_or_url).stem
    match = re.search(r"[A-Za-z]{2,}\d{4}", name)
    if match:
        return match.group(0).upper()

    clean_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return clean_name or "REPORT"


def extract_company_code(pdf_files: List[str]) -> str:
    """
    Infer a company code from a collection of PDF filenames.
    """
    if not pdf_files:
        return "UNKNOWN"

    company_codes = set()
    for pdf_file in pdf_files:
        stem = Path(pdf_file).stem
        match = re.search(r"([A-Za-z]{2,})(\d{4})", stem)
        if match:
            company_codes.add(match.group(1).upper())
            continue

        fallback = re.search(r"^([A-Za-z]+)", stem)
        if fallback and len(fallback.group(1)) >= 2:
            company_codes.add(fallback.group(1).upper())

    if len(company_codes) == 1:
        return next(iter(company_codes))
    if len(company_codes) > 1:
        return "MIXED"
    return "UNKNOWN"


def get_pdf_files(directory: str) -> List[str]:
    """
    Return sorted PDF file paths discovered in the given directory.
    """
    return sorted(str(path) for path in Path(directory).glob("*.pdf") if path.is_file())
