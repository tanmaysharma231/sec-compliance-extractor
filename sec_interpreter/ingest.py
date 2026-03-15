from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional, Tuple

import requests

from .utils import get_logger

_logger = get_logger("sec_interpreter.ingest")

_SEC_USER_AGENT = "ComplianceResearch research@example.com"
_REQUEST_TIMEOUT = 30  # seconds

# Lines that are just a page number (possibly with surrounding whitespace)
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")

# Inline footnote markers: a number that appears alone after punctuation/space,
# e.g. "some text.6 \n" → "some text. "
_FOOTNOTE_INLINE_RE = re.compile(r"(?<=[a-z\.\,\;\:\)\"])\d{1,3}(?=\s)")

# PDF encoding artifacts: replacement char + curly quotes + other common mojibake
_PDF_QUOTE_MAP = str.maketrans({
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u2013": "-",  # en dash
    "\u2014": "--", # em dash
    "\u2022": "*",  # bullet
    "\ufffd": "'",  # replacement char fallback
})
_REPLACEMENT_CHAR_RE = re.compile(r"\ufffd+")

# Collapse runs of blank lines to a single blank line
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Table-of-contents lines: contain 4+ consecutive dots (leader dots)
_TOC_LINE_RE = re.compile(r"\.{4,}")


def fetch_rule_text(
    source: str,
    page_range: Optional[Tuple[int, int]] = None,
) -> str:
    """Fetch and return clean plain text from a SEC regulatory document.

    Args:
        source: A URL (http/https) or local file path (.pdf or .txt).
        page_range: Optional (start, end) 1-based page numbers for PDFs,
                    e.g. (1, 50) extracts pages 1–50 inclusive.
                    If None, all pages are extracted.

    Returns:
        Clean plain text suitable for use as RuleExtractorInput.rule_text.
    """
    source = source.strip()

    if _is_url(source):
        return _fetch_url(source, page_range)
    else:
        return _read_local(source, page_range)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _fetch_url(url: str, page_range: Optional[Tuple[int, int]]) -> str:
    _logger.info("Fetching URL: %s", url)
    response = requests.get(
        url,
        timeout=_REQUEST_TIMEOUT,
        headers={"User-Agent": _SEC_USER_AGENT},
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        _logger.info("Detected PDF (%d bytes)", len(response.content))
        return _extract_pdf(io.BytesIO(response.content), page_range)
    else:
        _logger.info("Detected HTML (%d bytes)", len(response.content))
        return _extract_html(response.content, response.encoding or "utf-8")


def _read_local(path_str: str, page_range: Optional[Tuple[int, int]]) -> str:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        _logger.info("Reading local PDF: %s", path)
        return _extract_pdf(io.BytesIO(path.read_bytes()), page_range)
    else:
        _logger.info("Reading local text file: %s", path)
        return path.read_text(encoding="utf-8-sig")


def _extract_pdf(stream: io.BytesIO, page_range: Optional[Tuple[int, int]]) -> str:
    from pypdf import PdfReader

    reader = PdfReader(stream)
    total_pages = len(reader.pages)
    _logger.info("PDF has %d pages", total_pages)

    if page_range is not None:
        start, end = page_range
        start_idx = max(0, start - 1)
        end_idx = min(total_pages, end)
    else:
        start_idx, end_idx = 0, total_pages

    _logger.info("Extracting pages %d–%d", start_idx + 1, end_idx)

    page_texts: list[str] = []
    for i in range(start_idx, end_idx):
        raw = reader.pages[i].extract_text() or ""
        cleaned = _clean_pdf_page(raw)
        if cleaned.strip():
            page_texts.append(cleaned)

    combined = "\n\n".join(page_texts)
    return _post_clean(combined)


def _clean_pdf_page(page_text: str) -> str:
    """Clean a single PDF page: strip page numbers, footnote markers, encoding artifacts."""
    lines = page_text.split("\n")
    kept: list[str] = []

    for line in lines:
        # Drop bare page-number lines
        if _PAGE_NUMBER_RE.match(line):
            continue
        # Drop table-of-contents leader lines (rows of dots)
        if _TOC_LINE_RE.search(line):
            continue
        # Drop "Conformed to Federal Register version" boilerplate
        if "conformed to federal register" in line.lower():
            continue
        # Remove inline footnote markers (e.g. "text.6 " → "text. ")
        line = _FOOTNOTE_INLINE_RE.sub("", line)
        # Normalize PDF encoding artifacts (curly quotes, dashes, replacement chars)
        line = line.translate(_PDF_QUOTE_MAP)
        kept.append(line)

    return "\n".join(kept)


def _extract_html(content: bytes, encoding: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "html.parser")

    # Remove nav, header, footer, script, style elements
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "noscript"]):
        tag.decompose()

    # Try to find a main content container
    main = (
        soup.find("main")
        or soup.find("div", class_=re.compile(r"main.content|article|body.text", re.I))
        or soup.find("body")
    )

    if main is None:
        main = soup

    # Extract paragraphs and headings as plain text blocks
    blocks: list[str] = []
    for tag in main.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            blocks.append(text)

    combined = "\n\n".join(blocks)
    return _post_clean(combined)


def _post_clean(text: str) -> str:
    """Final cleanup pass: normalize whitespace and blank lines."""
    # Collapse multiple spaces/tabs to single space (within lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Trim trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse 3+ blank lines to 2
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()
