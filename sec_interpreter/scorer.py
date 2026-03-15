"""
sec_interpreter/scorer.py

Purpose: Pure-Python hot-zone flag scoring for RichChunk objects.
         Assigns boolean flags (has_dates, has_scope, has_obligations, has_definitions)
         based on regex pattern matching against chunk text.
         Also builds compact index rows for the Locator LLM prompt.

Fits in flow: IngestGraph → chunk_sections → [score_chunks] → extract_summary → …

Called by: ingest_graph.score_chunks node
Calls:     schemas.RichChunk
"""
from __future__ import annotations

import re

from .schemas import RichChunk

# ---------------------------------------------------------------------------
# Keyword patterns per hot-zone category
# ---------------------------------------------------------------------------

DATE_KEYWORDS = [
    r"effective date",
    r"compliance date",
    r"will become effective",
    r"applicable date",
    r"phase.in",
    r"\btransition\b",
    r"\d+ days after",
]

SCOPE_KEYWORDS = [
    r"applies to",
    r"applicability",
    r"\bscope\b",
    r"\bcovered\b",
    r"\bregistrant",
    r"\bissuer\b",
    r"broker.dealer",
    r"investment adviser",
    r"\bexempt\b",
    r"does not apply",
]

OBLIGATION_KEYWORDS = [
    r"\bmust\b",
    r"\bshall\b",
    r"is required",
    r"are required",
    r"we are adopting",
    r"we are amending",
    r"\bamend\b",
    r"§",
    r"\bCFR\b",
    r"\bItem\b",
    r"\bRule\b",
]

DEFINITION_KEYWORDS = [
    r"\bmeans\b",
    r"\bdefinition\b",
    r"for purposes of",
    r"as defined in",
]

EXAMPLE_KEYWORDS = [
    r"for example",
    r"for instance",
    r"in this case",
    r"if a company",
    r"if the registrant",
    r"\bday \d+\b",
    r"\bby day\b",
    r"\bsuch as\b",
    r"illustrat",
    r"consider a",
    r"hypothetical",
]

# Patterns that appear ONLY in legally operative CFR amendment text, never in commentary.
# Text patterns: matched against chunk body.
CODIFIED_TEXT_KEYWORDS = [
    r"\* \* \* \* \*",                  # Five-asterisk CFR ellipsis -- strongest signal
    r"to read as follows",              # Canonical Federal Register amendment phrase
    r"Authority:\s+15 U\.S\.C\.",       # Statutory authority blocks
    r"Pub\.\s*L\.\s*\d{3}-\d{3}",      # Public Law citations
    r"§\s*\d{3,}\.\d",                  # Specific CFR section numbers e.g. "§ 229.106"
    r"Instruction\s+\d+\s+to\s+Item",  # CFR instruction references
    r"of this chapter",                 # CFR cross-reference language
]

# Heading patterns: matched against heading_path entries.
CODIFIED_HEADING_KEYWORDS = [
    r"PART\s+22[0-9]",           # PART 229 (Regulation S-K)
    r"PART\s+23[0-9]",           # PART 232 (EDGAR), PART 239 (Securities Act forms)
    r"PART\s+24[0-9]",           # PART 240 (Exchange Act rules), PART 249 (Exchange Act forms)
    r"Add\s+§",                  # "Add § 229.106 to read as follows"
    r"Amend\s+§",                # "Amend § 232.405 by adding paragraph"
    r"Revise\s+Form",            # "Revise Form 20-F"
    r"continues to read",        # "authority citation ... continues to read as follows"
    r"GENERAL INSTRUCTIONS",
    r"INFORMATION TO BE INCLUDED IN THE REPORT",
    r"List of Subjects",         # Precedes every codified section block
]

_DATE_RE = re.compile("|".join(DATE_KEYWORDS), re.IGNORECASE)
_SCOPE_RE = re.compile("|".join(SCOPE_KEYWORDS), re.IGNORECASE)
_OBL_RE = re.compile("|".join(OBLIGATION_KEYWORDS), re.IGNORECASE)
_DEF_RE = re.compile("|".join(DEFINITION_KEYWORDS), re.IGNORECASE)
_CODIFIED_TEXT_RE = re.compile("|".join(CODIFIED_TEXT_KEYWORDS), re.IGNORECASE)
_CODIFIED_HEADING_RE = re.compile("|".join(CODIFIED_HEADING_KEYWORDS), re.IGNORECASE)
_EXAMPLE_RE = re.compile("|".join(EXAMPLE_KEYWORDS), re.IGNORECASE)


def score_chunk(chunk: RichChunk) -> RichChunk:
    """Return a copy of chunk with has_* boolean flags populated from text content."""
    text = chunk.text
    heading_text = " ".join(chunk.heading_path)
    has_codified = bool(_CODIFIED_TEXT_RE.search(text)) or bool(
        _CODIFIED_HEADING_RE.search(heading_text)
    )
    return chunk.model_copy(
        update={
            "has_dates": bool(_DATE_RE.search(text)),
            "has_scope": bool(_SCOPE_RE.search(text)),
            "has_obligations": bool(_OBL_RE.search(text)),
            "has_definitions": bool(_DEF_RE.search(text)),
            "has_codified_text": has_codified,
            "has_example": bool(_EXAMPLE_RE.search(text)),
        }
    )


def build_index_row(chunk: RichChunk) -> dict:
    """Return compact metadata row for Locator LLM prompt.

    Format: { src_id, heading, chars, flags, preview }
    Three independent signals so Locator works even when any one is weak.
    """
    flags = []
    if chunk.has_dates:
        flags.append("dates")
    if chunk.has_scope:
        flags.append("scope")
    if chunk.has_obligations:
        flags.append("obl")
    if chunk.has_definitions:
        flags.append("def")
    if chunk.has_codified_text:
        flags.append("codified")

    heading = " — ".join(chunk.heading_path) if chunk.heading_path else "UNLABELED"
    preview = chunk.text[:200].replace("\n", " ").strip()

    return {
        "src_id": chunk.src_id,
        "heading": heading,
        "chars": chunk.char_len,
        "flags": flags,
        "preview": preview,
    }
