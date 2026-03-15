"""
sec_interpreter/segmenter.py

Purpose: Heading-aware segmentation of SEC Federal Register documents into a flat
         list of Section objects, each carrying a heading_path for hierarchy context.

Fits in flow: IngestGraph → fetch_document → [segment_document] → chunk_sections → …

Called by: ingest_graph.segment_document node
Calls:     schemas.Section
"""
from __future__ import annotations

import re
from typing import List

from .schemas import Section

KNOWN_ANCHORS = {
    "SUMMARY",
    "DATES",
    "EFFECTIVE DATE",
    "COMPLIANCE DATE",
    "APPLICABILITY",
    "SUPPLEMENTARY INFORMATION",
    "BACKGROUND",
    "DISCUSSION",
    "AMENDMENTS",
    "DEFINITIONS",
}

# Numbered heading prefix: "I.", "II.", "A.", "B.", "1.", "1.05"
# Leading integer capped at 2 digits so footnote refs like "478." and zip codes like
# "20549." are excluded. Decimal sub-segments allow up to 3 digits (e.g. "1.100.").
_NUMBERED_RE = re.compile(
    r"^(?:[IVX]+|[A-Z]|[0-9]{1,2}(?:\.[0-9]{1,3})*)\.(?:\s|$)",
    re.IGNORECASE,
)


def _heading_level(label: str) -> int:
    """Assign a numeric depth level to a heading label."""
    stripped = label.strip()
    # Roman numeral prefix: I., II., III. → top level 0
    if re.match(r"^[IVX]+\.(\s|$)", stripped, re.IGNORECASE):
        return 0
    # Single uppercase letter prefix: A., B. → level 1
    if re.match(r"^[A-Z]\.(\s|$)", stripped):
        return 1
    # Lowercase letter prefix: a., b. → level 3
    if re.match(r"^[a-z]\.(\s|$)", stripped):
        return 3
    # Decimal / integer prefix: 1., 1.1., 1.05. → level 2
    if re.match(r"^[0-9]+(?:\.[0-9]+)*\.(\s|$)", stripped):
        return 2
    # ALL-CAPS / known anchor → top level
    return 0


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False

    # Known anchor (case-insensitive exact match)
    if stripped.upper() in KNOWN_ANCHORS:
        return True

    # Numbered section prefix — require at least 2 consecutive letters somewhere in the
    # line so bare references like "229.103.", "1.05.", "745." are excluded
    if _NUMBERED_RE.match(stripped) and re.search(r"[A-Za-z]{2,}", stripped):
        return True

    # ALL-CAPS short line not ending in sentence-terminal punctuation.
    # Exclude: parentheticals "(17 CFR)", lines ending in digit/comma (journal citations
    # like "ACCT. & PUB. POLICY 509, 509-519"), and lines with no real word letters.
    last_char = stripped[-1]
    if (
        stripped == stripped.upper()
        and last_char not in ".!?;,"
        and not last_char.isdigit()
        and not stripped.startswith("(")
        and re.search(r"[A-Z]{2,}", stripped)   # at least 2 consecutive uppercase letters
    ):
        return True

    return False


_MAX_LEVELS = 4


def _build_section_id(level_counters: List[int], level: int) -> str:
    """Build a fixed-width hierarchical section ID from level counters.

    Always 8 digits (2 per level x 4 levels). Unused levels are padded with 00.
    Read in pairs left-to-right to decode position at each depth.
    Examples:
      L0 (2nd top-level)         -> "02000000"
      L1 (1st sub of 2nd top)    -> "02010000"
      L2 (3rd sub-sub)           -> "02010300"
      L3 (1st deepest)           -> "02010301"
    """
    return "".join(f"{level_counters[i]:02d}" for i in range(_MAX_LEVELS))


def segment_document(text: str) -> List[Section]:
    """Scan text line-by-line, detect headings, and build a flat list of Sections.

    Hierarchy is encoded in heading_path (e.g. ["I. Introduction", "A. Sub", "1. Detail"]).
    section_id encodes position at each level (e.g. "02-01-03").
    Text that precedes the first heading is placed in an UNLABELED section (id "00").
    Returns a non-empty list -- at minimum one UNLABELED section.
    """
    lines = text.splitlines()
    sections: List[Section] = []

    heading_stack: List[tuple[int, str]] = []  # (level, label)
    current_heading_path: List[str] = ["UNLABELED"]
    current_body_lines: List[str] = []
    level_counters: List[int] = [0] * _MAX_LEVELS
    current_section_id: str = "00000000"

    def flush() -> None:
        body = "\n".join(current_body_lines).strip()
        if not body:
            return
        sections.append(
            Section(
                section_id=current_section_id,
                heading_path=list(current_heading_path),
                level=len(current_heading_path) - 1,
                section_text=body,
            )
        )

    for line in lines:
        if _is_heading(line):
            flush()
            current_body_lines = []

            label = line.strip()
            level = _heading_level(label)

            # Increment counter at this level and reset all deeper levels
            level_counters[level] += 1
            for deeper in range(level + 1, _MAX_LEVELS):
                level_counters[deeper] = 0

            current_section_id = _build_section_id(level_counters, level)

            # Pop stack to correct parent level and push new heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, label))
            current_heading_path = [h[1] for h in heading_stack]
        else:
            current_body_lines.append(line)

    flush()

    # Fallback: if nothing was parsed, put entire text into UNLABELED
    if not sections:
        sections.append(
            Section(
                section_id="00000000",
                heading_path=["UNLABELED"],
                level=0,
                section_text=text.strip(),
            )
        )

    return sections
