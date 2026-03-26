"""
sec_interpreter/structure.py

Structure scan -- maps document heading structure to extraction targets
without requiring an LLM locator pass.

Two public functions:
  structure_scan(artifact_dir)  -> StructureScanResult
  gap_check(extraction_output, scan_result, logger) -> dict
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional

from .schemas import ObligationSection, StructureScanResult
from .tools import extract_references_from_text

logger = logging.getLogger("sec_interpreter")

# Regex that matches lettered section headings like "A.", "B.", "C."
_LETTER_RE = re.compile(r"^[A-Z]\.")

# Regex that matches roman numeral or numeric section headings like "III.", "IV.", "2."
_ROMAN_OR_NUM_RE = re.compile(r"^(?:[IVX]+|[0-9]+)\.")

# Named section keywords (case-insensitive, matched against last heading element)
_NAMED_KEYWORDS = ["effective", "compliance date", "applicability", "exemption", "codified text"]


def _heading_path_starts_with(path: List[str], prefix: List[str]) -> bool:
    """Return True if path starts with every element in prefix (exact match per element)."""
    if len(path) < len(prefix):
        return False
    return path[: len(prefix)] == prefix


def structure_scan(artifact_dir: str) -> StructureScanResult:
    """
    Scan ingest artifacts for document structure and build a StructureScanResult.

    Reads sections.json and chunks.json from artifact_dir.
    Identifies lettered obligation sections under "Discussion of Final Amendments",
    collects structured chunk IDs, named section chunk IDs, and CFR citations.

    Saves structure_scan_result.json to artifact_dir and returns StructureScanResult.
    """
    run_id = os.path.basename(artifact_dir.rstrip("/\\"))

    sections_path = os.path.join(artifact_dir, "sections.json")
    chunks_path = os.path.join(artifact_dir, "chunks.json")

    # Graceful degradation: return empty result if artifacts missing
    if not os.path.exists(sections_path):
        logger.warning("structure_scan: sections.json not found in %s", artifact_dir)
        return StructureScanResult(run_id=run_id)

    with open(sections_path, encoding="utf-8") as f:
        sections = json.load(f)

    chunks_by_section: dict = {}
    if os.path.exists(chunks_path):
        with open(chunks_path, encoding="utf-8") as f:
            raw_chunks = json.load(f)
        for chunk in raw_chunks:
            sid = chunk.get("section_id", "")
            if sid not in chunks_by_section:
                chunks_by_section[sid] = []
            chunks_by_section[sid].append(chunk)
    else:
        logger.warning("structure_scan: chunks.json not found in %s", artifact_dir)
        raw_chunks = []

    # ------------------------------------------------------------------
    # Step 1: Find the full prefix path up to (and including) the
    #         "Discussion of Final Amendments" element.
    # ------------------------------------------------------------------
    # The segmenter may split heading levels differently depending on the doc:
    #   Real doc:  ['II. Discussion of Final Amendments', 'A. Disclosure...', '3. Final']
    #   Test data: ['II.', 'Discussion of Final Amendments', 'A. Disclosure...']
    # We locate whichever path element contains the phrase, then record the
    # full prefix up to that element as the discussion_prefix.
    discussion_prefix: List[str] = []

    for section in sections:
        path = section.get("heading_path", [])
        for idx, element in enumerate(path):
            if "discussion of final amendments" in element.lower():
                discussion_prefix = path[: idx + 1]
                break
        if discussion_prefix:
            break

    if not discussion_prefix:
        logger.warning(
            "structure_scan: no 'Discussion of Final Amendments' section found in %s",
            artifact_dir,
        )
        return StructureScanResult(run_id=run_id)

    letter_depth = len(discussion_prefix)  # index where lettered elements live

    # ------------------------------------------------------------------
    # Step 2: Collect unique lettered sections one level below the prefix.
    # ------------------------------------------------------------------
    # Any section whose path starts with discussion_prefix and has a lettered
    # element at letter_depth is part of an obligation section.
    obligation_sections: List[ObligationSection] = []
    seen_letters: dict = {}  # letter -> index into obligation_sections

    for section in sections:
        heading_path = section.get("heading_path", [])
        if len(heading_path) <= letter_depth:
            continue
        if heading_path[:letter_depth] != discussion_prefix:
            continue
        lettered_heading = heading_path[letter_depth]
        if not _LETTER_RE.match(lettered_heading):
            continue

        letter = lettered_heading[0]

        if letter not in seen_letters:
            section_id = section.get("section_id", "")
            full_heading = " > ".join(discussion_prefix + [lettered_heading])
            obl_section = ObligationSection(
                section_letter=letter,
                heading=full_heading,
                section_id=section_id,
                cfr_citations=[],
                structured_chunk_ids=[],
            )
            obligation_sections.append(obl_section)
            seen_letters[letter] = len(obligation_sections) - 1

        idx = seen_letters[letter]
        seen_chunk_ids: set = set(obligation_sections[idx].structured_chunk_ids)

        # Collect final + codified chunks from this subsection
        s_id = section.get("section_id", "")
        for chunk in chunks_by_section.get(s_id, []):
            role = chunk.get("subsection_role", "")
            codified = chunk.get("has_codified_text", False)
            if role == "final" or codified:
                src_id = chunk.get("src_id", "")
                if src_id and src_id not in seen_chunk_ids:
                    obligation_sections[idx].structured_chunk_ids.append(src_id)
                    seen_chunk_ids.add(src_id)

        # Accumulate CFR citations from section text
        section_text = section.get("section_text", "")
        new_cites = extract_references_from_text(section_text)
        existing = set(obligation_sections[idx].cfr_citations)
        for cite in new_cites:
            if cite not in existing:
                obligation_sections[idx].cfr_citations.append(cite)
                existing.add(cite)

        logger.debug(
            "structure_scan: section %s subsection %s -- %d chunks so far",
            letter, heading_path[-1] if heading_path else "?",
            len(obligation_sections[idx].structured_chunk_ids),
        )

    for obl_section in obligation_sections:
        logger.debug(
            "structure_scan: obligation section %s -- %s (%d chunks, %d CFR cites)",
            obl_section.section_letter, obl_section.heading,
            len(obl_section.structured_chunk_ids), len(obl_section.cfr_citations),
        )

    # ------------------------------------------------------------------
    # Step 4: Find named sections by keyword
    # ------------------------------------------------------------------
    named_section_chunk_ids: List[str] = []
    named_seen: set = set()

    for section in sections:
        heading_path = section.get("heading_path", [])
        last_elem = heading_path[-1].lower() if heading_path else ""
        matched = any(kw in last_elem for kw in _NAMED_KEYWORDS)
        if not matched:
            continue
        s_id = section.get("section_id", "")
        for chunk in chunks_by_section.get(s_id, []):
            src_id = chunk.get("src_id", "")
            if src_id and src_id not in named_seen:
                named_section_chunk_ids.append(src_id)
                named_seen.add(src_id)

    # ------------------------------------------------------------------
    # Step 5: Build aggregate structured_chunk_ids (deduplicated union)
    # ------------------------------------------------------------------
    all_structured: List[str] = []
    all_seen: set = set()
    for obl in obligation_sections:
        for src_id in obl.structured_chunk_ids:
            if src_id not in all_seen:
                all_structured.append(src_id)
                all_seen.add(src_id)

    result = StructureScanResult(
        run_id=run_id,
        obligation_sections=obligation_sections,
        named_section_chunk_ids=named_section_chunk_ids,
        expected_obligation_count=len(obligation_sections),
        structured_chunk_ids=all_structured,
    )

    # ------------------------------------------------------------------
    # Step 6: Save artifact
    # ------------------------------------------------------------------
    out_path = os.path.join(artifact_dir, "structure_scan_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)
    logger.info(
        "structure_scan: saved %s  (%d obligation sections)",
        out_path, len(obligation_sections),
    )

    return result


def gap_check(
    extraction_output: dict,
    scan_result: StructureScanResult,
    logger: logging.Logger,
) -> dict:
    """
    Compare extraction output against structure scan to detect gaps.

    Returns a gap_report dict with:
      expected_count, extracted_count, count_gap, flagged_sections
    """
    obligations = extraction_output.get("key_obligations", [])
    extracted_count = len(obligations)
    expected_count = scan_result.expected_obligation_count
    count_gap = max(0, expected_count - extracted_count)

    # Collect all CFR citations from extracted obligations
    all_extracted_cites: set = set()
    for obl in obligations:
        for cite in obl.get("cited_sections", []):
            all_extracted_cites.add(cite)

    # Flag obligation sections whose CFR citations were not matched
    flagged_sections: List[dict] = []
    for obl_section in scan_result.obligation_sections:
        if not obl_section.cfr_citations:
            # No CFR citations to check -- skip
            continue
        matched = any(cite in all_extracted_cites for cite in obl_section.cfr_citations)
        if not matched:
            flagged_sections.append({
                "section_letter": obl_section.section_letter,
                "heading": obl_section.heading,
                "cfr_citations": obl_section.cfr_citations,
                "reason": "no CFR citations matched",
            })
            logger.warning(
                "gap_check: section %s (%s) has unmatched CFR citations: %s",
                obl_section.section_letter,
                obl_section.heading,
                obl_section.cfr_citations,
            )

    if count_gap > 0:
        logger.warning(
            "gap_check: expected %d obligations, extracted %d (gap=%d)",
            expected_count, extracted_count, count_gap,
        )

    gap_report = {
        "expected_count": expected_count,
        "extracted_count": extracted_count,
        "count_gap": count_gap,
        "flagged_sections": flagged_sections,
    }
    return gap_report
