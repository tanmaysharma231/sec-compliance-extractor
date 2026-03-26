"""
tests/test_structure.py

Tests for sec_interpreter/structure.py:
  - structure_scan()
  - gap_check()
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import List

import pytest

from sec_interpreter.structure import gap_check, structure_scan
from sec_interpreter.schemas import ObligationSection, StructureScanResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_logger = logging.getLogger("sec_interpreter")


def _write_artifacts(tmp_dir: str, sections: list, chunks: list) -> None:
    """Write sections.json and chunks.json to tmp_dir."""
    with open(os.path.join(tmp_dir, "sections.json"), "w", encoding="utf-8") as f:
        json.dump(sections, f)
    with open(os.path.join(tmp_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)


def _make_section(
    section_id: str,
    heading_path: List[str],
    section_text: str = "",
) -> dict:
    return {
        "section_id": section_id,
        "heading_path": heading_path,
        "section_text": section_text,
    }


def _make_chunk(
    src_id: str,
    section_id: str,
    subsection_role: str = "other",
    has_codified_text: bool = False,
) -> dict:
    return {
        "src_id": src_id,
        "section_id": section_id,
        "subsection_role": subsection_role,
        "has_codified_text": has_codified_text,
    }


# ---------------------------------------------------------------------------
# Shared fixture: 3 lettered sections (A, B, C) under Discussion parent
# ---------------------------------------------------------------------------

def _make_standard_sections() -> list:
    return [
        _make_section("SEC-001", ["I.", "Background"]),
        _make_section("SEC-002", ["II.", "Discussion of Final Amendments"]),
        _make_section("SEC-003", ["II.", "Discussion of Final Amendments", "A. Disclosure on Form 8-K"]),
        _make_section("SEC-004", ["II.", "Discussion of Final Amendments", "B. Periodic Reporting"]),
        _make_section("SEC-005", ["II.", "Discussion of Final Amendments", "C. Annual Report Governance"]),
        _make_section("SEC-006", ["III.", "Effective Date"]),
    ]


# ---------------------------------------------------------------------------
# structure_scan tests
# ---------------------------------------------------------------------------

class TestStructureScan:

    def test_finds_lettered_sections(self) -> None:
        """Given A, B, C children under Discussion, scan finds 3 obligation sections."""
        sections = _make_standard_sections()
        chunks: list = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_artifacts(tmp_dir, sections, chunks)
            result = structure_scan(tmp_dir)

        assert len(result.obligation_sections) == 3
        letters = [obl.section_letter for obl in result.obligation_sections]
        assert letters == ["A", "B", "C"]

    def test_collects_final_chunks(self) -> None:
        """Chunks with subsection_role='final' appear in structured_chunk_ids; 'proposed' do not."""
        sections = _make_standard_sections()
        chunks = [
            _make_chunk("src:0", "SEC-003", subsection_role="final"),
            _make_chunk("src:1", "SEC-003", subsection_role="proposed"),
            _make_chunk("src:2", "SEC-004", subsection_role="final"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_artifacts(tmp_dir, sections, chunks)
            result = structure_scan(tmp_dir)

        all_ids = result.structured_chunk_ids
        assert "src:0" in all_ids
        assert "src:2" in all_ids
        assert "src:1" not in all_ids

    def test_collects_codified_chunks(self) -> None:
        """Chunks with has_codified_text=True appear in structured_chunk_ids regardless of role."""
        sections = _make_standard_sections()
        chunks = [
            _make_chunk("src:10", "SEC-003", subsection_role="other", has_codified_text=True),
            _make_chunk("src:11", "SEC-004", subsection_role="other", has_codified_text=False),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_artifacts(tmp_dir, sections, chunks)
            result = structure_scan(tmp_dir)

        assert "src:10" in result.structured_chunk_ids
        assert "src:11" not in result.structured_chunk_ids

    def test_extracts_cfr_citations(self) -> None:
        """section_text containing a CFR citation is parsed into cfr_citations."""
        sections = [
            _make_section("SEC-001", ["II.", "Discussion of Final Amendments"]),
            _make_section(
                "SEC-002",
                ["II.", "Discussion of Final Amendments", "A. Disclosure"],
                section_text="Registrants must comply with 17 CFR 229.106(b) and related rules.",
            ),
        ]
        chunks: list = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_artifacts(tmp_dir, sections, chunks)
            result = structure_scan(tmp_dir)

        assert len(result.obligation_sections) == 1
        cites = result.obligation_sections[0].cfr_citations
        assert any("229.106" in c for c in cites)

    def test_finds_named_sections(self) -> None:
        """A section whose last heading element contains 'effective' collects its chunks."""
        sections = [
            _make_section("SEC-001", ["II.", "Discussion of Final Amendments"]),
            _make_section("SEC-002", ["II.", "Discussion of Final Amendments", "A. Disclosure"]),
            _make_section("SEC-003", ["III.", "Effective Date and Compliance"]),
        ]
        chunks = [
            _make_chunk("src:20", "SEC-003", subsection_role="other"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_artifacts(tmp_dir, sections, chunks)
            result = structure_scan(tmp_dir)

        assert "src:20" in result.named_section_chunk_ids

    def test_handles_missing_discussion_section(self) -> None:
        """If no 'Discussion of Final Amendments' section exists, returns empty result gracefully."""
        sections = [
            _make_section("SEC-001", ["I.", "Background"]),
            _make_section("SEC-002", ["II.", "Final Rules"]),
        ]
        chunks: list = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_artifacts(tmp_dir, sections, chunks)
            result = structure_scan(tmp_dir)

        assert result.obligation_sections == []
        assert result.expected_obligation_count == 0
        assert result.structured_chunk_ids == []

    def test_expected_obligation_count(self) -> None:
        """expected_obligation_count equals the number of lettered sections found."""
        sections = _make_standard_sections()
        chunks: list = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_artifacts(tmp_dir, sections, chunks)
            result = structure_scan(tmp_dir)

        assert result.expected_obligation_count == len(result.obligation_sections)
        assert result.expected_obligation_count == 3

    def test_saves_artifact(self) -> None:
        """structure_scan writes structure_scan_result.json to artifact_dir."""
        sections = _make_standard_sections()
        chunks: list = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_artifacts(tmp_dir, sections, chunks)
            structure_scan(tmp_dir)
            assert os.path.exists(os.path.join(tmp_dir, "structure_scan_result.json"))

    def test_run_id_from_dirname(self) -> None:
        """run_id is taken from the basename of artifact_dir."""
        sections = _make_standard_sections()
        chunks: list = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_artifacts(tmp_dir, sections, chunks)
            result = structure_scan(tmp_dir)
            assert result.run_id == os.path.basename(tmp_dir)

    def test_structured_chunk_ids_deduplicated(self) -> None:
        """Duplicate src_ids are not repeated in structured_chunk_ids."""
        sections = _make_standard_sections()
        # Same src_id appears in two sections that are both descendants of A.
        chunks = [
            _make_chunk("src:99", "SEC-003", subsection_role="final"),
            _make_chunk("src:99", "SEC-003", subsection_role="final"),  # exact duplicate
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_artifacts(tmp_dir, sections, chunks)
            result = structure_scan(tmp_dir)

        assert result.structured_chunk_ids.count("src:99") == 1

    def test_stops_at_non_lettered_sibling(self) -> None:
        """Sections under 'III.' (sibling of Discussion parent) are not included."""
        sections = [
            _make_section("SEC-001", ["II.", "Discussion of Final Amendments"]),
            _make_section("SEC-002", ["II.", "Discussion of Final Amendments", "A. Disclosure"]),
            _make_section("SEC-003", ["III.", "Economic Analysis"]),
            # This section is under III., not under Discussion -- should not be obligation
            _make_section("SEC-004", ["III.", "Economic Analysis", "A. Costs"]),
        ]
        chunks: list = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_artifacts(tmp_dir, sections, chunks)
            result = structure_scan(tmp_dir)

        # Only the A under Discussion should be found
        assert len(result.obligation_sections) == 1
        assert result.obligation_sections[0].section_letter == "A"


# ---------------------------------------------------------------------------
# gap_check tests
# ---------------------------------------------------------------------------

class TestGapCheck:

    def _make_scan_result(
        self,
        obligation_sections: List[ObligationSection],
    ) -> StructureScanResult:
        return StructureScanResult(
            run_id="test-run",
            obligation_sections=obligation_sections,
            expected_obligation_count=len(obligation_sections),
        )

    def _make_extraction_output(self, obligations: list) -> dict:
        return {"key_obligations": obligations}

    def _make_obligation(self, obl_id: str, cited_sections: List[str]) -> dict:
        return {
            "obligation_id": obl_id,
            "obligation_text": "Some obligation text.",
            "cited_sections": cited_sections,
        }

    def test_no_gaps_when_counts_match_and_cites_covered(self) -> None:
        """No flagged sections and count_gap=0 when all citations matched."""
        obl_sections = [
            ObligationSection(
                section_letter="A",
                heading="II. Discussion of Final Amendments A. Disclosure",
                section_id="SEC-003",
                cfr_citations=["17 CFR 229.106(b)"],
            ),
        ]
        scan_result = self._make_scan_result(obl_sections)
        extraction_output = self._make_extraction_output([
            self._make_obligation("OBL-001", cited_sections=["17 CFR 229.106(b)"]),
        ])

        report = gap_check(extraction_output, scan_result, _logger)

        assert report["count_gap"] == 0
        assert report["flagged_sections"] == []
        assert report["extracted_count"] == 1
        assert report["expected_count"] == 1

    def test_flags_count_gap(self) -> None:
        """count_gap > 0 when fewer obligations extracted than expected."""
        obl_sections = [
            ObligationSection(
                section_letter="A",
                heading="A. Disclosure",
                section_id="SEC-003",
                cfr_citations=[],
            ),
            ObligationSection(
                section_letter="B",
                heading="B. Periodic Reporting",
                section_id="SEC-004",
                cfr_citations=[],
            ),
            ObligationSection(
                section_letter="C",
                heading="C. Annual Governance",
                section_id="SEC-005",
                cfr_citations=[],
            ),
        ]
        scan_result = self._make_scan_result(obl_sections)
        extraction_output = self._make_extraction_output([
            self._make_obligation("OBL-001", cited_sections=[]),
        ])

        report = gap_check(extraction_output, scan_result, _logger)

        assert report["count_gap"] == 2
        assert report["expected_count"] == 3
        assert report["extracted_count"] == 1

    def test_flags_section_when_cites_missing(self) -> None:
        """A section with cfr_citations not covered by extracted obligations appears in flagged_sections."""
        obl_sections = [
            ObligationSection(
                section_letter="A",
                heading="A. Disclosure",
                section_id="SEC-003",
                cfr_citations=["17 CFR 229.106(b)"],
            ),
        ]
        scan_result = self._make_scan_result(obl_sections)
        extraction_output = self._make_extraction_output([
            # This obligation cites a different section -- no match
            self._make_obligation("OBL-001", cited_sections=["17 CFR 240.13a-11"]),
        ])

        report = gap_check(extraction_output, scan_result, _logger)

        assert len(report["flagged_sections"]) == 1
        flagged = report["flagged_sections"][0]
        assert flagged["section_letter"] == "A"
        assert "17 CFR 229.106(b)" in flagged["cfr_citations"]
        assert flagged["reason"] == "no CFR citations matched"

    def test_no_flag_when_section_has_no_citations(self) -> None:
        """A section with empty cfr_citations is not flagged even if nothing extracted."""
        obl_sections = [
            ObligationSection(
                section_letter="B",
                heading="B. Periodic Reporting",
                section_id="SEC-004",
                cfr_citations=[],  # no CFR cites -- should not be flagged
            ),
        ]
        scan_result = self._make_scan_result(obl_sections)
        extraction_output = self._make_extraction_output([])

        report = gap_check(extraction_output, scan_result, _logger)

        assert report["flagged_sections"] == []

    def test_gap_report_keys_present(self) -> None:
        """gap_report always contains all four expected keys."""
        scan_result = self._make_scan_result([])
        extraction_output = self._make_extraction_output([])

        report = gap_check(extraction_output, scan_result, _logger)

        for key in ("expected_count", "extracted_count", "count_gap", "flagged_sections"):
            assert key in report

    def test_count_gap_never_negative(self) -> None:
        """count_gap is 0 even if more obligations are extracted than expected."""
        obl_sections = [
            ObligationSection(
                section_letter="A",
                heading="A. Disclosure",
                section_id="SEC-003",
                cfr_citations=[],
            ),
        ]
        scan_result = self._make_scan_result(obl_sections)
        extraction_output = self._make_extraction_output([
            self._make_obligation("OBL-001", cited_sections=[]),
            self._make_obligation("OBL-002", cited_sections=[]),
            self._make_obligation("OBL-003", cited_sections=[]),
        ])

        report = gap_check(extraction_output, scan_result, _logger)

        assert report["count_gap"] == 0
        assert report["extracted_count"] == 3
        assert report["expected_count"] == 1
