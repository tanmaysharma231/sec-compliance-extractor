from __future__ import annotations

import json
import os
import tempfile

import pytest

from sec_interpreter.tools import (
    detect_ambiguous_terms,
    extract_references_from_text,
    get_section_family_chunks,
    get_surrounding_context,
    lookup_definition,
)


# ---------------------------------------------------------------------------
# detect_ambiguous_terms
# ---------------------------------------------------------------------------

def test_detects_known_terms() -> None:
    text = "The registrant must disclose any material cybersecurity incident promptly."
    found = detect_ambiguous_terms(text)
    assert "material" in found
    assert "promptly" in found


def test_detects_term_as_whole_word() -> None:
    # "materiality" contains "material" -- both should match as separate terms
    text = "The materiality determination is subject to reasonable judgment."
    found = detect_ambiguous_terms(text)
    assert "materiality" in found
    assert "reasonable" in found
    # "material" is a separate entry -- word-boundary match means it won't
    # match inside "materiality" so it should NOT appear
    assert "material" not in found


def test_returns_empty_for_clean_text() -> None:
    text = "The registrant shall file Form 8-K within four business days."
    found = detect_ambiguous_terms(text)
    assert found == []


def test_case_insensitive() -> None:
    text = "MATERIAL cybersecurity risks must be disclosed."
    found = detect_ambiguous_terms(text)
    assert "material" in found


# ---------------------------------------------------------------------------
# extract_references_from_text
# ---------------------------------------------------------------------------

def test_extracts_standard_cfr_citation() -> None:
    text = "As required by 17 CFR 229.106(b), registrants must disclose."
    refs = extract_references_from_text(text)
    assert len(refs) == 1
    assert "229.106" in refs[0]


def test_extracts_multiple_citations() -> None:
    text = (
        "See 17 CFR 229.106 for disclosure requirements and "
        "CFR 240.13a-11 for incident reporting obligations."
    )
    refs = extract_references_from_text(text)
    assert len(refs) == 2


def test_deduplicates_citations() -> None:
    text = "See 17 CFR 229.106 and 17 CFR 229.106 again."
    refs = extract_references_from_text(text)
    assert len(refs) == 1


def test_returns_empty_for_no_citations() -> None:
    text = "This section has no CFR references."
    refs = extract_references_from_text(text)
    assert refs == []


def test_handles_cfr_without_title_number() -> None:
    text = "Pursuant to CFR 229.106, the registrant must..."
    refs = extract_references_from_text(text)
    assert len(refs) == 1


# ---------------------------------------------------------------------------
# lookup_definition -- uses temp files, no LLM
# ---------------------------------------------------------------------------

def _write_artifact_files(
    tmp_dir: str,
    sections: list,
    classifications: list,
) -> None:
    with open(os.path.join(tmp_dir, "sections.json"), "w", encoding="utf-8") as f:
        json.dump(sections, f)
    with open(os.path.join(tmp_dir, "section_classifications.json"), "w", encoding="utf-8") as f:
        json.dump(classifications, f)


def test_lookup_definition_finds_term() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sections = [
            {
                "section_id": "SEC-001",
                "heading_path": ["Definitions"],
                "section_text": (
                    "The term 'material' means information that a reasonable investor "
                    "would consider important in making an investment decision."
                ),
            }
        ]
        classifications = [
            {"section_id": "SEC-001", "content_type": "definition"}
        ]
        _write_artifact_files(tmp, sections, classifications)

        result = lookup_definition("material", tmp)
        assert result is not None
        assert "reasonable investor" in result


def test_lookup_definition_ignores_non_definition_sections() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sections = [
            {
                "section_id": "SEC-001",
                "heading_path": ["Background"],
                "section_text": "The term material is used throughout this rule.",
            }
        ]
        classifications = [
            {"section_id": "SEC-001", "content_type": "commentary"}
        ]
        _write_artifact_files(tmp, sections, classifications)

        result = lookup_definition("material", tmp)
        assert result is None


def test_lookup_definition_returns_none_when_term_absent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sections = [
            {
                "section_id": "SEC-001",
                "heading_path": ["Definitions"],
                "section_text": "The term 'registrant' means any entity subject to Exchange Act reporting.",
            }
        ]
        classifications = [
            {"section_id": "SEC-001", "content_type": "definition"}
        ]
        _write_artifact_files(tmp, sections, classifications)

        result = lookup_definition("material", tmp)
        assert result is None


def test_lookup_definition_returns_none_when_files_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = lookup_definition("material", tmp)
        assert result is None


# ---------------------------------------------------------------------------
# get_surrounding_context -- uses temp files, no LLM
# ---------------------------------------------------------------------------

def _make_sections(n: int) -> list:
    return [
        {
            "section_id": f"SEC-{i:03d}",
            "heading_path": [f"Section {i}"],
            "section_text": f"Text of section {i}.",
        }
        for i in range(n)
    ]


def test_surrounding_context_returns_neighbors() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sections = _make_sections(5)
        with open(os.path.join(tmp, "sections.json"), "w", encoding="utf-8") as f:
            json.dump(sections, f)

        result = get_surrounding_context("SEC-002", tmp, window=1)
        # Should return SEC-001 and SEC-003, not SEC-002 itself
        assert len(result) == 2
        assert any("section 1" in r.lower() for r in result)
        assert any("section 3" in r.lower() for r in result)
        assert not any("section 2" in r.lower() for r in result)


def test_surrounding_context_clips_at_boundaries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sections = _make_sections(3)
        with open(os.path.join(tmp, "sections.json"), "w", encoding="utf-8") as f:
            json.dump(sections, f)

        # First section -- no predecessor
        result = get_surrounding_context("SEC-000", tmp, window=2)
        assert len(result) == 2  # SEC-001 and SEC-002 only


def test_surrounding_context_returns_empty_for_unknown_id() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sections = _make_sections(3)
        with open(os.path.join(tmp, "sections.json"), "w", encoding="utf-8") as f:
            json.dump(sections, f)

        result = get_surrounding_context("SEC-999", tmp)
        assert result == []


def test_surrounding_context_returns_empty_when_file_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = get_surrounding_context("SEC-001", tmp)
        assert result == []


# ---------------------------------------------------------------------------
# get_section_family_chunks -- returns List[dict]
# ---------------------------------------------------------------------------

def _make_chunks_json(tmp_dir: str, chunks: list) -> None:
    with open(os.path.join(tmp_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)


def test_family_chunks_returns_list_of_dicts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_chunks_json(tmp, [
            {
                "src_id": "src:0",
                "section_id": "SEC-001",
                "heading_path": ["II.", "A.", "2. Comments"],
                "section_family": "A.",
                "subsection_role": "comments",
                "text": "Some commenter text.",
            },
        ])
        result = get_section_family_chunks("SEC-001", tmp)
        assert isinstance(result, list)
        assert len(result) == 1
        item = result[0]
        assert "src_id" in item
        assert "subsection_role" in item
        assert "heading" in item
        assert "text" in item
        assert item["src_id"] == "src:0"
        assert item["subsection_role"] == "comments"


def test_family_chunks_returns_empty_for_missing_section_id() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_chunks_json(tmp, [
            {
                "src_id": "src:0",
                "section_id": "SEC-001",
                "heading_path": ["II.", "A."],
                "section_family": "A.",
                "subsection_role": "comments",
                "text": "text",
            },
        ])
        result = get_section_family_chunks("SEC-999", tmp)
        assert result == []


def test_family_chunks_filters_by_subsection_role() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_chunks_json(tmp, [
            {
                "src_id": "src:0",
                "section_id": "SEC-001",
                "heading_path": ["II.", "A.", "1. Proposed"],
                "section_family": "A.",
                "subsection_role": "proposed",
                "text": "Proposed text.",
            },
            {
                "src_id": "src:1",
                "section_id": "SEC-002",
                "heading_path": ["II.", "A.", "2. Comments"],
                "section_family": "A.",
                "subsection_role": "comments",
                "text": "Comments text.",
            },
        ])
        # Default roles = ["comments", "final"] -- should skip "proposed"
        result = get_section_family_chunks("SEC-001", tmp)
        assert len(result) == 1
        assert result[0]["src_id"] == "src:1"
