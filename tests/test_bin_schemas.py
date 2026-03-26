"""
tests/test_bin_schemas.py

Tests for new schemas in the structure-first pipeline rebuild:
  BinFinding, BinPassOutput, ObligationSection, StructureScanResult,
  and rule_provision on KeyObligation.
"""
from __future__ import annotations

import pytest

from sec_interpreter.schemas import (
    BinFinding,
    BinPassOutput,
    KeyObligation,
    ObligationSection,
    StructureScanResult,
    VALID_BIN_TYPES,
)


# ---------------------------------------------------------------------------
# KeyObligation -- rule_provision field
# ---------------------------------------------------------------------------

def test_key_obligation_accepts_rule_provision() -> None:
    obl = KeyObligation.model_validate({
        "obligation_id": "OBL-001",
        "rule_provision": "17 CFR 229.106(b)",
        "obligation_text": "Registrants must disclose.",
        "source_citations": [],
    })
    assert obl.rule_provision == "17 CFR 229.106(b)"


def test_key_obligation_rule_provision_optional() -> None:
    obl = KeyObligation.model_validate({
        "obligation_id": "OBL-001",
        "obligation_text": "Registrants must disclose.",
        "source_citations": [],
    })
    assert obl.rule_provision is None


# ---------------------------------------------------------------------------
# BinFinding
# ---------------------------------------------------------------------------

def test_bin_finding_validates_correctly() -> None:
    finding = BinFinding.model_validate({
        "finding_type": "scope_modifier",
        "text": "Smaller reporting companies have extended deadlines.",
        "related_to": ["OBL-001", "OBL-002"],
        "source_chunks": ["src:5"],
    })
    assert finding.finding_type == "scope_modifier"
    assert finding.related_to == ["OBL-001", "OBL-002"]
    assert finding.notes is None


def test_bin_finding_ignores_extra_fields() -> None:
    finding = BinFinding.model_validate({
        "finding_type": "definition",
        "text": "Material means...",
        "related_to": [],
        "source_chunks": [],
        "unexpected_field": "should be ignored",
    })
    assert finding.finding_type == "definition"


def test_bin_finding_rejects_invalid_type() -> None:
    with pytest.raises(Exception):
        BinFinding.model_validate({
            "finding_type": "invalid_type",
            "text": "Some text.",
        })


def test_bin_finding_all_valid_types() -> None:
    for bt in VALID_BIN_TYPES:
        finding = BinFinding.model_validate({
            "finding_type": bt,
            "text": "Some text.",
        })
        assert finding.finding_type == bt


def test_bin_finding_defaults_to_empty_lists() -> None:
    finding = BinFinding.model_validate({
        "finding_type": "not_relevant",
        "text": "Some text.",
    })
    assert finding.related_to == []
    assert finding.source_chunks == []


# ---------------------------------------------------------------------------
# BinPassOutput
# ---------------------------------------------------------------------------

def test_bin_pass_output_validates_correctly() -> None:
    output = BinPassOutput.model_validate({
        "run_id": "abc123",
        "findings": [
            {
                "finding_type": "definition",
                "text": "Material means...",
                "related_to": ["OBL-001"],
                "source_chunks": ["src:10"],
            }
        ],
    })
    assert output.run_id == "abc123"
    assert len(output.findings) == 1
    assert output.findings[0].finding_type == "definition"


def test_bin_pass_output_empty_findings() -> None:
    output = BinPassOutput.model_validate({"run_id": "abc123"})
    assert output.findings == []


def test_bin_pass_output_ignores_extra_fields() -> None:
    output = BinPassOutput.model_validate({
        "run_id": "abc123",
        "findings": [],
        "extra_key": "ignored",
    })
    assert output.run_id == "abc123"


# ---------------------------------------------------------------------------
# ObligationSection
# ---------------------------------------------------------------------------

def test_obligation_section_validates_correctly() -> None:
    section = ObligationSection.model_validate({
        "section_letter": "A",
        "heading": "A. Disclosure of Cybersecurity Incidents",
        "section_id": "SEC-010",
        "cfr_citations": ["17 CFR 229.106(b)"],
        "structured_chunk_ids": ["src:8", "src:17"],
    })
    assert section.section_letter == "A"
    assert section.cfr_citations == ["17 CFR 229.106(b)"]
    assert section.structured_chunk_ids == ["src:8", "src:17"]


def test_obligation_section_defaults_to_empty_lists() -> None:
    section = ObligationSection.model_validate({
        "section_letter": "B",
        "heading": "B. Some Heading",
        "section_id": "SEC-020",
    })
    assert section.cfr_citations == []
    assert section.structured_chunk_ids == []


# ---------------------------------------------------------------------------
# StructureScanResult
# ---------------------------------------------------------------------------

def test_structure_scan_result_validates_correctly() -> None:
    result = StructureScanResult.model_validate({
        "run_id": "abc123",
        "obligation_sections": [
            {
                "section_letter": "A",
                "heading": "A. Heading",
                "section_id": "SEC-010",
                "cfr_citations": [],
                "structured_chunk_ids": ["src:8"],
            }
        ],
        "named_section_chunk_ids": ["src:78"],
        "expected_obligation_count": 1,
        "structured_chunk_ids": ["src:8"],
    })
    assert result.run_id == "abc123"
    assert result.expected_obligation_count == 1
    assert len(result.obligation_sections) == 1
    assert result.named_section_chunk_ids == ["src:78"]


def test_structure_scan_result_empty_defaults() -> None:
    result = StructureScanResult.model_validate({"run_id": "abc123"})
    assert result.obligation_sections == []
    assert result.named_section_chunk_ids == []
    assert result.expected_obligation_count == 0
    assert result.structured_chunk_ids == []
