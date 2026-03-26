"""
tests/test_case_brief.py

Tests for build_bin_pass_prompt and build_case_brief_prompt.
"""
from __future__ import annotations

from sec_interpreter.prompts import build_bin_pass_prompt, build_case_brief_prompt


# ---------------------------------------------------------------------------
# build_bin_pass_prompt
# ---------------------------------------------------------------------------

def _make_flagged_chunks(n: int = 2) -> list:
    return [
        {
            "src_id": f"src:{i}",
            "heading": f"II > A > {i}",
            "text": f"Registrants must file within {i + 1} days of determining materiality.",
            "has_obligations": True,
            "has_codified_text": False,
            "has_scope": False,
        }
        for i in range(n)
    ]


def _make_known_obligations(n: int = 2) -> list:
    return [
        {
            "obligation_id": f"OBL-{i:03d}",
            "obligation_text": f"Obligation text {i}.",
        }
        for i in range(1, n + 1)
    ]


def test_bin_prompt_contains_known_obligation_ids() -> None:
    chunks = _make_flagged_chunks(2)
    obligations = _make_known_obligations(2)
    prompt = build_bin_pass_prompt(chunks, obligations)
    assert "OBL-001" in prompt
    assert "OBL-002" in prompt


def test_bin_prompt_contains_chunk_text() -> None:
    chunks = _make_flagged_chunks(2)
    prompt = build_bin_pass_prompt(chunks, [])
    assert "Registrants must file" in prompt


def test_bin_prompt_contains_all_valid_bin_types() -> None:
    prompt = build_bin_pass_prompt([], [])
    for bin_type in [
        "missed_obligation",
        "scope_modifier",
        "implied_requirement",
        "definition",
        "edge_case",
        "not_relevant",
    ]:
        assert bin_type in prompt


def test_bin_prompt_contains_src_ids() -> None:
    chunks = _make_flagged_chunks(3)
    prompt = build_bin_pass_prompt(chunks, [])
    assert "src:0" in prompt
    assert "src:1" in prompt
    assert "src:2" in prompt


def test_bin_prompt_handles_empty_inputs() -> None:
    prompt = build_bin_pass_prompt([], [])
    assert "no obligations extracted yet" in prompt


def test_bin_prompt_contains_schema_hint() -> None:
    prompt = build_bin_pass_prompt([], [])
    # Schema hint should include the findings key
    assert "findings" in prompt
    assert "finding_type" in prompt


# ---------------------------------------------------------------------------
# build_case_brief_prompt
# ---------------------------------------------------------------------------

def _make_extraction_output() -> dict:
    return {
        "rule_metadata": {
            "rule_title": "SEC Cybersecurity Disclosure Rule",
            "release_number": "33-11216",
            "effective_date": "2023-09-05",
        },
        "key_obligations": [
            {
                "obligation_id": "OBL-001",
                "obligation_text": "Registrants must disclose material cybersecurity incidents on Form 8-K.",
                "cited_sections": ["17 CFR 229.106(b)"],
                "deadline": "4 business days",
            },
            {
                "obligation_id": "OBL-002",
                "obligation_text": "Annual disclosure of cybersecurity risk management.",
                "cited_sections": ["17 CFR 229.106(a)"],
                "deadline": None,
            },
        ],
    }


def _make_bin_findings() -> list:
    return [
        {
            "finding_type": "scope_modifier",
            "text": "Smaller reporting companies have a 30-day extension.",
            "related_to": ["OBL-001"],
            "notes": "Applies to SRCs only",
        },
        {
            "finding_type": "definition",
            "text": "Material means a substantial likelihood a reasonable investor would consider important.",
            "related_to": ["OBL-001", "OBL-002"],
            "notes": None,
        },
        {
            "finding_type": "not_relevant",
            "text": "Economic analysis section.",
            "related_to": [],
            "notes": None,
        },
    ]


def _make_interpretation_output() -> dict:
    return {
        "run_id": "test-run",
        "rule_title": "SEC Cybersecurity Disclosure Rule",
        "interpretations": [
            {
                "obligation_id": "OBL-001",
                "primary_interpretation": "File Form 8-K within 4 business days of materiality determination.",
                "compliance_implication": "Establish an incident response process.",
                "confidence_level": "high",
            }
        ],
    }


def test_brief_prompt_contains_rule_title() -> None:
    prompt = build_case_brief_prompt(
        _make_extraction_output(), _make_bin_findings(), _make_interpretation_output(), []
    )
    assert "SEC Cybersecurity Disclosure Rule" in prompt


def test_brief_prompt_contains_obligation_ids() -> None:
    prompt = build_case_brief_prompt(
        _make_extraction_output(), _make_bin_findings(), _make_interpretation_output(), []
    )
    assert "OBL-001" in prompt
    assert "OBL-002" in prompt


def test_brief_prompt_contains_bin_findings() -> None:
    prompt = build_case_brief_prompt(
        _make_extraction_output(), _make_bin_findings(), _make_interpretation_output(), []
    )
    assert "scope_modifier" in prompt
    assert "definition" in prompt


def test_brief_prompt_excludes_not_relevant_findings() -> None:
    prompt = build_case_brief_prompt(
        _make_extraction_output(), _make_bin_findings(), _make_interpretation_output(), []
    )
    # not_relevant text should not appear
    assert "Economic analysis section." not in prompt


def test_brief_prompt_contains_interpretation_summary() -> None:
    prompt = build_case_brief_prompt(
        _make_extraction_output(), _make_bin_findings(), _make_interpretation_output(), []
    )
    assert "Form 8-K within 4 business days" in prompt


def test_brief_prompt_contains_named_section_texts() -> None:
    named_texts = ["[II > Effective Dates]\nThe rule is effective September 5, 2023."]
    prompt = build_case_brief_prompt(
        _make_extraction_output(), [], {}, named_texts
    )
    assert "September 5, 2023" in prompt


def test_brief_prompt_contains_section_headers() -> None:
    prompt = build_case_brief_prompt(
        _make_extraction_output(), [], {}, []
    )
    for section in [
        "OBLIGATIONS",
        "BIN FINDINGS",
        "INTERPRETATION SUMMARIES",
        "NAMED SECTIONS",
    ]:
        assert section in prompt


def test_brief_prompt_handles_empty_inputs() -> None:
    prompt = build_case_brief_prompt({}, [], {}, [])
    # Should not raise, should produce a non-empty prompt
    assert len(prompt) > 100
    assert "OBLIGATIONS" in prompt


def test_brief_prompt_contains_release_number() -> None:
    prompt = build_case_brief_prompt(
        _make_extraction_output(), [], {}, []
    )
    assert "33-11216" in prompt
