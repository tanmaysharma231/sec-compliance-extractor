from __future__ import annotations

import pytest
from pydantic import ValidationError

from sec_interpreter.schemas import (
    InterpretationOutput,
    KeyObligation,
    ObligationInterpretation,
)


# ---------------------------------------------------------------------------
# KeyObligation -- new optional fields: trigger, deadline, disclosure_fields, evidence
# ---------------------------------------------------------------------------

def test_key_obligation_new_fields_all_optional() -> None:
    """Existing obligations without new fields should still validate."""
    obl = KeyObligation(
        obligation_id="OBL-001",
        obligation_text="Registrants should disclose material incidents.",
        source_citations=["src:0"],
    )
    assert obl.trigger is None
    assert obl.deadline is None
    assert obl.disclosure_fields == []
    assert obl.evidence == []


def test_key_obligation_new_fields_populated() -> None:
    obl = KeyObligation(
        obligation_id="OBL-001",
        obligation_text="Disclose material cybersecurity incidents.",
        trigger="Incident determined to be material",
        deadline="4 business days",
        disclosure_fields=["nature of incident", "scope", "timing", "material impact"],
        evidence=["Form 8-K filing", "materiality determination memo"],
        cited_sections=["17 CFR 229.106(b)"],
        source_citations=["src:0"],
    )
    assert obl.trigger == "Incident determined to be material"
    assert obl.deadline == "4 business days"
    assert len(obl.disclosure_fields) == 4
    assert "Form 8-K filing" in obl.evidence


def test_key_obligation_rejects_bad_source_citation() -> None:
    with pytest.raises(ValidationError, match="src:<index>"):
        KeyObligation(
            obligation_id="OBL-001",
            obligation_text="Some obligation.",
            source_citations=["chunk-0"],  # wrong format
        )


def test_key_obligation_disclosure_fields_is_list_of_strings() -> None:
    obl = KeyObligation(
        obligation_id="OBL-002",
        obligation_text="Annual disclosure of cybersecurity governance.",
        disclosure_fields=["board oversight description", "management expertise"],
        source_citations=["src:1"],
    )
    assert all(isinstance(f, str) for f in obl.disclosure_fields)


# ---------------------------------------------------------------------------
# ObligationInterpretation
# ---------------------------------------------------------------------------

def test_obligation_interpretation_minimal() -> None:
    interp = ObligationInterpretation(
        obligation_id="OBL-001",
        primary_interpretation="The registrant must file Form 8-K within 4 business days.",
        compliance_implication="Establish an incident response process with materiality determination.",
    )
    assert interp.confidence_level == "medium"
    assert interp.supporting_sections == []
    assert interp.alternative_interpretations == []
    assert interp.ambiguous_terms == []


def test_obligation_interpretation_full() -> None:
    interp = ObligationInterpretation(
        obligation_id="OBL-001",
        primary_interpretation="File Form 8-K within 4 business days of materiality determination.",
        supporting_sections=["17 CFR 229.106(b)", "17 CFR 240.13a-11"],
        alternative_interpretations=[
            "The clock may start at discovery rather than at formal determination."
        ],
        ambiguous_terms=["material", "promptly"],
        compliance_implication="Build an incident escalation and materiality review process.",
        confidence_level="high",
    )
    assert interp.confidence_level == "high"
    assert len(interp.supporting_sections) == 2
    assert "material" in interp.ambiguous_terms


def test_obligation_interpretation_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        ObligationInterpretation(
            obligation_id="OBL-001",
            primary_interpretation="Some interpretation.",
            compliance_implication="Some implication.",
            confidence_level="very_high",  # not in Literal["high", "medium", "low"]
        )


def test_obligation_interpretation_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ObligationInterpretation(
            obligation_id="OBL-001",
            primary_interpretation="Some interpretation.",
            compliance_implication="Some implication.",
            unknown_field="should not be here",
        )


# ---------------------------------------------------------------------------
# InterpretationOutput
# ---------------------------------------------------------------------------

def test_interpretation_output_empty_interpretations() -> None:
    output = InterpretationOutput(
        run_id="abc123",
        rule_title="SEC Cybersecurity Disclosure Rule",
    )
    assert output.interpretations == []


def test_interpretation_output_with_interpretations() -> None:
    interps = [
        ObligationInterpretation(
            obligation_id=f"OBL-00{i}",
            primary_interpretation=f"Interpretation {i}.",
            compliance_implication=f"Implication {i}.",
            confidence_level="medium",
        )
        for i in range(1, 4)
    ]
    output = InterpretationOutput(
        run_id="abc123",
        rule_title="SEC Cybersecurity Disclosure Rule",
        interpretations=interps,
    )
    assert len(output.interpretations) == 3
    assert output.interpretations[0].obligation_id == "OBL-001"


def test_interpretation_output_roundtrip_json() -> None:
    """model_dump -> model_validate round-trip preserves all fields."""
    interp = ObligationInterpretation(
        obligation_id="OBL-001",
        primary_interpretation="File Form 8-K within 4 days.",
        compliance_implication="Build incident escalation process.",
        confidence_level="high",
        ambiguous_terms=["material"],
    )
    output = InterpretationOutput(
        run_id="test-run",
        rule_title="Test Rule",
        interpretations=[interp],
    )
    dumped = output.model_dump(mode="json")
    restored = InterpretationOutput.model_validate(dumped)

    assert restored.run_id == "test-run"
    assert restored.interpretations[0].confidence_level == "high"
    assert restored.interpretations[0].ambiguous_terms == ["material"]
