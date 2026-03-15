from __future__ import annotations

import pytest

from sec_interpreter.module import FakeLLM, RuleExtractorModule
from tests.fixtures import sample_input_dict, sample_output_dict


def test_output_validates_with_fake_llm() -> None:
    module = RuleExtractorModule(llm=FakeLLM(sample_output_dict()))
    result = module.run(sample_input_dict(strict_citations=False))

    assert result.rule_metadata.rule_title == (
        "SEC Cybersecurity Risk Management, Strategy, Governance, and Incident Disclosure Rules"
    )
    assert len(result.key_obligations) == 3
    assert result.key_obligations[0].obligation_id == "OBL-001"
    assert len(result.compliance_impact_areas) == 4


def test_safe_language_rejects_banned_terms() -> None:
    bad_output = sample_output_dict()
    # Inject a banned term into an obligation
    bad_output["key_obligations"][0]["obligation_text"] = (
        "Registrants are in violation of the disclosure requirement."
    )

    module = RuleExtractorModule(llm=FakeLLM(bad_output))

    with pytest.raises(ValueError, match="safe language"):
        module.run(sample_input_dict(strict_citations=False))
