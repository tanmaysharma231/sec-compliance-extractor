from __future__ import annotations

import pytest

from sec_interpreter.module import FakeLLM, RuleExtractorModule
from tests.fixtures import sample_input_dict, sample_output_dict


def test_assumption_required_when_ambiguity() -> None:
    """Output with assumptions list passes validation."""
    output = sample_output_dict()
    # sample output already has an assumption — just confirm it validates cleanly
    module = RuleExtractorModule(llm=FakeLLM(output))
    result = module.run(sample_input_dict(strict_citations=False))

    assert len(result.assumptions) >= 1
    assert result.assumptions[0].assumption_text


def test_obligation_link_validation() -> None:
    """A ComplianceImpactArea that links to an unknown obligation_id raises ValueError."""
    bad_output = sample_output_dict()
    # Reference an obligation that doesn't exist
    bad_output["compliance_impact_areas"][0]["linked_obligation_ids"] = ["OBL-999"]

    module = RuleExtractorModule(llm=FakeLLM(bad_output))

    with pytest.raises(ValueError, match="unknown obligation_id"):
        module.run(sample_input_dict(strict_citations=False))
