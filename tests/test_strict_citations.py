from __future__ import annotations

import pytest

from sec_interpreter.module import FakeLLM, RuleExtractorModule
from tests.fixtures import sample_input_dict, sample_output_dict


def test_strict_citations_passes_with_citations() -> None:
    """All obligations have source_citations and all entity types have citations → passes."""
    module = RuleExtractorModule(llm=FakeLLM(sample_output_dict()))
    result = module.run(sample_input_dict(strict_citations=True))

    assert all(len(obl.source_citations) > 0 for obl in result.key_obligations)
    assert all(ent.citation for ent in result.affected_entity_types)


def test_strict_citations_fails_without_citations() -> None:
    """An obligation with no source_citations raises ValueError in strict mode."""
    strict_input = sample_input_dict(strict_citations=True)

    bad_output = sample_output_dict()
    bad_output["key_obligations"][0]["source_citations"] = []

    module = RuleExtractorModule(llm=FakeLLM(bad_output))

    with pytest.raises(ValueError, match="must include at least one source_citation"):
        module.run(strict_input)


def test_citation_index_out_of_range() -> None:
    """src:99 is out of range for a 3-chunk document → raises ValueError."""
    bad_output = sample_output_dict()
    bad_output["key_obligations"][0]["source_citations"] = ["src:99"]

    module = RuleExtractorModule(llm=FakeLLM(bad_output))

    with pytest.raises(ValueError, match="Citation index out of range"):
        module.run(sample_input_dict(strict_citations=False))
