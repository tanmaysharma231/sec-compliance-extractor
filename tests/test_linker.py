"""
tests/test_linker.py

Unit tests for the obligation context linker:
  - ObligationContextLinks schema
  - _link_family_context helper
  - build_context_linker_prompt helper
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from sec_interpreter.schemas import ObligationContextLinks
from sec_interpreter.prompts import build_context_linker_prompt
from sec_interpreter.interpret_graph import _link_family_context


# ---------------------------------------------------------------------------
# ObligationContextLinks schema
# ---------------------------------------------------------------------------

def test_schema_validates_correctly() -> None:
    data = {"key_indices": [0, 2], "supporting_indices": [1], "skip_indices": [3]}
    links = ObligationContextLinks.model_validate(data)
    assert links.key_indices == [0, 2]
    assert links.supporting_indices == [1]
    assert links.skip_indices == [3]


def test_schema_ignores_extra_fields() -> None:
    data = {
        "key_indices": [0],
        "supporting_indices": [],
        "skip_indices": [1],
        "unexpected_field": "should be ignored",
    }
    links = ObligationContextLinks.model_validate(data)
    assert links.key_indices == [0]


def test_schema_defaults_to_empty_lists() -> None:
    links = ObligationContextLinks.model_validate({})
    assert links.key_indices == []
    assert links.supporting_indices == []
    assert links.skip_indices == []


# ---------------------------------------------------------------------------
# _link_family_context
# ---------------------------------------------------------------------------

def _make_chunks(n: int) -> list:
    return [
        {
            "src_id": f"src:{i}",
            "subsection_role": "comments",
            "heading": f"II > A > {i}",
            "text": f"Chunk text {i}.",
        }
        for i in range(n)
    ]


def _make_obl(obl_id: str = "OBL-001") -> dict:
    return {
        "obligation_id": obl_id,
        "obligation_text": "Registrants must disclose material cybersecurity incidents.",
    }


def _make_cheap_llm(response_json: str) -> Any:
    mock = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = response_json
    mock.invoke.return_value = mock_resp
    return mock


_LOG = logging.getLogger("test_linker")


def test_returns_empty_on_empty_input() -> None:
    cheap_llm = MagicMock()
    result = _link_family_context(_make_obl(), [], cheap_llm, _LOG)
    assert result == []
    cheap_llm.invoke.assert_not_called()


def test_returns_all_chunks_on_llm_failure() -> None:
    chunks = _make_chunks(3)
    cheap_llm = MagicMock()
    cheap_llm.invoke.side_effect = RuntimeError("LLM unavailable")
    result = _link_family_context(_make_obl(), chunks, cheap_llm, _LOG)
    assert result == chunks


def test_filters_correctly_on_valid_llm_output() -> None:
    chunks = _make_chunks(5)
    llm_response = '{"key_indices": [0, 2], "supporting_indices": [1], "skip_indices": [3, 4]}'
    cheap_llm = _make_cheap_llm(llm_response)
    result = _link_family_context(_make_obl(), chunks, cheap_llm, _LOG)
    # Should keep key (0, 2) + supporting (1), skip (3, 4)
    kept_ids = [c["src_id"] for c in result]
    assert "src:0" in kept_ids
    assert "src:1" in kept_ids
    assert "src:2" in kept_ids
    assert "src:3" not in kept_ids
    assert "src:4" not in kept_ids


def test_clamps_out_of_range_indices_falls_back_to_all() -> None:
    chunks = _make_chunks(3)
    # LLM returns index 99 which is out of range; keep set becomes empty
    llm_response = '{"key_indices": [99], "supporting_indices": [], "skip_indices": [0, 1, 2]}'
    cheap_llm = _make_cheap_llm(llm_response)
    result = _link_family_context(_make_obl(), chunks, cheap_llm, _LOG)
    # All clamped out -> fallback returns all chunks
    assert result == chunks


def test_clamps_valid_and_invalid_indices() -> None:
    chunks = _make_chunks(3)
    # Mix of valid (0) and invalid (99) indices
    llm_response = '{"key_indices": [0, 99], "supporting_indices": [], "skip_indices": [1, 2]}'
    cheap_llm = _make_cheap_llm(llm_response)
    result = _link_family_context(_make_obl(), chunks, cheap_llm, _LOG)
    kept_ids = [c["src_id"] for c in result]
    assert kept_ids == ["src:0"]


# ---------------------------------------------------------------------------
# build_context_linker_prompt
# ---------------------------------------------------------------------------

def test_prompt_contains_obligation_text() -> None:
    chunks = _make_chunks(3)
    obl_text = "Registrants must disclose material cybersecurity incidents within 4 days."
    prompt = build_context_linker_prompt(obl_text, "OBL-001", chunks)
    assert obl_text in prompt


def test_prompt_contains_idx_column() -> None:
    chunks = _make_chunks(3)
    prompt = build_context_linker_prompt("Some obligation.", "OBL-001", chunks)
    # Table should have idx column header
    assert "idx" in prompt
    # And 0-based indices
    assert "0" in prompt
    assert "1" in prompt
    assert "2" in prompt


def test_prompt_exhaustive_partition_rule() -> None:
    chunks = _make_chunks(2)
    prompt = build_context_linker_prompt("Some obligation.", "OBL-001", chunks)
    # Should tell LLM every index must appear in exactly one list
    assert "Every index 0 to 1 must appear in exactly one list" in prompt
