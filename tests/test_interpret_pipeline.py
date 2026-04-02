"""
tests/test_interpret_pipeline.py

Tests for:
  - search_chunks_for_term (new tool)
  - agentic term lookup loop in run_interpret_pipeline
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sec_interpreter.tools import search_chunks_for_term

_LOG = logging.getLogger("test")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_chunks(artifact_dir: str, chunks: list) -> None:
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)


def _make_chunk(
    src_id: str,
    text: str,
    has_definitions: bool = False,
    has_example: bool = False,
) -> dict:
    return {
        "src_id": src_id,
        "section_id": "03010300",
        "heading_path": ["II. Discussion", "A. Incidents"],
        "text": text,
        "has_definitions": has_definitions,
        "has_example": has_example,
        "has_obligations": False,
        "subsection_role": "final",
        "section_family": "A",
    }


# ---------------------------------------------------------------------------
# search_chunks_for_term
# ---------------------------------------------------------------------------

def test_returns_top_matches_by_hit_count() -> None:
    chunks = [
        _make_chunk("src:0", "material means something important to investors"),
        _make_chunk("src:1", "material material material appears three times in this text"),
        _make_chunk("src:2", "no relevant content here"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = os.path.join(tmp, "artifacts", "run1")
        _write_chunks(artifact_dir, chunks)
        results = search_chunks_for_term("material", artifact_dir, top_n=3)

    assert len(results) == 2
    # src:1 has 3 hits, src:0 has 1 hit -> src:1 should rank first
    assert results[0]["src_id"] == "src:1"
    assert results[1]["src_id"] == "src:0"


def test_prefers_has_definitions_over_raw_hit_count() -> None:
    chunks = [
        _make_chunk("src:0", "material material", has_definitions=False),
        _make_chunk("src:1", "material", has_definitions=True),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = os.path.join(tmp, "artifacts", "run1")
        _write_chunks(artifact_dir, chunks)
        results = search_chunks_for_term("material", artifact_dir, top_n=2)

    # src:1: score=1+2=3; src:0: score=2 -> src:1 wins
    assert results[0]["src_id"] == "src:1"


def test_has_example_bonus() -> None:
    chunks = [
        _make_chunk("src:0", "material material", has_example=False),
        _make_chunk("src:1", "material", has_example=True),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = os.path.join(tmp, "artifacts", "run1")
        _write_chunks(artifact_dir, chunks)
        results = search_chunks_for_term("material", artifact_dir, top_n=2)

    # src:0: score=2; src:1: score=1+1=2 -> tie, both returned
    assert len(results) == 2


def test_empty_on_no_match() -> None:
    chunks = [_make_chunk("src:0", "no relevant content here")]
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = os.path.join(tmp, "artifacts", "run1")
        _write_chunks(artifact_dir, chunks)
        results = search_chunks_for_term("material", artifact_dir)

    assert results == []


def test_empty_on_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        results = search_chunks_for_term("material", tmp)
    assert results == []


def test_respects_top_n() -> None:
    chunks = [
        _make_chunk(f"src:{i}", "material " * (i + 1)) for i in range(10)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = os.path.join(tmp, "artifacts", "run1")
        _write_chunks(artifact_dir, chunks)
        results = search_chunks_for_term("material", artifact_dir, top_n=3)

    assert len(results) == 3


def test_result_has_required_keys() -> None:
    chunks = [_make_chunk("src:0", "material disclosure required")]
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = os.path.join(tmp, "artifacts", "run1")
        _write_chunks(artifact_dir, chunks)
        results = search_chunks_for_term("material", artifact_dir)

    assert len(results) == 1
    assert set(results[0].keys()) >= {"src_id", "heading", "text"}


def test_word_boundary_match() -> None:
    # "immaterial" should not match a search for "material" as whole word
    chunks = [
        _make_chunk("src:0", "the incident was immaterial to investors"),
        _make_chunk("src:1", "material impact on operations"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = os.path.join(tmp, "artifacts", "run1")
        _write_chunks(artifact_dir, chunks)
        results = search_chunks_for_term("material", artifact_dir)

    assert len(results) == 1
    assert results[0]["src_id"] == "src:1"


# ---------------------------------------------------------------------------
# Term lookup loop in run_interpret_pipeline
# ---------------------------------------------------------------------------

def _make_interpretation_json(
    obligation_id: str = "OBL-001",
    lookup_requests: list = None,
    needs_more_context: bool = False,
) -> str:
    return json.dumps({
        "obligation_id": obligation_id,
        "primary_interpretation": "The registrant must disclose the incident.",
        "supporting_sections": [],
        "alternative_interpretations": [],
        "ambiguous_terms": [],
        "compliance_implication": "Establish a disclosure process.",
        "confidence_level": "medium",
        "needs_more_context": needs_more_context,
        "lookup_requests": lookup_requests or [],
    })


def _write_pipeline_artifacts(artifact_dir: str) -> None:
    os.makedirs(artifact_dir, exist_ok=True)

    validated = {
        "rule_metadata": {"rule_title": "Test Rule"},
        "key_obligations": [
            {
                "obligation_id": "OBL-001",
                "obligation_text": "Registrants must disclose material cybersecurity incidents.",
                "cited_sections": [],
                "source_citations": ["src:0"],
                "trigger": None,
                "deadline": None,
            }
        ],
    }
    with open(os.path.join(artifact_dir, "validated_output.json"), "w") as f:
        json.dump(validated, f)

    chunks = [_make_chunk("src:0", "material disclosure required within four business days")]
    _write_chunks(artifact_dir, chunks)


def test_term_lookup_loop_fires_on_lookup_requests() -> None:
    """LLM requests lookup on first call; pipeline fetches and re-interprets once."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        run_id = "test-run"
        artifact_dir = os.path.join(tmp, "artifacts", run_id)
        _write_pipeline_artifacts(artifact_dir)

        call_count = 0

        def fake_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(content=_make_interpretation_json(
                    lookup_requests=["material"]
                ))
            return SimpleNamespace(content=_make_interpretation_json(
                lookup_requests=[]
            ))

        llm = MagicMock()
        llm.invoke.side_effect = fake_invoke

        from sec_interpreter.interpret_graph import run_interpret_pipeline
        os.chdir(tmp)
        try:
            output = run_interpret_pipeline(run_id, llm, llm, _LOG)
        finally:
            os.chdir(original_cwd)

        assert llm.invoke.call_count == 2
        assert len(output.interpretations) == 1


def test_term_lookup_loop_skips_reinterpret_if_no_chunks_found() -> None:
    """If search returns nothing for the requested term, do not re-interpret."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        run_id = "test-run"
        artifact_dir = os.path.join(tmp, "artifacts", run_id)
        _write_pipeline_artifacts(artifact_dir)

        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(
            content=_make_interpretation_json(lookup_requests=["xyzzy"])
        )

        from sec_interpreter.interpret_graph import run_interpret_pipeline
        os.chdir(tmp)
        try:
            output = run_interpret_pipeline(run_id, llm, llm, _LOG)
        finally:
            os.chdir(original_cwd)

        # "xyzzy" matches nothing -> no re-interpret
        assert llm.invoke.call_count == 1
        assert len(output.interpretations) == 1
