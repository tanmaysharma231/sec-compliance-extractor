"""
tests/test_bin.py

Tests for sec_interpreter/bin_graph.py:
  - run_bin_pass()

Uses unittest.mock.MagicMock for cheap_llm and tempfile.TemporaryDirectory
for artifact directories. The function under test uses os.path.join("artifacts", run_id)
as its artifact_dir, so tests monkey-patch the path by writing fixtures directly to
temp dirs and passing matching run_ids -- or they supply absolute artifact paths via
a thin wrapper approach.

Strategy: we write chunks.json and structure_scan_result.json into a temp directory,
then call run_bin_pass with a run_id whose artifact_dir resolves to that temp dir.
Because run_bin_pass constructs artifact_dir as os.path.join("artifacts", run_id),
we change the working directory to a temp parent so "artifacts/<run_id>" lands inside
our controlled directory.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import pytest

from sec_interpreter.bin_graph import run_bin_pass
from sec_interpreter.schemas import BinPassOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_ID = "test-run-001"


def _write_chunks(artifact_dir: str, chunks: list) -> None:
    with open(os.path.join(artifact_dir, "chunks.json"), "w", encoding="utf-8") as fh:
        json.dump(chunks, fh)


def _write_scan_result(artifact_dir: str, structured_ids: list) -> None:
    scan = {
        "run_id": _RUN_ID,
        "obligation_sections": [],
        "named_section_chunk_ids": [],
        "expected_obligation_count": 0,
        "structured_chunk_ids": structured_ids,
    }
    with open(os.path.join(artifact_dir, "structure_scan_result.json"), "w", encoding="utf-8") as fh:
        json.dump(scan, fh)


def _make_chunk(
    src_id: str,
    has_obligations: bool = False,
    has_codified_text: bool = False,
    has_scope: bool = False,
) -> dict:
    return {
        "src_id": src_id,
        "text": f"Chunk text for {src_id}.",
        "has_obligations": has_obligations,
        "has_codified_text": has_codified_text,
        "has_scope": has_scope,
        "has_definitions": False,
        "has_dates": False,
    }


def _make_llm_response(findings: list) -> MagicMock:
    """Return a MagicMock that looks like an LLM response with JSON content."""
    mock_resp = MagicMock()
    mock_resp.content = json.dumps({"findings": findings})
    return mock_resp


def _make_finding(finding_type: str = "scope_modifier") -> dict:
    return {
        "finding_type": finding_type,
        "text": "Smaller reporting companies have extended deadlines.",
        "related_to": ["OBL-001"],
        "source_chunks": ["src:5"],
        "notes": "Applies only to SRCs.",
    }


def _make_logger() -> logging.Logger:
    return logging.getLogger("test_bin")


def _setup_artifacts(tmp_parent: str, chunks: list, structured_ids: list) -> str:
    """
    Create artifacts/<run_id>/ inside tmp_parent, write fixtures, and return artifact_dir.
    Also sets os.chdir so run_bin_pass resolves paths correctly.
    """
    artifact_dir = os.path.join(tmp_parent, "artifacts", _RUN_ID)
    os.makedirs(artifact_dir)
    _write_chunks(artifact_dir, chunks)
    _write_scan_result(artifact_dir, structured_ids)
    return artifact_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunBinPass(unittest.TestCase):

    def setUp(self) -> None:
        self._orig_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        self._tmp_dir = self._tmp.name
        os.chdir(self._tmp_dir)

    def tearDown(self) -> None:
        os.chdir(self._orig_cwd)
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # test_filters_only_flagged_chunks
    # ------------------------------------------------------------------

    def test_filters_only_flagged_chunks(self) -> None:
        """Chunks without flags are excluded; LLM is called only with flagged chunks."""
        chunks = [
            _make_chunk("src:0", has_obligations=True),   # flagged -- in structured
            _make_chunk("src:1", has_obligations=True),   # flagged -- NOT in structured
            _make_chunk("src:2"),                          # no flags -- NOT in structured
        ]
        _setup_artifacts(self._tmp_dir, chunks, structured_ids=["src:0"])

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response([])

        logger = _make_logger()
        run_bin_pass(_RUN_ID, {}, mock_llm, logger)

        # LLM must have been called (flagged=src:1)
        mock_llm.invoke.assert_called_once()

        # Inspect the prompt passed to LLM -- should NOT contain src:0 (structured) or src:2 (no flags)
        call_args = mock_llm.invoke.call_args
        prompt_content = call_args[0][0][0].content  # HumanMessage.content
        self.assertIn("src:1", prompt_content)
        self.assertNotIn("src:2", prompt_content)

    # ------------------------------------------------------------------
    # test_returns_empty_when_no_flagged_chunks
    # ------------------------------------------------------------------

    def test_returns_empty_when_no_flagged_chunks(self) -> None:
        """If all remaining chunks have no flags, return empty BinPassOutput without calling LLM."""
        chunks = [
            _make_chunk("src:0"),   # no flags, not in structured
            _make_chunk("src:1"),   # no flags, not in structured
        ]
        _setup_artifacts(self._tmp_dir, chunks, structured_ids=[])

        mock_llm = MagicMock()
        logger = _make_logger()

        result = run_bin_pass(_RUN_ID, {}, mock_llm, logger)

        mock_llm.invoke.assert_not_called()
        self.assertIsInstance(result, BinPassOutput)
        self.assertEqual(result.findings, [])

    # ------------------------------------------------------------------
    # test_returns_empty_when_no_scan_result
    # ------------------------------------------------------------------

    def test_returns_empty_when_no_scan_result(self) -> None:
        """If structure_scan_result.json is missing, return empty BinPassOutput."""
        # Write only chunks -- no scan result
        artifact_dir = os.path.join(self._tmp_dir, "artifacts", _RUN_ID)
        os.makedirs(artifact_dir)
        _write_chunks(artifact_dir, [_make_chunk("src:0", has_obligations=True)])
        # Do NOT write structure_scan_result.json

        mock_llm = MagicMock()
        logger = _make_logger()

        result = run_bin_pass(_RUN_ID, {}, mock_llm, logger)

        mock_llm.invoke.assert_not_called()
        self.assertIsInstance(result, BinPassOutput)
        self.assertEqual(result.findings, [])

    # ------------------------------------------------------------------
    # test_parses_llm_output_correctly
    # ------------------------------------------------------------------

    def test_parses_llm_output_correctly(self) -> None:
        """Mock LLM returns valid JSON with one finding -> BinPassOutput has 1 finding."""
        chunks = [_make_chunk("src:0", has_obligations=True)]
        _setup_artifacts(self._tmp_dir, chunks, structured_ids=[])

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response([_make_finding("scope_modifier")])

        logger = _make_logger()
        result = run_bin_pass(_RUN_ID, {}, mock_llm, logger)

        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].finding_type, "scope_modifier")

    # ------------------------------------------------------------------
    # test_skips_invalid_findings
    # ------------------------------------------------------------------

    def test_skips_invalid_findings(self) -> None:
        """One valid + one invalid finding_type -> only valid one in output."""
        chunks = [_make_chunk("src:0", has_obligations=True)]
        _setup_artifacts(self._tmp_dir, chunks, structured_ids=[])

        invalid_finding = {
            "finding_type": "not_a_real_type",  # invalid
            "text": "Some text.",
        }
        valid_finding = _make_finding("definition")

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response([valid_finding, invalid_finding])

        logger = _make_logger()
        result = run_bin_pass(_RUN_ID, {}, mock_llm, logger)

        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].finding_type, "definition")

    # ------------------------------------------------------------------
    # test_missed_obligation_logged_as_warning
    # ------------------------------------------------------------------

    def test_missed_obligation_logged_as_warning(self) -> None:
        """finding_type=missed_obligation triggers logger.warning."""
        chunks = [_make_chunk("src:0", has_obligations=True)]
        _setup_artifacts(self._tmp_dir, chunks, structured_ids=[])

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response([_make_finding("missed_obligation")])

        mock_logger = MagicMock(spec=logging.Logger)

        run_bin_pass(_RUN_ID, {}, mock_llm, mock_logger)

        # logger.warning should have been called with the missed obligation message
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        self.assertTrue(
            any("missed obligation" in call for call in warning_calls),
            msg=f"Expected 'missed obligation' warning. Calls: {warning_calls}",
        )

    # ------------------------------------------------------------------
    # test_saves_bin_findings_json
    # ------------------------------------------------------------------

    def test_saves_bin_findings_json(self) -> None:
        """After run, bin_findings.json exists in artifact_dir."""
        chunks = [_make_chunk("src:0", has_scope=True)]
        artifact_dir = _setup_artifacts(self._tmp_dir, chunks, structured_ids=[])

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response([_make_finding("scope_modifier")])

        logger = _make_logger()
        run_bin_pass(_RUN_ID, {}, mock_llm, logger)

        out_path = os.path.join(artifact_dir, "bin_findings.json")
        self.assertTrue(os.path.exists(out_path), msg=f"bin_findings.json not found at {out_path}")

        with open(out_path, encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertEqual(saved["run_id"], _RUN_ID)
        self.assertEqual(len(saved["findings"]), 1)

    # ------------------------------------------------------------------
    # test_handles_llm_failure_gracefully
    # ------------------------------------------------------------------

    def test_handles_llm_failure_gracefully(self) -> None:
        """LLM raises exception -> returns empty BinPassOutput, no crash."""
        chunks = [_make_chunk("src:0", has_obligations=True)]
        _setup_artifacts(self._tmp_dir, chunks, structured_ids=[])

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        logger = _make_logger()
        result = run_bin_pass(_RUN_ID, {}, mock_llm, logger)

        self.assertIsInstance(result, BinPassOutput)
        self.assertEqual(result.findings, [])
