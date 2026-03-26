"""
sec_interpreter/bin_graph.py

Bin pass: cheap LLM secondary scan over flagged chunks not already in the
structured extraction set. Looks for missed obligations, scope modifiers,
definitions, edge cases, and implied requirements.

Runs AFTER primary extraction (extract_graph.py).

Flow:
    load chunks + structure_scan_result
        |
    filter to remaining (not in structured_chunk_ids)
        |
    filter to flagged (has_obligations OR has_codified_text OR has_scope)
        |
    build_bin_pass_prompt -> cheap_llm -> parse -> BinPassOutput
        |
    save bin_findings.json
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, List

from langchain_core.messages import HumanMessage

from .prompts import build_bin_pass_prompt
from .schemas import BinFinding, BinPassOutput, StructureScanResult
from .utils import parse_json_object


def run_bin_pass(
    run_id: str,
    extraction_output: dict,
    cheap_llm: Any,
    logger: logging.Logger,
) -> BinPassOutput:
    """
    Secondary reviewer pass over flagged chunks not sent to the main extractor.

    run_id           -- identifies the artifact directory (artifacts/{run_id})
    extraction_output -- validated_output dict with key_obligations list
    cheap_llm        -- cheap LLM (gpt-4o-mini style)
    logger           -- caller-supplied logger

    Returns BinPassOutput (empty findings list on any non-recoverable error).
    """
    artifact_dir = os.path.join("artifacts", run_id)

    # ------------------------------------------------------------------
    # 1. Load chunks.json
    # ------------------------------------------------------------------
    chunks_path = os.path.join(artifact_dir, "chunks.json")
    with open(chunks_path, encoding="utf-8") as fh:
        chunks: List[dict] = json.load(fh)

    # ------------------------------------------------------------------
    # 2. Load structure_scan_result.json
    # ------------------------------------------------------------------
    scan_path = os.path.join(artifact_dir, "structure_scan_result.json")
    if not os.path.exists(scan_path):
        logger.warning(
            "bin_pass: structure_scan_result.json not found for run_id=%s -- skipping", run_id
        )
        return BinPassOutput(run_id=run_id)

    with open(scan_path, encoding="utf-8") as fh:
        scan_data = json.load(fh)
    scan_result = StructureScanResult.model_validate(scan_data)

    # ------------------------------------------------------------------
    # 3. Compute remaining and flagged chunks
    # ------------------------------------------------------------------
    structured_ids = set(scan_result.structured_chunk_ids)
    remaining = [c for c in chunks if c.get("src_id") not in structured_ids]
    flagged = [
        c for c in remaining
        if c.get("has_obligations") or c.get("has_codified_text") or c.get("has_scope")
    ]

    if not flagged:
        logger.info(
            "bin_pass: no flagged chunks outside structured set for run_id=%s -- skipping", run_id
        )
        empty_output = BinPassOutput(run_id=run_id)
        _save_output(empty_output, artifact_dir)
        return empty_output

    # ------------------------------------------------------------------
    # 4. Call cheap LLM
    # ------------------------------------------------------------------
    known_obligations = extraction_output.get("key_obligations", [])
    prompt = build_bin_pass_prompt(flagged, known_obligations)

    try:
        response = cheap_llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.warning(
            "bin_pass: LLM call failed for run_id=%s: %s -- returning empty output", run_id, exc
        )
        return BinPassOutput(run_id=run_id)

    # ------------------------------------------------------------------
    # 5. Parse response
    # ------------------------------------------------------------------
    try:
        parsed = parse_json_object(raw)
    except Exception as exc:
        logger.warning(
            "bin_pass: failed to parse LLM response for run_id=%s: %s -- returning empty output",
            run_id, exc,
        )
        return BinPassOutput(run_id=run_id)

    findings_data = parsed.get("findings", [])
    findings: List[BinFinding] = []
    for f in findings_data:
        try:
            finding = BinFinding.model_validate(f)
            findings.append(finding)
        except Exception as exc:
            logger.warning("bin_pass: skipping invalid finding: %s -- %s", f, exc)

    # ------------------------------------------------------------------
    # 6. Log missed obligations as warnings
    # ------------------------------------------------------------------
    for finding in findings:
        if finding.finding_type == "missed_obligation":
            logger.warning(
                "    bin_pass: potential missed obligation found: %s", finding.text[:100]
            )

    # ------------------------------------------------------------------
    # 7. Build output + save
    # ------------------------------------------------------------------
    output = BinPassOutput(run_id=run_id, findings=findings)

    _save_output(output, artifact_dir)

    missed = sum(1 for f in findings if f.finding_type == "missed_obligation")
    scope = sum(1 for f in findings if f.finding_type == "scope_modifier")
    definition = sum(1 for f in findings if f.finding_type == "definition")
    edge_case = sum(1 for f in findings if f.finding_type == "edge_case")
    implied = sum(1 for f in findings if f.finding_type == "implied_requirement")

    logger.info(
        "bin_pass: %d findings (%d missed_obl, %d scope, %d def, %d edge, %d implied)",
        len(findings), missed, scope, definition, edge_case, implied,
    )

    return output


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_output(output: BinPassOutput, artifact_dir: str) -> None:
    """Write bin_findings.json to artifact_dir."""
    out_path = os.path.join(artifact_dir, "bin_findings.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output.model_dump(mode="json"), fh, indent=2)
