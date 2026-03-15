"""
sec_interpreter/comprehend.py

Document Comprehension Control Run -- standalone calibration tool.

Reads all chunks from an existing ingest run_id, processes each chunk with an
LLM to classify it, then synthesises a document-level understanding.  The
result is saved as artifacts/{run_id}/control.json and a locator comparison
is printed to stdout.

Usage:
    python -m sec_interpreter.cli comprehend --run-id <run_id>
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, List, Set

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from .extract_graph import _extract_usage, _normalize_content
from .module import DeterministicLLM, _load_env_llm
from .schemas import RichChunk
from .utils import get_logger, parse_json_object, repair_json

load_dotenv()
logger = get_logger("sec_interpreter.comprehend")

# ---------------------------------------------------------------------------
# Valid content types for the per-chunk classifier
# ---------------------------------------------------------------------------
VALID_CONTENT_TYPES = {
    "final_rule_text",
    "obligation",
    "definition",
    "commentary",
    "comments",
    "economic_analysis",
    "procedural",
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _chunk_classify_prompt(chunk: RichChunk) -> str:
    heading = " > ".join(chunk.heading_path) if chunk.heading_path else "(no heading)"
    return (
        "You are reviewing a chunk from an SEC regulatory document.\n\n"
        f"Chunk ID: {chunk.src_id}\n"
        f"Section heading: {heading}\n"
        f"Char length: {chunk.char_len}\n\n"
        "---BEGIN CHUNK TEXT---\n"
        f"{chunk.text}\n"
        "---END CHUNK TEXT---\n\n"
        "Respond with a JSON object with exactly these fields:\n"
        '  "content_type": one of "final_rule_text", "obligation", "definition", '
        '"commentary", "comments", "economic_analysis", "procedural"\n'
        '  "summary": 2-3 sentence plain-English description of what this chunk contains\n'
        '  "important": true if this chunk is important for compliance extraction '
        "(i.e. it contains codified rule text, obligations, definitions, or key dates), "
        "false otherwise\n\n"
        "Return ONLY the JSON object, no other text."
    )


def _synthesis_prompt(chunk_summaries: List[dict], document_summary: str) -> str:
    summaries_text = json.dumps(chunk_summaries, indent=2)
    doc_summary_section = (
        f"Document summary (from ingest pipeline):\n{document_summary}\n\n"
        if document_summary
        else ""
    )
    return (
        "You are a compliance analyst reviewing an SEC regulatory document.\n"
        "You have been given per-chunk summaries of the full document.\n\n"
        + doc_summary_section
        + "Per-chunk summaries:\n"
        + summaries_text
        + "\n\n"
        "Based on the above, respond with a JSON object with exactly these fields:\n"
        '  "regulatory_objective": 1-2 sentence description of what this regulation achieves\n'
        '  "primary_obligations": list of strings, the main obligations imposed by this rule\n'
        '  "scope": which entities or activities are covered (as a string)\n'
        '  "effective_dates": key compliance dates (as a string)\n'
        '  "governance_notes": any notable governance or implementation requirements\n'
        '  "important_chunks": list of src_ids (e.g. ["src:8", "src:151"]) that are '
        "critical for compliance extraction -- include chunks with codified rule text, "
        "obligations, definitions, and key dates\n\n"
        "Return ONLY the JSON object, no other text."
    )


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------

def run_comprehend(run_id: str) -> None:
    """Run the two-pass comprehension control and save control.json."""
    artifact_dir = os.path.join("artifacts", run_id)
    chunks_path = os.path.join(artifact_dir, "chunks.json")
    summary_path = os.path.join(artifact_dir, "summary.txt")
    locator_path = os.path.join(artifact_dir, "locator_selection.json")
    control_path = os.path.join(artifact_dir, "control.json")

    # -- Load chunks ----------------------------------------------------------
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"No chunks.json found for run_id={run_id}. Run ingest first."
        )

    with open(chunks_path, encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks: List[RichChunk] = []
    for entry in chunks_data:
        if "src_id" in entry:
            chunks.append(RichChunk.model_validate(entry))
        else:
            # Old plain-string format -- wrap in minimal RichChunk
            text = entry.get("text", "")
            src_id = entry.get("id", f"src:{len(chunks)}")
            idx = len(chunks)
            chunks.append(
                RichChunk(
                    src_id=src_id,
                    section_id="SEC-LEGACY",
                    heading_path=["UNLABELED"],
                    chunk_index_in_section=idx,
                    text=text,
                    char_len=len(text),
                    token_estimate=len(text) // 4,
                )
            )

    logger.info("Loaded %d chunks from run_id=%s", len(chunks), run_id)

    # -- Load document summary ------------------------------------------------
    document_summary = ""
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            document_summary = f.read().strip()

    # -- Load locator selection (for comparison) ------------------------------
    locator_selected_ids: Set[str] = set()
    if os.path.exists(locator_path):
        with open(locator_path, encoding="utf-8") as f:
            loc_data = json.load(f)
        for key in ("date_chunks", "scope_chunks", "obligation_chunks",
                    "definition_chunks", "other_key_chunks"):
            locator_selected_ids.update(loc_data.get(key, []))
        logger.info("Loaded locator selection: %d unique chunks", len(locator_selected_ids))
    else:
        logger.info("No locator_selection.json found -- skipping locator comparison")

    # -- Init LLM -------------------------------------------------------------
    llm = _load_env_llm() or DeterministicLLM()
    model_name = os.getenv("SEC_INTERPRETER_MODEL", "DeterministicLLM")
    logger.info("Using model: %s", model_name)

    # -- Pass 1: per-chunk classification -------------------------------------
    print(f"Pass 1: classifying {len(chunks)} chunks...")
    chunk_summaries: List[dict] = []
    per_chunk_total_tokens = 0

    for i, chunk in enumerate(chunks):
        prompt = _chunk_classify_prompt(chunk)
        response = llm.invoke([HumanMessage(content=prompt)])
        usage = _extract_usage(response)
        per_chunk_total_tokens += usage.get("total_tokens", 0)

        raw = _normalize_content(getattr(response, "content", response))
        summary_entry = _parse_chunk_summary(chunk, raw)
        chunk_summaries.append(summary_entry)

        if (i + 1) % 20 == 0 or (i + 1) == len(chunks):
            print(f"  Processed {i + 1}/{len(chunks)} chunks...")

    logger.info("Pass 1 complete. Total tokens: %d", per_chunk_total_tokens)

    # -- Pass 2: document synthesis -------------------------------------------
    print("Pass 2: synthesising document understanding...")
    synth_prompt = _synthesis_prompt(chunk_summaries, document_summary)
    response = llm.invoke([HumanMessage(content=synth_prompt)])
    synthesis_usage = _extract_usage(response)
    synthesis_tokens = synthesis_usage.get("total_tokens", 0)

    raw_synthesis = _normalize_content(getattr(response, "content", response))
    document_understanding = _parse_synthesis(raw_synthesis)
    logger.info("Pass 2 complete. Synthesis tokens: %d", synthesis_tokens)

    # -- Assemble and save control.json ---------------------------------------
    timestamp = datetime.now(timezone.utc).isoformat()
    grand_total = per_chunk_total_tokens + synthesis_tokens

    control = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model": model_name,
        "chunk_count": len(chunks),
        "chunk_summaries": chunk_summaries,
        "document_understanding": document_understanding,
        "token_usage": {
            "per_chunk_total": per_chunk_total_tokens,
            "synthesis": synthesis_tokens,
            "grand_total": grand_total,
        },
    }

    with open(control_path, "w", encoding="utf-8") as f:
        json.dump(control, f, indent=2)

    print(f"control.json saved to {control_path}")

    # -- Print locator comparison ---------------------------------------------
    _print_comparison(document_understanding, locator_selected_ids)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_chunk_summary(chunk: RichChunk, raw: str) -> dict:
    """Parse per-chunk LLM response, falling back to safe defaults on error."""
    parsed: Any = None
    for candidate in [raw, repair_json(raw)]:
        try:
            parsed = parse_json_object(candidate)
            break
        except Exception:
            continue

    if parsed is None:
        logger.warning("Failed to parse chunk summary for %s", chunk.src_id)
        return {
            "src_id": chunk.src_id,
            "section_id": chunk.section_id,
            "heading_path": chunk.heading_path,
            "content_type": "procedural",
            "important": False,
            "summary": "(parse error)",
        }

    content_type = parsed.get("content_type", "procedural")
    if content_type not in VALID_CONTENT_TYPES:
        content_type = "procedural"

    return {
        "src_id": chunk.src_id,
        "section_id": chunk.section_id,
        "heading_path": chunk.heading_path,
        "content_type": content_type,
        "important": bool(parsed.get("important", False)),
        "summary": str(parsed.get("summary", ""))[:500],
    }


def _parse_synthesis(raw: str) -> dict:
    """Parse synthesis LLM response, returning best-effort dict on error."""
    parsed: Any = None
    for candidate in [raw, repair_json(raw)]:
        try:
            parsed = parse_json_object(candidate)
            break
        except Exception:
            continue

    if parsed is None:
        logger.warning("Failed to parse synthesis response")
        return {
            "regulatory_objective": "(parse error)",
            "primary_obligations": [],
            "scope": "",
            "effective_dates": "",
            "governance_notes": "",
            "important_chunks": [],
        }

    # Normalise important_chunks to a list of strings
    raw_ic = parsed.get("important_chunks", [])
    if not isinstance(raw_ic, list):
        raw_ic = []
    important_chunks = [str(x) for x in raw_ic if x]

    return {
        "regulatory_objective": str(parsed.get("regulatory_objective", "")),
        "primary_obligations": [str(x) for x in parsed.get("primary_obligations", [])],
        "scope": str(parsed.get("scope", "")),
        "effective_dates": str(parsed.get("effective_dates", "")),
        "governance_notes": str(parsed.get("governance_notes", "")),
        "important_chunks": important_chunks,
    }


# ---------------------------------------------------------------------------
# Comparison output
# ---------------------------------------------------------------------------

def _print_comparison(
    document_understanding: dict,
    locator_selected_ids: Set[str],
) -> None:
    """Print locator vs control comparison to stdout."""
    control_important = set(document_understanding.get("important_chunks", []))

    print("")
    print("=" * 60)
    print("LOCATOR vs CONTROL")
    print("=" * 60)

    if not locator_selected_ids:
        print("No locator_selection.json found -- cannot compare.")
        print(f"Control identified {len(control_important)} important chunks.")
        return

    overlap = control_important & locator_selected_ids
    missed = control_important - locator_selected_ids
    false_positives = locator_selected_ids - control_important

    print(f"Control identified   : {len(control_important)} important chunks")
    print(f"Locator selected     : {len(locator_selected_ids)} chunks")
    print(f"Overlap              : {len(overlap)} chunks (both agree)")

    if missed:
        missed_sorted = _sort_src_ids(missed)
        print(f"Missed by locator    : {', '.join(missed_sorted)}")
    else:
        print("Missed by locator    : none")

    if false_positives:
        fp_sorted = _sort_src_ids(false_positives)
        print(f"Locator false positives: {', '.join(fp_sorted)}")
    else:
        print("Locator false positives: none")

    if control_important:
        coverage = len(overlap) / len(control_important)
        print(
            f"Coverage score       : {len(overlap)} / {len(control_important)} "
            f"({coverage:.0%})"
        )
    else:
        print("Coverage score       : N/A (control found no important chunks)")

    print("=" * 60)


def _sort_src_ids(ids: Set[str]) -> List[str]:
    """Sort src_ids numerically by the integer after the colon."""
    def _key(s: str) -> int:
        try:
            return int(s.split(":")[1])
        except (IndexError, ValueError):
            return 0
    return sorted(ids, key=_key)
