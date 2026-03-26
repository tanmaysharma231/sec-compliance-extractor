"""
sec_interpreter/interpret_graph.py

Interpretation pipeline: for each extracted obligation, assembles a legal
context bundle then produces a structured interpretation.

Flow per obligation:
    build_initial_context  (no LLM: definition lookup + surrounding sections)
        |
    resolve_references     (agentic loop: fetch CFR + LLM judge, max_depth=2)
        |
    interpret_obligation   (one LLM call -> ObligationInterpretation)

Called by: InterpretModule.run()
Saves:     artifacts/{run_id}/interpretation.json
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage

from .prompts import build_interpretation_prompt, build_reference_judge_prompt, build_context_linker_prompt
from .schemas import BinPassOutput, InterpretationOutput, ObligationInterpretation, ObligationContextLinks
from .tools import (
    detect_ambiguous_terms,
    extract_references_from_text,
    fetch_cfr,
    get_surrounding_context,
    lookup_definition,
)
from .utils import get_logger, parse_json_object

MAX_REFERENCE_DEPTH = 2


def run_interpret_pipeline(
    run_id: str,
    llm: Any,
    cheap_llm: Any,
    logger: logging.Logger,
) -> InterpretationOutput:
    """
    Main entry point. Loads artifacts, processes each obligation, returns output.

    llm       -- main model (gpt-4o) for interpretation
    cheap_llm -- cheap model (gpt-4o-mini) for reference judge
    """
    artifact_dir = os.path.join("artifacts", run_id)

    # Load extracted obligations
    output_path = os.path.join(artifact_dir, "validated_output.json")
    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"No validated_output.json for run_id={run_id}. Run extract first."
        )
    with open(output_path, encoding="utf-8") as f:
        extracted = json.load(f)

    rule_title = extracted.get("rule_metadata", {}).get("rule_title", "Unknown Rule")
    obligations = extracted.get("key_obligations", [])

    logger.info(
        "Interpret: run_id=%s  rule=%r  obligations=%d",
        run_id, rule_title, len(obligations),
    )

    # Load bin findings if available (pre-linked context per obligation)
    bin_findings = []
    bin_findings_path = os.path.join(artifact_dir, "bin_findings.json")
    if os.path.exists(bin_findings_path):
        with open(bin_findings_path, encoding="utf-8") as f:
            bin_data = json.load(f)
        bin_output = BinPassOutput.model_validate(bin_data)
        bin_findings = bin_output.findings
        logger.info("Interpret: loaded %d bin findings", len(bin_findings))

    interpretations: List[ObligationInterpretation] = []

    for obl in obligations:
        obl_id = obl.get("obligation_id", "?")
        logger.info("  Processing %s: %s", obl_id, obl.get("obligation_text", "")[:60])

        context_bundle = _build_initial_context(obl, artifact_dir, logger)

        # Use pre-linked bin findings for discussion context
        obl_findings = [
            f for f in bin_findings
            if obl_id in f.related_to and f.finding_type != "not_relevant"
        ]
        if obl_findings:
            context_bundle["discussion"] = [
                f"[{f.finding_type}] {f.text}" for f in obl_findings
            ]
            logger.info("    bin findings: %d passages for %s", len(obl_findings), obl_id)

        context_bundle = _resolve_references(obl, context_bundle, cheap_llm, logger)
        interpretation = _interpret_obligation(obl, context_bundle, llm, logger)
        interpretations.append(interpretation)

    output = InterpretationOutput(
        run_id=run_id,
        rule_title=rule_title,
        interpretations=interpretations,
    )

    # Save artifact
    out_path = os.path.join(artifact_dir, "interpretation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output.model_dump(mode="json"), f, indent=2)
    logger.info("Interpretation saved to %s", out_path)

    return output


# ---------------------------------------------------------------------------
# Step 1: Build initial context (no LLM)
# ---------------------------------------------------------------------------

def _build_initial_context(obl: dict, artifact_dir: str, logger: logging.Logger) -> dict:
    """Look up definitions for ambiguous terms + pull surrounding sections."""
    obligation_text = obl.get("obligation_text", "")

    # Detect ambiguous terms and look up definitions
    terms = detect_ambiguous_terms(obligation_text)
    definitions = []
    for term in terms:
        defn = lookup_definition(term, artifact_dir)
        if defn:
            definitions.append(f"Definition of '{term}':\n{defn}")
            logger.debug("    definition found for %r", term)
        else:
            logger.debug("    no definition found for %r", term)

    # Get surrounding context using source_citations -> section_id
    surrounding = []
    section_id = _get_section_id_for_obligation(obl, artifact_dir)
    if section_id:
        surrounding = get_surrounding_context(section_id, artifact_dir, window=2)
        logger.debug("    surrounding context: %d sections", len(surrounding))

    logger.info(
        "    initial context: %d definitions, %d surrounding sections",
        len(definitions), len(surrounding),
    )

    return {
        "definitions": definitions,
        "surrounding": surrounding,
        "discussion": [],      # filled by bin findings lookup in per-obligation loop
        "cfr_texts": {},       # filled by resolve_references
        "fetched_refs": set(), # track what we've already fetched
    }


def _format_family_chunks(chunks: List[dict]) -> List[str]:
    """Format raw family chunk dicts into '[heading]\\ntext' strings for LLM prompts."""
    return [f"[{c.get('heading', '')}]\n{c.get('text', '')[:2000]}" for c in chunks]


def _link_family_context(
    obl: dict,
    family_chunks: List[dict],
    cheap_llm: Any,
    logger: logging.Logger,
) -> List[dict]:
    """
    Linker pass: ask cheap LLM to classify each family chunk as key | supporting | skip
    for this obligation. Returns filtered list (key + supporting only).

    Fallback: returns all chunks unchanged on LLM failure or empty keep set.
    """
    if not family_chunks:
        return []

    obl_id = obl.get("obligation_id", "?")
    obl_text = obl.get("obligation_text", "")
    prompt = build_context_linker_prompt(obl_text, obl_id, family_chunks)

    try:
        response = cheap_llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_json_object(raw)
        links = ObligationContextLinks.model_validate(parsed)

        keep_indices = set(links.key_indices + links.supporting_indices)
        # Clamp to valid range
        n = len(family_chunks)
        keep_indices = {i for i in keep_indices if 0 <= i < n}

        if not keep_indices:
            logger.warning(
                "    linker: all indices out of range for %s -- using all %d chunks",
                obl_id, n,
            )
            return family_chunks

        result = [family_chunks[i] for i in sorted(keep_indices)]
        logger.info(
            "    linker: %s  input=%d  key=%d  supporting=%d  skip=%d  kept=%d",
            obl_id, n,
            len(links.key_indices), len(links.supporting_indices),
            len(links.skip_indices), len(result),
        )
        return result

    except Exception as e:
        logger.warning(
            "    linker: failed for %s: %s -- using all %d chunks as fallback",
            obl_id, e, len(family_chunks),
        )
        return family_chunks


def _get_section_id_for_obligation(obl: dict, artifact_dir: str) -> Optional[str]:
    """Map an obligation's source_citations (src:N) to a section_id via chunks.json."""
    source_citations = obl.get("source_citations", [])
    if not source_citations:
        return None

    chunks_path = os.path.join(artifact_dir, "chunks.json")
    if not os.path.exists(chunks_path):
        return None

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    # Build src_id -> section_id map
    src_to_section = {c.get("src_id"): c.get("section_id") for c in chunks if "src_id" in c}

    # Use the first citation
    first_cite = source_citations[0]
    return src_to_section.get(first_cite)


# ---------------------------------------------------------------------------
# Step 2: Resolve references (agentic loop)
# ---------------------------------------------------------------------------

def _resolve_references(
    obl: dict,
    context_bundle: dict,
    cheap_llm: Any,
    logger: logging.Logger,
) -> dict:
    """
    Fetch CFR sections cited in the obligation, then let the LLM judge decide
    if deeper fetching is needed (max MAX_REFERENCE_DEPTH rounds).
    """
    cited_sections = obl.get("cited_sections", [])

    # Initial fetch: all cited_sections
    for ref in cited_sections:
        if ref not in context_bundle["fetched_refs"]:
            text = fetch_cfr(ref)
            if text:
                context_bundle["cfr_texts"][ref] = text
                context_bundle["fetched_refs"].add(ref)
                logger.info("    fetched CFR: %r (%d chars)", ref, len(text))

    # Agentic loop: ask judge if more fetching needed
    for depth in range(MAX_REFERENCE_DEPTH):
        if not context_bundle["cfr_texts"]:
            break  # nothing fetched yet, no point asking judge

        # Find all references visible in fetched text
        all_fetched_text = "\n\n".join(context_bundle["cfr_texts"].values())
        refs_seen = [
            r for r in extract_references_from_text(all_fetched_text)
            if r not in context_bundle["fetched_refs"]
        ]

        if not refs_seen:
            logger.debug("    judge: no new refs found in fetched text, stopping")
            break

        context_summary = _build_context_summary(context_bundle)
        prompt = build_reference_judge_prompt(
            obl.get("obligation_text", ""),
            context_summary,
            refs_seen,
        )

        try:
            response = cheap_llm.invoke([HumanMessage(content=prompt)])
            decision = (response.content if hasattr(response, "content") else str(response)).strip()
        except Exception as e:
            logger.warning("    judge call failed: %s -- stopping loop", e)
            break

        logger.info("    judge (depth=%d): %r", depth + 1, decision[:80])

        if decision.upper() == "SUFFICIENT" or decision.upper().startswith("SUFFICIENT"):
            break

        # Decision is a reference to fetch
        next_ref = decision.strip()
        if next_ref not in refs_seen:
            logger.warning("    judge returned ref not in refs_seen list: %r -- stopping", next_ref)
            break

        text = fetch_cfr(next_ref)
        if text:
            context_bundle["cfr_texts"][next_ref] = text
            context_bundle["fetched_refs"].add(next_ref)
            logger.info("    fetched additional CFR: %r (%d chars)", next_ref, len(text))
        else:
            logger.warning("    fetch failed for %r -- stopping loop", next_ref)
            break

    logger.info(
        "    resolve_references done: %d CFR sections fetched",
        len(context_bundle["cfr_texts"]),
    )
    return context_bundle


def _build_context_summary(context_bundle: dict) -> str:
    """Compact summary of what has been assembled so far."""
    parts = []
    if context_bundle.get("definitions"):
        parts.append("DEFINITIONS:\n" + "\n".join(context_bundle["definitions"])[:1000])
    if context_bundle.get("cfr_texts"):
        for ref, text in context_bundle["cfr_texts"].items():
            parts.append(f"CFR [{ref}]:\n{text[:800]}")
    return "\n\n".join(parts) or "(empty)"


# ---------------------------------------------------------------------------
# Step 3: Interpret obligation (one LLM call)
# ---------------------------------------------------------------------------

def _interpret_obligation(
    obl: dict,
    context_bundle: dict,
    llm: Any,
    logger: logging.Logger,
) -> ObligationInterpretation:
    """Run one LLM call to produce ObligationInterpretation from the context bundle."""
    prompt = build_interpretation_prompt(obl, context_bundle)

    fallback = ObligationInterpretation(
        obligation_id=obl.get("obligation_id", "OBL-???"),
        primary_interpretation=obl.get("obligation_text", ""),
        compliance_implication="Review this obligation with legal counsel.",
        confidence_level="low",
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_json_object(raw)
        result = ObligationInterpretation.model_validate(parsed)
        logger.info(
            "    interpretation done: confidence=%s  ambiguous=%s",
            result.confidence_level, result.ambiguous_terms,
        )
        return result
    except Exception as e:
        logger.warning("    interpretation failed for %s: %s -- using fallback", obl.get("obligation_id"), e)
        return fallback
