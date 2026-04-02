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

import datetime
import json
import logging
import os
import re
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage

from .prompts import build_interpretation_prompt, build_reference_judge_prompt
from .schemas import BinPassOutput, InterpretationOutput, ObligationInterpretation
from .tools import (
    detect_ambiguous_terms,
    extract_references_from_text,
    fetch_cfr,
    get_section_family_chunks,
    lookup_definition,
    search_chunks_for_term,
)
from .utils import get_logger, parse_json_object

MAX_REFERENCE_DEPTH = 2


def _trace(artifact_dir: str, event: dict) -> None:
    """Append one structured event to trace.jsonl."""
    event["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    trace_path = os.path.join(artifact_dir, "trace.jsonl")
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


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

        context_bundle, section_id = _build_initial_context(obl, artifact_dir, logger)

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

        _trace(artifact_dir, {
            "event": "interpret_obligation_start",
            "obligation_id": obl_id,
            "anchor_chunks": obl.get("source_citations", []),
            "bin_findings_count": len(obl_findings),
            "section_id": section_id,
        })

        context_bundle = _resolve_references(obl, context_bundle, cheap_llm, logger)
        pass_number = 1
        interpretation = _interpret_obligation(obl, context_bundle, llm, logger, obligations)

        _trace(artifact_dir, {
            "event": "interpret_pass",
            "obligation_id": obl_id,
            "pass": pass_number,
            "confidence": interpretation.confidence_level,
            "lookup_requests": interpretation.lookup_requests,
            "needs_more_context": interpretation.needs_more_context,
        })

        # Merge ambiguous_terms into lookup queue -- LLM sometimes flags a term as ambiguous
        # in ambiguous_terms but forgets to also put it in lookup_requests. Treat both as the
        # same signal. Skip terms already looked up.
        already_looked_up = set(context_bundle["lookup_results"].keys())
        terms_to_lookup = list(interpretation.lookup_requests)
        for term in interpretation.ambiguous_terms:
            # Strip any explanation suffix -- LLM may use ": reason" or " - reason"
            clean = re.split(r"\s*[-:]\s+", term.strip("'\""), maxsplit=1)[0].strip("'\" ")
            if clean and clean.lower() not in {t.lower() for t in terms_to_lookup} \
                    and clean not in already_looked_up:
                terms_to_lookup.append(clean)

        # Term lookup loop: LLM signals which terms to look up, system fetches, re-interpret once
        if terms_to_lookup:
            logger.info(
                "    term lookups for %s: %s", obl_id, terms_to_lookup
            )
            found_any = False
            for term in terms_to_lookup:
                chunks = search_chunks_for_term(term, artifact_dir, top_n=5)
                chunk_ids = [c["src_id"] for c in chunks]
                _trace(artifact_dir, {
                    "event": "term_lookup",
                    "obligation_id": obl_id,
                    "term": term,
                    "chunks_found": chunk_ids,
                })
                if chunks:
                    context_bundle["lookup_results"][term] = [
                        f"[{c['heading']}]\n{c['text'][:1500]}" for c in chunks
                    ]
                    logger.info("    lookup %r: %d chunks found", term, len(chunks))
                    found_any = True
                else:
                    logger.info("    lookup %r: no matches", term)
            if found_any:
                pass_number += 1
                interpretation = _interpret_obligation(obl, context_bundle, llm, logger, obligations)
                _trace(artifact_dir, {
                    "event": "interpret_pass",
                    "obligation_id": obl_id,
                    "pass": pass_number,
                    "confidence": interpretation.confidence_level,
                    "lookup_requests": interpretation.lookup_requests,
                    "needs_more_context": interpretation.needs_more_context,
                })

        # Expansion pass: if LLM still needs more, add proposed/other chunks from section family
        if interpretation.needs_more_context and section_id:
            logger.info(
                "    expansion requested for %s -- fetching proposed+other chunks", obl_id
            )
            extra_chunks = get_section_family_chunks(
                section_id, artifact_dir, subsection_roles=["proposed", "other"]
            )
            if extra_chunks:
                extra_context = [
                    f"[{c['heading']}]\n{c['text'][:1500]}" for c in extra_chunks
                ]
                context_bundle["anchor_context"] = (
                    context_bundle["anchor_context"] + extra_context
                )
                _trace(artifact_dir, {
                    "event": "expand_context",
                    "obligation_id": obl_id,
                    "roles": ["proposed", "other"],
                    "chunks_added": len(extra_chunks),
                })
                logger.info(
                    "    expansion: added %d proposed/other chunks -- re-interpreting",
                    len(extra_chunks),
                )
                pass_number += 1
                interpretation = _interpret_obligation(obl, context_bundle, llm, logger, obligations)
                _trace(artifact_dir, {
                    "event": "interpret_pass",
                    "obligation_id": obl_id,
                    "pass": pass_number,
                    "confidence": interpretation.confidence_level,
                    "lookup_requests": interpretation.lookup_requests,
                    "needs_more_context": interpretation.needs_more_context,
                })
            else:
                logger.info("    expansion: no proposed/other chunks found for %s", obl_id)

        _trace(artifact_dir, {
            "event": "interpret_obligation_complete",
            "obligation_id": obl_id,
            "total_passes": pass_number,
            "final_confidence": interpretation.confidence_level,
        })

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

def _build_initial_context(
    obl: dict,
    artifact_dir: str,
    logger: logging.Logger,
) -> tuple[dict, Optional[str]]:
    """
    Build context bundle for one obligation.

    Sends all final-role chunks from the obligation's section as anchor context
    so the interpreter sees the complete adopted rule text for that section,
    not just the 1-2 chunks the extractor happened to cite.

    Returns (context_bundle, section_id).
    section_id is returned so the caller can run an expansion pass if needed.
    """
    obligation_text = obl.get("obligation_text", "")

    # Detect ambiguous terms and look up definitions
    terms = detect_ambiguous_terms(obligation_text)
    definitions = []
    for term in terms:
        defn = lookup_definition(term, artifact_dir)
        if defn:
            definitions.append(f"Definition of '{term}':\n{defn}")
            logger.debug("    definition found for %r", term)

    # Resolve source citation -> section_id
    section_id = _get_section_id_for_obligation(obl, artifact_dir)

    # Load all final-role chunks from the obligation's section so the interpreter
    # sees the full adopted rule text, not just the extractor-cited subset.
    # Falls back to extractor-cited chunks only if section_id is unavailable.
    if section_id:
        anchor_context = _load_section_final_chunks(section_id, artifact_dir)
        logger.info(
            "    initial context: %d definitions, %d section-final chunks (family=%s)",
            len(definitions), len(anchor_context), section_id,
        )
    else:
        anchor_context = _load_source_chunks(obl, artifact_dir)
        logger.info(
            "    initial context: %d definitions, %d anchor chunks (no section_id)",
            len(definitions), len(anchor_context),
        )

    bundle = {
        "definitions": definitions,
        "anchor_context": anchor_context,
        "discussion": [],                  # filled by bin findings in caller
        "cfr_texts": {},                   # filled by resolve_references
        "fetched_refs": set(),             # track already-fetched CFR refs
        "lookup_results": {},              # term -> passages, filled by term lookup loop
    }
    return bundle, section_id


def _load_source_chunks(obl: dict, artifact_dir: str) -> List[str]:
    """Load the chunks the extractor cited for this obligation as formatted strings."""
    source_citations = obl.get("source_citations", [])
    if not source_citations:
        return []

    chunks_path = os.path.join(artifact_dir, "chunks.json")
    if not os.path.exists(chunks_path):
        return []

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    src_map = {c.get("src_id"): c for c in chunks if "src_id" in c}
    results = []
    for cite in source_citations:
        chunk = src_map.get(cite)
        if chunk:
            heading = " > ".join(chunk.get("heading_path", []))
            results.append(f"[{heading}]\n{chunk.get('text', '')[:1500]}")
    return results


def _load_section_final_chunks(section_family: str, artifact_dir: str) -> List[str]:
    """Load all final-role chunks that share the same section_family as formatted strings."""
    chunks_path = os.path.join(artifact_dir, "chunks.json")
    if not os.path.exists(chunks_path):
        return []

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    results = []
    for chunk in chunks:
        if (chunk.get("section_family") == section_family
                and chunk.get("subsection_role") == "final"):
            heading = " > ".join(chunk.get("heading_path", []))
            results.append(f"[{heading}]\n{chunk.get('text', '')}")
    return results


def _get_section_id_for_obligation(obl: dict, artifact_dir: str) -> Optional[str]:
    """Map an obligation's source_citations to its section_family via chunks.json."""
    source_citations = obl.get("source_citations", [])
    if not source_citations:
        return None

    chunks_path = os.path.join(artifact_dir, "chunks.json")
    if not os.path.exists(chunks_path):
        return None

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    # Build src_id -> section_family map
    src_to_family = {c.get("src_id"): c.get("section_family") for c in chunks if "src_id" in c}

    # Use the first citation that has a family; fall back through all citations
    for cite in source_citations:
        family = src_to_family.get(cite)
        if family:
            return family
    return None


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
    sibling_obligations: list = None,
) -> ObligationInterpretation:
    """Run one LLM call to produce ObligationInterpretation from the context bundle."""
    prompt = build_interpretation_prompt(obl, context_bundle, sibling_obligations)

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
