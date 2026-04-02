"""
sec_interpreter/extract_graph.py

Purpose: LangGraph pipeline for Stage 2 (ExtractGraph).
         Loads RichChunk artifacts, runs a deterministic structure_scan to select
         relevant chunks, then runs the full Extractor pass on selected chunks.

Old flow: load_chunks → [locator_pass] → extract_structured_fields → validate_output
                                       ↗ (retry)  ↖               → save_extract_artifacts
New flow: load_chunks → [structure_scan_pass] → extract_structured_fields → validate_output
                                              ↗ (retry)  ↖               → save_extract_artifacts

The structure_scan_pass is SKIPPED in direct/inline-text mode (RuleExtractorModule) to
preserve backward compatibility with tests that use FakeLLM.

Called by: ExtractModule.run(), RuleExtractorModule.run()
Calls:     prompts.*, schemas.*, utils.*
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .prompts import (
    build_extractor_prompt,
    build_locator_prompt,
    build_retry_user_prompt,
    build_section_extractor_prompt,
    build_system_prompt,
    build_user_prompt,
)
from .schemas import COMPLIANCE_CONTENT_TYPES, DocumentMap, LocatorSelection, RichChunk, RuleExtractorInput, RuleExtractorOutput, SectionExtractOutput, StructureScanResult
from .scorer import build_index_row
from .structure import gap_check, structure_scan
from .utils import (
    enforce_citation_bounds,
    enforce_obligation_links,
    enforce_safe_language,
    enforce_strict_citations,
    get_logger,
    parse_json_object,
    repair_json,
)

MAX_RETRIES = 2


class ExtractState(TypedDict, total=False):
    run_id: str                              # ties back to the ingest artifacts
    payload: RuleExtractorInput              # rule_text + strict_citations
    chunks: List[RichChunk]                  # full chunk list (for citation bounds)
    summary_text: str                        # loaded from artifacts
    scan_result: StructureScanResult         # structure_scan output (replaces locator_selection)
    locator_selection: LocatorSelection      # kept for backward compat artifact writing
    selected_chunks: List[RichChunk]         # chunks passed to Extractor
    skip_locator: bool                       # True in direct/inline-text mode
    raw_output: str
    output: Optional[RuleExtractorOutput]
    retry_count: int
    last_error: Optional[str]
    token_usage: dict                        # accumulated token counts across LLM calls
    section_partial_outputs: List[dict]      # raw dicts from each per-section LLM call


def build_extract_graph(llm: Any, logger: logging.Logger):
    structured_llm = _try_structured_output(llm, logger)

    # ------------------------------------------------------------------
    # Node: load_chunks
    # Supports two modes:
    #   1. run_id provided, payload=None → load RichChunk objects from artifacts
    #   2. payload.rule_text provided  → chunk in memory (direct/backward compat)
    # ------------------------------------------------------------------
    def load_chunks(state: ExtractState) -> ExtractState:
        run_id = state.get("run_id")
        payload = state.get("payload")

        if run_id and not payload:
            # ── Artifact mode: load from previous ingest run ──────────────
            artifact_dir = os.path.join("artifacts", run_id)
            chunks_path = os.path.join(artifact_dir, "chunks.json")
            input_path = os.path.join(artifact_dir, "input.txt")
            summary_path = os.path.join(artifact_dir, "summary.txt")

            if not os.path.exists(chunks_path):
                raise FileNotFoundError(
                    f"No chunks.json found for run_id={run_id}. "
                    "Run the ingest pipeline first."
                )

            with open(chunks_path, encoding="utf-8") as f:
                chunks_data = json.load(f)

            # Backward compat: old format had {"id": "src:N", "text": "..."}
            chunks: List[RichChunk] = []
            for entry in chunks_data:
                if "src_id" in entry:
                    # New RichChunk format
                    chunks.append(RichChunk.model_validate(entry))
                else:
                    # Old plain-string format — wrap in minimal RichChunk
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

            # Load rule_text for payload
            rule_text = ""
            if os.path.exists(input_path):
                with open(input_path, encoding="utf-8") as f:
                    rule_text = f.read()

            # Load summary text
            summary_text = ""
            if os.path.exists(summary_path):
                with open(summary_path, encoding="utf-8") as f:
                    summary_text = f.read().strip()

            strict_citations = False
            raw_strict = state.get("payload") or state.get("strict_citations", False)
            if isinstance(raw_strict, bool):
                strict_citations = raw_strict
            elif isinstance(raw_strict, RuleExtractorInput):
                strict_citations = raw_strict.strict_citations

            payload = RuleExtractorInput(rule_text=rule_text, strict_citations=strict_citations)
            logger.info(
                "Loaded %d chunks from run_id=%s artifacts (summary: %d chars)",
                len(chunks), run_id, len(summary_text),
            )
            return {
                "payload": payload,
                "chunks": chunks,
                "summary_text": summary_text,
                "skip_locator": state.get("skip_locator", False),
                "retry_count": 0,
                "last_error": None,
            }

        else:
            # ── Direct mode: chunk from payload.rule_text (backward compat) ──
            from .utils import chunk_rule_text
            raw_chunks = chunk_rule_text(payload.rule_text)
            chunks = [
                RichChunk(
                    src_id=f"src:{i}",
                    section_id="SEC-INLINE",
                    heading_path=["UNLABELED"],
                    chunk_index_in_section=i,
                    text=c,
                    char_len=len(c),
                    token_estimate=len(c) // 4,
                )
                for i, c in enumerate(raw_chunks)
            ]
            logger.info("Chunked rule_text into %d chunks (direct mode)", len(chunks))
            return {
                "chunks": chunks,
                "summary_text": "",
                "skip_locator": True,   # bypass locator in direct mode
                "retry_count": 0,
                "last_error": None,
            }

    # ------------------------------------------------------------------
    # Routing after load_chunks
    # ------------------------------------------------------------------
    def _route_after_load(state: ExtractState) -> Literal["structure_scan_pass", "extract_structured_fields"]:
        if state.get("skip_locator", False):
            return "extract_structured_fields"
        return "structure_scan_pass"

    # ------------------------------------------------------------------
    # Node: structure_scan_pass
    # Replaces the LLM-based locator with a deterministic heading scan.
    # Reads sections.json + chunks.json from artifact_dir to identify
    # lettered obligation sections and collect final/codified chunk IDs.
    # Falls back to all chunks if the scan finds nothing (e.g. old artifacts
    # without section_family / subsection_role populated).
    # ------------------------------------------------------------------
    def structure_scan_pass(state: ExtractState) -> ExtractState:
        run_id = state["run_id"]
        chunks = state["chunks"]
        artifact_dir = os.path.join("artifacts", run_id)

        src_index = {c.src_id: i for i, c in enumerate(chunks)}
        chunk_map = {c.src_id: c for c in chunks}

        # Run deterministic structure scan
        scan_result = structure_scan(artifact_dir)

        structured_ids = set(scan_result.structured_chunk_ids)

        if structured_ids:
            selected_chunks = [c for c in chunks if c.src_id in structured_ids]
            # Also include named section chunks (effective dates, scope, exemptions)
            for sid in scan_result.named_section_chunk_ids:
                if sid in chunk_map and sid not in structured_ids:
                    selected_chunks.append(chunk_map[sid])
                    structured_ids.add(sid)
            # Always include src:0 (cover page) -- has title, release number, effective date
            if "src:0" not in structured_ids and chunks:
                selected_chunks.append(chunks[0])
            # Preserve document order
            selected_chunks = sorted(
                selected_chunks,
                key=lambda c: src_index.get(c.src_id, 0),
            )
            logger.info(
                "structure_scan selected %d / %d chunks (%d obligation sections)",
                len(selected_chunks), len(chunks), len(scan_result.obligation_sections),
            )
        else:
            # Fallback: scan found nothing -- old artifacts without section_family/subsection_role
            logger.warning(
                "structure_scan returned no structured_chunk_ids for run_id=%s; "
                "falling back to all chunks",
                run_id,
            )
            selected_chunks = list(chunks)

        return {
            "scan_result": scan_result,
            "selected_chunks": selected_chunks,
            "token_usage": {"locator": {}},
        }

    # ------------------------------------------------------------------
    # Node: extract_sections_loop
    # Runs one LLM call per obligation section identified by structure_scan.
    # First section uses full RuleExtractorOutput schema (produces rule_metadata
    # and rule_summary). Subsequent sections use SectionExtractOutput (partial).
    # Prior obligations are passed as cross-section context to avoid re-extraction.
    # ------------------------------------------------------------------
    section_structured_llm = _try_section_structured_output(llm, logger)

    def extract_sections_loop(state: ExtractState) -> ExtractState:
        scan_result: StructureScanResult = state["scan_result"]
        selected_chunks: List[RichChunk] = state.get("selected_chunks") or state["chunks"]
        chunks = state["chunks"]
        summary_text = state.get("summary_text", "")
        payload = state["payload"]
        existing_usage = state.get("token_usage") or {}

        chunk_map = {c.src_id: c for c in selected_chunks}
        all_chunk_src_index = {c.src_id: i for i, c in enumerate(chunks)}

        prior_obligations: List[dict] = []
        partial_outputs: List[dict] = []
        section_token_usages: List[dict] = []

        for sec_idx, obl_section in enumerate(scan_result.obligation_sections):
            # Collect chunks for this section in document order
            sec_chunk_ids = set(obl_section.structured_chunk_ids)
            sec_chunks = sorted(
                [chunk_map[sid] for sid in sec_chunk_ids if sid in chunk_map],
                key=lambda c: all_chunk_src_index.get(c.src_id, 999999),
            )

            if not sec_chunks:
                logger.warning(
                    "section %s has no chunks in selected set -- skipping",
                    obl_section.section_letter,
                )
                continue

            obligation_id_start = len(prior_obligations) + 1
            section_heading = obl_section.heading

            system_prompt = build_system_prompt()

            if sec_idx == 0:
                # First section: use full schema so rule_metadata + rule_summary are extracted.
                # Augment with named section chunks (scope, effective dates) and src:0 (cover page).
                extra_ids = set(scan_result.named_section_chunk_ids) | {"src:0"}
                extra_chunks = sorted(
                    [c for c in selected_chunks if c.src_id in extra_ids and c.src_id not in sec_chunk_ids],
                    key=lambda c: all_chunk_src_index.get(c.src_id, 999999),
                )
                first_section_chunks = sorted(
                    sec_chunks + extra_chunks,
                    key=lambda c: all_chunk_src_index.get(c.src_id, 999999),
                )
                user_prompt = build_extractor_prompt(payload, first_section_chunks)
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

                raw_output = None
                usage = {}
                if structured_llm is not None:
                    try:
                        result = structured_llm.invoke(messages)
                        raw_response = result.get("raw")
                        usage = _extract_usage(raw_response) if raw_response else {}
                        parsed: RuleExtractorOutput = result.get("parsed")
                        if parsed is None:
                            raise ValueError(result.get("parsing_error") or "Structured output returned None")
                        raw_output = json.dumps(parsed.model_dump(mode="json"))
                    except Exception as exc:
                        logger.warning("Section 0 structured output failed (%s), falling back", exc)
                        raw_output = None

                if raw_output is None:
                    response = llm.invoke(messages)
                    usage = _extract_usage(response)
                    raw_output = _normalize_content(getattr(response, "content", response))

                partial_outputs.append({"_raw": raw_output, "_is_first": True})
                section_token_usages.append(usage)

                # Parse for prior_obligations update
                try:
                    raw_dict = parse_json_object(raw_output) if isinstance(raw_output, str) else {}
                    for obl in raw_dict.get("key_obligations", []):
                        prior_obligations.append({
                            "obligation_id": obl.get("obligation_id", ""),
                            "obligation_text": obl.get("obligation_text", "")[:80],
                        })
                except Exception:
                    pass

            else:
                # Subsequent sections: partial schema, prior obligations as context.
                user_prompt = build_section_extractor_prompt(
                    section_heading=section_heading,
                    section_chunks=sec_chunks,
                    prior_obligations=prior_obligations,
                    summary_text=summary_text,
                    obligation_id_start=obligation_id_start,
                    strict_citations=payload.strict_citations,
                )
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

                raw_output = None
                usage = {}
                if section_structured_llm is not None:
                    try:
                        result = section_structured_llm.invoke(messages)
                        raw_response = result.get("raw")
                        usage = _extract_usage(raw_response) if raw_response else {}
                        parsed_sec: SectionExtractOutput = result.get("parsed")
                        if parsed_sec is None:
                            raise ValueError(result.get("parsing_error") or "Structured output returned None")
                        raw_output = json.dumps(parsed_sec.model_dump(mode="json"))
                    except Exception as exc:
                        logger.warning("Section %s structured output failed (%s), falling back", obl_section.section_letter, exc)
                        raw_output = None

                if raw_output is None:
                    response = llm.invoke(messages)
                    usage = _extract_usage(response)
                    raw_output = _normalize_content(getattr(response, "content", response))

                partial_outputs.append({"_raw": raw_output, "_is_first": False})
                section_token_usages.append(usage)

                # Update prior_obligations
                try:
                    raw_dict = parse_json_object(raw_output) if isinstance(raw_output, str) else {}
                    for obl in raw_dict.get("key_obligations", []):
                        prior_obligations.append({
                            "obligation_id": obl.get("obligation_id", ""),
                            "obligation_text": obl.get("obligation_text", "")[:80],
                        })
                except Exception:
                    pass

            logger.info(
                "section %s done -- running total %d obligations",
                obl_section.section_letter,
                len(prior_obligations),
            )

        # Merge all partial outputs into one RuleExtractorOutput-shaped dict
        parsed_partials = []
        for p in partial_outputs:
            try:
                d = parse_json_object(p["_raw"])
                d["_is_first"] = p["_is_first"]
                parsed_partials.append(d)
            except Exception as exc:
                logger.warning("Failed to parse partial output: %s", exc)

        merged = _merge_section_outputs(parsed_partials, chunks)
        merged_json = json.dumps(merged)

        # Accumulate token usage
        combined_extractor_usage = {
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in section_token_usages),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in section_token_usages),
            "total_tokens": sum(u.get("total_tokens", 0) for u in section_token_usages),
        }

        return {
            "raw_output": merged_json,
            "output": None,
            "section_partial_outputs": partial_outputs,
            "token_usage": {**existing_usage, "extractor": combined_extractor_usage},
        }

    # ------------------------------------------------------------------
    # Node: extract_structured_fields
    # ------------------------------------------------------------------
    def extract_structured_fields(state: ExtractState) -> ExtractState:
        payload = state["payload"]
        chunks = state["chunks"]
        retry_count = state.get("retry_count", 0)
        last_error = state.get("last_error")

        # Use selected_chunks if locator ran; otherwise use all chunks
        selected_chunks: Optional[List[RichChunk]] = state.get("selected_chunks")
        if selected_chunks is None:
            selected_chunks = chunks

        system_prompt = build_system_prompt()
        if retry_count > 0 and last_error:
            logger.info("Retry %d/%d — injecting error into prompt", retry_count, MAX_RETRIES)
            # For retry, rebuild prompt with selected chunks
            user_prompt = _build_retry_prompt(payload, selected_chunks, last_error)
        else:
            user_prompt = build_extractor_prompt(payload, selected_chunks)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Structured output path (preferred when LLM supports it)
        # include_raw=True means invoke() returns {"raw": AIMessage, "parsed": model, "parsing_error": ...}
        existing_usage = state.get("token_usage") or {}
        if structured_llm is not None:
            try:
                result = structured_llm.invoke(messages)
                raw_response = result.get("raw")
                extractor_usage = _extract_usage(raw_response) if raw_response else {}
                output: RuleExtractorOutput = result.get("parsed")
                if output is None:
                    raise ValueError(result.get("parsing_error") or "Structured output parse returned None")
                raw_output = json.dumps(output.model_dump(mode="json"))
                return {
                    "raw_output": raw_output,
                    "output": output,
                    "token_usage": {**existing_usage, "extractor": extractor_usage},
                }
            except Exception as exc:
                logger.warning("Structured output failed (%s), falling back to manual parse", exc)

        # Manual parse fallback
        response = llm.invoke(messages)
        extractor_usage = _extract_usage(response)
        raw_output = _normalize_content(getattr(response, "content", response))
        return {
            "raw_output": raw_output,
            "output": None,
            "token_usage": {**existing_usage, "extractor": extractor_usage},
        }

    # ------------------------------------------------------------------
    # Node: validate_output
    # ------------------------------------------------------------------
    def validate_output(state: ExtractState) -> ExtractState:
        payload = state["payload"]
        chunks = state["chunks"]   # full chunk list for citation bounds
        raw_output = state.get("raw_output", "")
        pre_parsed: Optional[RuleExtractorOutput] = state.get("output")

        # Structured output already gave us a Pydantic object — just run business checks
        if pre_parsed is not None:
            try:
                enforce_citation_bounds(pre_parsed, len(chunks))
                enforce_strict_citations(pre_parsed, payload)
                enforce_obligation_links(pre_parsed)
                enforce_safe_language(pre_parsed)
                return {"output": pre_parsed, "last_error": None}
            except Exception as exc:
                logger.warning("Business rule check failed: %s", exc)
                return {"output": None, "last_error": str(exc)}

        # Manual parse path
        parsed = None
        last_exc = None
        for idx, candidate in enumerate([raw_output, repair_json(raw_output)]):
            try:
                parsed = parse_json_object(candidate)
                break
            except Exception as exc:
                last_exc = exc
                if idx == 0:
                    logger.warning("Initial JSON parse failed, attempting repair")

        if parsed is None:
            logger.error("Failed to parse model output:\n%s", raw_output)
            return {"output": None, "last_error": f"JSON parse failed: {last_exc}"}

        try:
            output = RuleExtractorOutput.model_validate(parsed)
            enforce_citation_bounds(output, len(chunks))
            enforce_strict_citations(output, payload)
            enforce_obligation_links(output)
            enforce_safe_language(output)
            return {"output": output, "last_error": None}
        except Exception as exc:
            logger.warning("Validation failed: %s", exc)
            return {"output": None, "last_error": str(exc)}

    # ------------------------------------------------------------------
    # Node: save_extract_artifacts
    # ------------------------------------------------------------------
    def save_extract_artifacts(state: ExtractState) -> ExtractState:
        run_id = state["run_id"]
        chunks = state["chunks"]
        raw_output = state.get("raw_output", "")
        output = state["output"]
        selected_chunks = state.get("selected_chunks") or chunks
        scan_result: Optional[StructureScanResult] = state.get("scan_result")

        artifact_dir = os.path.join("artifacts", run_id)
        os.makedirs(artifact_dir, exist_ok=True)

        _write_file(artifact_dir, "raw_model_output.txt", raw_output)
        _write_file(
            artifact_dir,
            "validated_output.json",
            json.dumps(output.model_dump(mode="json"), indent=2),
        )

        # Run gap_check if we have a scan_result (structure_scan_pass ran)
        if scan_result is not None:
            output_dict = output.model_dump(mode="json")
            gap_report = gap_check(output_dict, scan_result, logger)
            _write_file(
                artifact_dir,
                "gap_report.json",
                json.dumps(gap_report, indent=2),
            )
            if gap_report["count_gap"] > 0 or gap_report["flagged_sections"]:
                logger.warning(
                    "gap_report: count_gap=%d, flagged_sections=%d",
                    gap_report["count_gap"],
                    len(gap_report["flagged_sections"]),
                )

        model_name = os.getenv("SEC_INTERPRETER_MODEL", "DeterministicLLM")
        timestamp = datetime.now(timezone.utc).isoformat()
        retries = state.get("retry_count", 0)

        token_usage = state.get("token_usage") or {}
        locator_usage = token_usage.get("locator", {})
        extractor_usage = token_usage.get("extractor", {})
        grand_total = locator_usage.get("total_tokens", 0) + extractor_usage.get("total_tokens", 0)

        structure_scan_used = scan_result is not None
        section_partial_outputs = state.get("section_partial_outputs") or []
        section_call_count = len(section_partial_outputs) if section_partial_outputs else 0
        log = (
            f"run_id: {run_id}\n"
            f"model: {model_name}\n"
            f"total_chunk_count: {len(chunks)}\n"
            f"selected_chunk_count: {len(selected_chunks)}\n"
            f"retries: {retries}\n"
            f"timestamp: {timestamp}\n"
            f"validation_result: success\n"
            f"structure_scan_used: {structure_scan_used}\n"
            f"section_call_count: {section_call_count}\n"
            f"extractor_prompt_tokens: {extractor_usage.get('prompt_tokens', 0)}\n"
            f"extractor_completion_tokens: {extractor_usage.get('completion_tokens', 0)}\n"
            f"extractor_total_tokens: {extractor_usage.get('total_tokens', 0)}\n"
            f"grand_total_tokens: {grand_total}\n"
        )
        _write_file(artifact_dir, "run_log.txt", log)

        # trace.jsonl -- structured events
        trace_events = [
            {
                "event": "extract_complete",
                "timestamp": timestamp,
                "total_chunks": len(chunks),
                "selected_chunks": len(selected_chunks),
                "retries": retries,
                "model": model_name,
                "structure_scan_used": structure_scan_used,
                "section_call_count": section_call_count,
                "token_usage": {
                    "extractor": extractor_usage,
                    "grand_total_tokens": grand_total,
                },
            }
        ]
        _write_file(
            artifact_dir,
            "trace.jsonl",
            "\n".join(json.dumps(e) for e in trace_events) + "\n",
        )

        logger.info("Extract artifacts written to %s/", artifact_dir)
        return {}

    # ------------------------------------------------------------------
    # Routing after validate_output
    # ------------------------------------------------------------------
    def _route_after_validation(
        state: ExtractState,
    ) -> Literal["save_extract_artifacts", "increment_retry"]:
        if state.get("output") is not None:
            return "save_extract_artifacts"
        retry_count = state.get("retry_count", 0)
        if retry_count < MAX_RETRIES:
            return "increment_retry"
        error = state.get("last_error", "Unknown error")
        raise ValueError(f"Extraction failed after {MAX_RETRIES} retries. Last error: {error}")

    def increment_retry(state: ExtractState) -> ExtractState:
        return {"retry_count": state.get("retry_count", 0) + 1}

    # ------------------------------------------------------------------
    # Routing after structure_scan_pass
    # ------------------------------------------------------------------
    def _route_after_scan(state: ExtractState) -> Literal["extract_sections_loop", "extract_structured_fields"]:
        scan = state.get("scan_result")
        if scan and scan.obligation_sections:
            return "extract_sections_loop"
        return "extract_structured_fields"

    # ------------------------------------------------------------------
    # Routing after increment_retry -- route back to same extractor path
    # ------------------------------------------------------------------
    def _route_retry(state: ExtractState) -> Literal["extract_sections_loop", "extract_structured_fields"]:
        if state.get("section_partial_outputs") is not None:
            return "extract_sections_loop"
        return "extract_structured_fields"

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    graph = StateGraph(ExtractState)
    graph.add_node("load_chunks", load_chunks)
    graph.add_node("structure_scan_pass", structure_scan_pass)
    graph.add_node("extract_sections_loop", extract_sections_loop)
    graph.add_node("extract_structured_fields", extract_structured_fields)
    graph.add_node("validate_output", validate_output)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("save_extract_artifacts", save_extract_artifacts)

    graph.add_edge(START, "load_chunks")
    graph.add_conditional_edges(
        "load_chunks",
        _route_after_load,
        {
            "structure_scan_pass": "structure_scan_pass",
            "extract_structured_fields": "extract_structured_fields",
        },
    )
    graph.add_conditional_edges(
        "structure_scan_pass",
        _route_after_scan,
        {
            "extract_sections_loop": "extract_sections_loop",
            "extract_structured_fields": "extract_structured_fields",
        },
    )
    graph.add_edge("extract_sections_loop", "validate_output")
    graph.add_edge("extract_structured_fields", "validate_output")
    graph.add_conditional_edges(
        "validate_output",
        _route_after_validation,
        {
            "save_extract_artifacts": "save_extract_artifacts",
            "increment_retry": "increment_retry",
        },
    )
    graph.add_conditional_edges(
        "increment_retry",
        _route_retry,
        {
            "extract_sections_loop": "extract_sections_loop",
            "extract_structured_fields": "extract_structured_fields",
        },
    )
    graph.add_edge("save_extract_artifacts", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _union_src_ids(selection: LocatorSelection) -> List[str]:
    """Deduplicated union of all src_ids from a LocatorSelection, order preserved."""
    all_ids = (
        selection.date_chunks
        + selection.scope_chunks
        + selection.obligation_chunks
        + selection.definition_chunks
        + selection.other_key_chunks
    )
    seen: set[str] = set()
    result: List[str] = []
    for sid in all_ids:
        if sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


def _build_retry_prompt(
    payload: RuleExtractorInput,
    selected_chunks: List[RichChunk],
    error: str,
) -> str:
    """Retry variant of extractor prompt — embeds previous validation error."""
    base = build_extractor_prompt(payload, selected_chunks)
    return (
        f"IMPORTANT: Your previous response failed validation with this error:\n"
        f"  {error}\n\n"
        "Fix the issue and try again. Pay close attention to:\n"
        "- citation indices must reference [src:N] chunks that exist above\n"
        "- obligation_id values must be OBL-001, OBL-002, etc.\n"
        "- compliance_impact_areas.area must be one of the valid values in the schema\n"
        "- do not use forbidden terms\n\n"
        + base
    )


def _extract_usage(response: Any) -> dict:
    """Extract token usage counts from a LangChain AIMessage response_metadata."""
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _try_structured_output(llm: Any, logger: logging.Logger) -> Any | None:
    if not hasattr(llm, "with_structured_output"):
        return None
    try:
        wrapped = llm.with_structured_output(RuleExtractorOutput, include_raw=True)
        logger.info("Structured output enabled for %s", type(llm).__name__)
        return wrapped
    except Exception as exc:
        logger.warning("with_structured_output not available: %s", exc)
        return None


def _try_section_structured_output(llm: Any, logger: logging.Logger) -> Any | None:
    if not hasattr(llm, "with_structured_output"):
        return None
    try:
        wrapped = llm.with_structured_output(SectionExtractOutput, include_raw=True)
        logger.info("Section structured output enabled for %s", type(llm).__name__)
        return wrapped
    except Exception as exc:
        logger.warning("with_structured_output (section) not available: %s", exc)
        return None


def _src_index(src_id: str) -> int:
    try:
        return int(src_id.split(":")[1])
    except (IndexError, ValueError):
        return 999999


def _renumber_obligations(obligations: List[dict]) -> tuple:
    """Returns (renumbered_list, old_to_new_id_map)."""
    old_to_new: dict = {}
    renumbered = []
    for i, obl in enumerate(obligations, start=1):
        new_id = f"OBL-{i:03d}"
        old_id = obl.get("obligation_id", new_id)
        old_to_new[old_id] = new_id
        updated = dict(obl)
        updated["obligation_id"] = new_id
        renumbered.append(updated)
    return renumbered, old_to_new


def _merge_section_outputs(partial_outputs: List[dict], all_chunks: List[RichChunk]) -> dict:
    """Merge per-section partial outputs into one RuleExtractorOutput-shaped dict.

    partial_outputs -- list of parsed dicts, each with _is_first flag.
    all_chunks      -- full chunk list (unused here, kept for future extension).
    """
    if not partial_outputs:
        return {
            "rule_metadata": {"rule_title": "", "release_number": None, "publication_date": None, "effective_date": None, "citations": []},
            "rule_summary": {"summary": "", "citations": []},
            "key_obligations": [],
            "affected_entity_types": [],
            "compliance_impact_areas": [],
            "assumptions": [],
        }

    # rule_metadata and rule_summary come from the first section (full schema call)
    first = partial_outputs[0]
    rule_metadata = first.get("rule_metadata") or {
        "rule_title": "", "release_number": None, "publication_date": None, "effective_date": None, "citations": []
    }
    rule_summary = first.get("rule_summary") or {"summary": "", "citations": []}

    # Collect all obligations in section order
    all_obligations = []
    for p in partial_outputs:
        all_obligations.extend(p.get("key_obligations") or [])

    renumbered_obls, old_to_new = _renumber_obligations(all_obligations)

    # Deduplicate affected_entity_types by entity_type
    seen_entities: set = set()
    merged_entities = []
    for p in partial_outputs:
        for ent in (p.get("affected_entity_types") or []):
            key = ent.get("entity_type", "")
            if key not in seen_entities:
                seen_entities.add(key)
                merged_entities.append(ent)

    # Merge compliance_impact_areas by area, applying id renumber map
    area_map: dict = {}
    for p in partial_outputs:
        for area_entry in (p.get("compliance_impact_areas") or []):
            area = area_entry.get("area", "")
            if area not in area_map:
                area_map[area] = {
                    "area": area,
                    "linked_obligation_ids": [],
                    "citations": [],
                }
            # Remap linked IDs
            for old_id in (area_entry.get("linked_obligation_ids") or []):
                new_id = old_to_new.get(old_id, old_id)
                if new_id not in area_map[area]["linked_obligation_ids"]:
                    area_map[area]["linked_obligation_ids"].append(new_id)
            for cite in (area_entry.get("citations") or []):
                if cite not in area_map[area]["citations"]:
                    area_map[area]["citations"].append(cite)
    merged_areas = list(area_map.values())

    # Concatenate assumptions without dedup
    all_assumptions = []
    for p in partial_outputs:
        all_assumptions.extend(p.get("assumptions") or [])

    return {
        "rule_metadata": rule_metadata,
        "rule_summary": rule_summary,
        "key_obligations": renumbered_obls,
        "affected_entity_types": merged_entities,
        "compliance_impact_areas": merged_areas,
        "assumptions": all_assumptions,
    }


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content)
    return str(content)


def _write_file(directory: str, filename: str, content: str) -> None:
    with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
        f.write(content)
