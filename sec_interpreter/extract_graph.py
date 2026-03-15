"""
sec_interpreter/extract_graph.py

Purpose: LangGraph pipeline for Stage 2 (ExtractGraph) — two LLM calls.
         Loads RichChunk artifacts, runs a cheap Locator pass to select relevant
         chunks, then runs the full Extractor pass on only selected chunks.

Old flow: load_chunks → extract_structured_fields → validate_output
                      ↗ (retry)  ↖                           → save_extract_artifacts
New flow: load_chunks → [locator_pass] → extract_structured_fields → validate_output
                                       ↗ (retry)  ↖               → save_extract_artifacts

The locator_pass is SKIPPED in direct/inline-text mode (RuleExtractorModule) to
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
    build_system_prompt,
    build_user_prompt,
)
from .schemas import COMPLIANCE_CONTENT_TYPES, DocumentMap, LocatorSelection, RichChunk, RuleExtractorInput, RuleExtractorOutput
from .scorer import build_index_row
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
    summary_text: str                        # NEW: loaded from artifacts
    locator_selection: LocatorSelection      # NEW: Locator LLM output
    selected_chunks: List[RichChunk]         # NEW: chunks passed to Extractor
    skip_locator: bool                       # NEW: True in direct/inline-text mode
    raw_output: str
    output: Optional[RuleExtractorOutput]
    retry_count: int
    last_error: Optional[str]
    token_usage: dict                        # accumulated token counts across LLM calls


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
    def _route_after_load(state: ExtractState) -> Literal["locator_pass", "extract_structured_fields"]:
        if state.get("skip_locator", False):
            return "extract_structured_fields"
        return "locator_pass"

    # ------------------------------------------------------------------
    # Node: locator_pass
    # If chunks have content_type populated (classify has been run), skip
    # the LLM locator entirely and filter deterministically by content type.
    # Falls back to LLM locator when content_type is not available.
    # ------------------------------------------------------------------
    def locator_pass(state: ExtractState) -> ExtractState:
        chunks = state["chunks"]
        summary_text = state.get("summary_text", "")

        # Check if classify has been run (any chunk has non-empty content_type)
        classified_chunks = [c for c in chunks if c.content_type]
        if classified_chunks:
            run_id = state["run_id"]
            src_index = {c.src_id: i for i, c in enumerate(chunks)}

            # Prefer document_map retrieval: section-level, deterministic section_id lists
            doc_map_path = os.path.join("artifacts", run_id, "document_map.json")
            if os.path.exists(doc_map_path):
                with open(doc_map_path, encoding="utf-8") as f:
                    doc_map = DocumentMap.model_validate(json.load(f))
                target_section_ids = set(doc_map.compliance_section_ids)
                selected_chunks = [c for c in chunks if c.section_id in target_section_ids]
                logger.info(
                    "document_map retrieval: %d / %d chunks selected (%d compliance sections)",
                    len(selected_chunks), len(chunks), len(target_section_ids),
                )
            else:
                # Fallback: content_type filter direct on chunks (no document_map)
                selected_chunks = [c for c in chunks if c.content_type in COMPLIANCE_CONTENT_TYPES]
                logger.info(
                    "content_type filter: %d / %d chunks selected (no document_map found)",
                    len(selected_chunks), len(chunks),
                )

            # Always include src:0 (cover page) -- has title, release number, effective date
            selected_ids = {c.src_id for c in selected_chunks}
            if "src:0" not in selected_ids and chunks:
                selected_chunks.append(chunks[0])
                logger.info("Prepended src:0 (cover page) for metadata extraction")

            # Preserve original order
            selected_chunks = sorted(selected_chunks, key=lambda c: src_index.get(c.src_id, 0))

            # Build a synthetic LocatorSelection for artifact compatibility
            obl_ids = [c.src_id for c in selected_chunks
                       if c.content_type in {"final_rule_text", "obligation"}]
            def_ids = [c.src_id for c in selected_chunks if c.content_type == "definition"]
            selection = LocatorSelection(obligation_chunks=obl_ids, definition_chunks=def_ids)

            artifact_dir = os.path.join("artifacts", run_id)
            os.makedirs(artifact_dir, exist_ok=True)
            _write_file(
                artifact_dir,
                "locator_selection.json",
                json.dumps(selection.model_dump(mode="json"), indent=2),
            )

            return {
                "locator_selection": selection,
                "selected_chunks": selected_chunks,
                "token_usage": {"locator": {}},
            }

        # -----------------------------------------------------------
        # Fallback: LLM-based locator (classify not run yet)
        # -----------------------------------------------------------
        # Build compact index table
        index_rows = [build_index_row(c) for c in chunks]

        # Build chunk lookup map
        chunk_map = {c.src_id: c for c in chunks}

        # Call Locator LLM
        locator_prompt = build_locator_prompt(summary_text, index_rows)
        messages = [HumanMessage(content=locator_prompt)]

        response = llm.invoke(messages)
        locator_usage = _extract_usage(response)
        raw = _normalize_content(getattr(response, "content", response))

        # Parse LocatorSelection
        try:
            parsed = parse_json_object(raw)
            selection = LocatorSelection.model_validate(parsed)
        except Exception as exc:
            logger.warning("Locator parse failed (%s); selecting all obligation-flagged chunks", exc)
            obl_ids = [c.src_id for c in chunks if c.has_obligations][:60]
            selection = LocatorSelection(obligation_chunks=obl_ids or [chunks[0].src_id])

        # Force-inject all codified chunks into obligation_chunks (deterministic, no LLM needed)
        codified_ids = [c.src_id for c in chunks if c.has_codified_text]
        if codified_ids:
            merged_obl = list(dict.fromkeys(selection.obligation_chunks + codified_ids))
            selection = LocatorSelection(
                date_chunks=selection.date_chunks,
                scope_chunks=selection.scope_chunks,
                obligation_chunks=merged_obl,
                definition_chunks=selection.definition_chunks,
                other_key_chunks=selection.other_key_chunks,
            )
            logger.info("Force-injected %d codified chunks into obligation_chunks", len(codified_ids))

        # Validate: all src_ids must exist
        all_selected_ids = _union_src_ids(selection)
        valid_ids = [sid for sid in all_selected_ids if sid in chunk_map]
        invalid = set(all_selected_ids) - set(valid_ids)
        if invalid:
            logger.warning("Locator returned unknown src_ids: %s -- dropping them", invalid)
        if not valid_ids:
            logger.warning("No valid src_ids from Locator; falling back to all chunks")
            valid_ids = [c.src_id for c in chunks]

        # Validate: obligation_chunks must be non-empty after filtering
        obl_valid = [sid for sid in selection.obligation_chunks if sid in chunk_map]
        if not obl_valid:
            logger.warning("obligation_chunks empty after validation; using all obligation-flagged chunks")
            obl_valid = [c.src_id for c in chunks if c.has_obligations]
            if not obl_valid:
                obl_valid = [chunks[0].src_id]
            selection = LocatorSelection(
                date_chunks=selection.date_chunks,
                scope_chunks=selection.scope_chunks,
                obligation_chunks=obl_valid,
                definition_chunks=selection.definition_chunks,
                other_key_chunks=selection.other_key_chunks,
            )

        # Apply cap: max 60 total
        final_ids = _union_src_ids(selection)
        final_ids = [sid for sid in final_ids if sid in chunk_map]
        if len(final_ids) > 60:
            logger.info("Locator selected %d chunks; capping at 60", len(final_ids))
            final_ids = final_ids[:60]

        # Resolve to full RichChunk objects, preserving original order
        src_order = {c.src_id: c.chunk_index_in_section + (int(c.src_id.split(":")[1]) * 1000)
                     for c in chunks}
        selected_chunks = sorted(
            [chunk_map[sid] for sid in set(final_ids)],
            key=lambda c: src_order.get(c.src_id, 0),
        )

        logger.info(
            "Locator selected %d / %d chunks (obligation: %d)",
            len(selected_chunks), len(chunks), len(obl_valid),
        )

        # Save locator_selection.json
        artifact_dir = os.path.join("artifacts", state["run_id"])
        os.makedirs(artifact_dir, exist_ok=True)
        _write_file(
            artifact_dir,
            "locator_selection.json",
            json.dumps(selection.model_dump(mode="json"), indent=2),
        )

        return {
            "locator_selection": selection,
            "selected_chunks": selected_chunks,
            "token_usage": {"locator": locator_usage},
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
        locator_selection = state.get("locator_selection")

        artifact_dir = os.path.join("artifacts", run_id)
        os.makedirs(artifact_dir, exist_ok=True)

        _write_file(artifact_dir, "raw_model_output.txt", raw_output)
        _write_file(
            artifact_dir,
            "validated_output.json",
            json.dumps(output.model_dump(mode="json"), indent=2),
        )

        # Save locator_selection.json if not already written (direct mode has no locator)
        locator_path = os.path.join(artifact_dir, "locator_selection.json")
        if locator_selection and not os.path.exists(locator_path):
            _write_file(
                artifact_dir,
                "locator_selection.json",
                json.dumps(locator_selection.model_dump(mode="json"), indent=2),
            )

        model_name = os.getenv("SEC_INTERPRETER_MODEL", "DeterministicLLM")
        timestamp = datetime.now(timezone.utc).isoformat()
        retries = state.get("retry_count", 0)

        token_usage = state.get("token_usage") or {}
        locator_usage = token_usage.get("locator", {})
        extractor_usage = token_usage.get("extractor", {})
        grand_total = locator_usage.get("total_tokens", 0) + extractor_usage.get("total_tokens", 0)

        log = (
            f"run_id: {run_id}\n"
            f"model: {model_name}\n"
            f"total_chunk_count: {len(chunks)}\n"
            f"selected_chunk_count: {len(selected_chunks)}\n"
            f"retries: {retries}\n"
            f"timestamp: {timestamp}\n"
            f"validation_result: success\n"
            f"locator_prompt_tokens: {locator_usage.get('prompt_tokens', 0)}\n"
            f"locator_completion_tokens: {locator_usage.get('completion_tokens', 0)}\n"
            f"locator_total_tokens: {locator_usage.get('total_tokens', 0)}\n"
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
                "locator_used": locator_selection is not None,
                "token_usage": {
                    "locator": locator_usage,
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
    # Build graph
    # ------------------------------------------------------------------
    graph = StateGraph(ExtractState)
    graph.add_node("load_chunks", load_chunks)
    graph.add_node("locator_pass", locator_pass)
    graph.add_node("extract_structured_fields", extract_structured_fields)
    graph.add_node("validate_output", validate_output)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("save_extract_artifacts", save_extract_artifacts)

    graph.add_edge(START, "load_chunks")
    graph.add_conditional_edges(
        "load_chunks",
        _route_after_load,
        {
            "locator_pass": "locator_pass",
            "extract_structured_fields": "extract_structured_fields",
        },
    )
    graph.add_edge("locator_pass", "extract_structured_fields")
    graph.add_edge("extract_structured_fields", "validate_output")
    graph.add_conditional_edges(
        "validate_output",
        _route_after_validation,
        {
            "save_extract_artifacts": "save_extract_artifacts",
            "increment_retry": "increment_retry",
        },
    )
    graph.add_edge("increment_retry", "extract_structured_fields")
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
