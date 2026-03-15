"""
sec_interpreter/classify_graph.py

Purpose: LangGraph pipeline for the Classify step — runs once per ingest run_id,
         classifies every section by content type, saves results permanently.

Flow:
    load_chunks -> classify_sections -> synthesise_document -> save_classify_artifacts

Cache: if section_classifications.json already exists for the run_id, skips all
       LLM calls and re-applies stored content_type values to chunks.

Called by: ClassifyModule.run()
Saves:     chunks.json (updated with content_type), section_classifications.json,
           document_map.json
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from .prompts import build_document_synthesis_prompt, build_section_classify_prompt
from .schemas import (
    DocumentMap,
    RichChunk,
    SectionClassification,
    COMPLIANCE_CONTENT_TYPES,
    VALID_CONTENT_TYPES,
)
from .utils import get_logger, parse_json_object


class ClassifyState(TypedDict, total=False):
    run_id: str
    chunks: List[RichChunk]
    section_classifications: List[SectionClassification]
    document_map: Optional[DocumentMap]
    cache_hit: bool


def build_classify_graph(llm: Any, logger: logging.Logger):

    # ------------------------------------------------------------------
    # Node: load_chunks
    # Checks cache first — if section_classifications.json exists, loads
    # it and re-applies content_type to chunks without any LLM calls.
    # ------------------------------------------------------------------
    def load_chunks(state: ClassifyState) -> ClassifyState:
        run_id = state["run_id"]
        artifact_dir = os.path.join("artifacts", run_id)
        chunks_path = os.path.join(artifact_dir, "chunks.json")
        classifications_path = os.path.join(artifact_dir, "section_classifications.json")

        if not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"No chunks.json found for run_id={run_id}. "
                "Run the ingest pipeline first."
            )

        with open(chunks_path, encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunks: List[RichChunk] = []
        for entry in chunks_data:
            if "src_id" in entry:
                chunks.append(RichChunk.model_validate(entry))
            else:
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

        # Cache check
        if os.path.exists(classifications_path):
            logger.info(
                "Using cached classification for run_id=%s (%s)",
                run_id, classifications_path,
            )
            with open(classifications_path, encoding="utf-8") as f:
                cached_data = json.load(f)

            section_classifications = [
                SectionClassification.model_validate(entry) for entry in cached_data
            ]

            # Re-apply content_type from cache to chunks
            section_type_map = {sc.section_id: sc.content_type for sc in section_classifications}
            for chunk in chunks:
                ct = section_type_map.get(chunk.section_id, "")
                object.__setattr__(chunk, "content_type", ct) if hasattr(chunk, "__slots__") else None
                # Pydantic v2 model — use model_copy to update
                chunks[chunks.index(chunk)] = chunk.model_copy(update={"content_type": ct})

            logger.info(
                "Re-applied content_type to %d chunks from cache (%d sections)",
                len(chunks), len(section_classifications),
            )
            return {
                "chunks": chunks,
                "section_classifications": section_classifications,
                "cache_hit": True,
            }

        logger.info(
            "Running fresh classification for run_id=%s (%d chunks)",
            run_id, len(chunks),
        )
        return {
            "chunks": chunks,
            "section_classifications": [],
            "cache_hit": False,
        }

    # ------------------------------------------------------------------
    # Routing after load_chunks
    # ------------------------------------------------------------------
    def _route_after_load(state: ClassifyState) -> str:
        if state.get("cache_hit", False):
            return "save_classify_artifacts"
        return "classify_sections"

    # ------------------------------------------------------------------
    # Node: classify_sections
    # Groups chunks by section_id, one LLM call per section group.
    # ------------------------------------------------------------------
    def classify_sections(state: ClassifyState) -> ClassifyState:
        chunks = state["chunks"]

        # Group by section_id, preserving chunk order
        section_groups: dict[str, list[RichChunk]] = defaultdict(list)
        section_order: list[str] = []
        for chunk in chunks:
            if chunk.section_id not in section_groups:
                section_order.append(chunk.section_id)
            section_groups[chunk.section_id].append(chunk)

        section_classifications: List[SectionClassification] = []
        section_type_map: dict[str, str] = {}

        total = len(section_order)
        logger.info("Classifying %d sections...", total)

        for idx, section_id in enumerate(section_order):
            group = section_groups[section_id]
            heading_path = group[0].heading_path

            # Combine chunk texts up to 6000-char hard cap
            combined = "\n\n".join(c.text for c in group)

            prompt = build_section_classify_prompt(heading_path, combined)
            messages = [HumanMessage(content=prompt)]

            content_type = "commentary"  # safe fallback
            summary = ""
            topics: list[str] = []
            useful_for: list[str] = []

            try:
                response = llm.invoke(messages)
                raw = _normalize_content(getattr(response, "content", response))
                parsed = parse_json_object(raw)

                # Validate content_type
                ct = parsed.get("content_type", "commentary")
                if ct not in VALID_CONTENT_TYPES:
                    logger.warning(
                        "Section %s: invalid content_type %r, defaulting to 'commentary'",
                        section_id, ct,
                    )
                    ct = "commentary"
                content_type = ct
                summary = parsed.get("summary", "")
                topics = parsed.get("topics", [])
                useful_for = parsed.get("useful_for", [])

            except Exception as exc:
                logger.warning("Section %s classification failed: %s", section_id, exc)

            sc = SectionClassification(
                section_id=section_id,
                heading_path=heading_path,
                content_type=content_type,
                summary=summary,
                topics=topics,
                useful_for=useful_for,
            )
            section_classifications.append(sc)
            section_type_map[section_id] = content_type

            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                logger.info("  Classified %d / %d sections", idx + 1, total)

        # Apply content_type back to all chunks
        updated_chunks = [
            chunk.model_copy(update={"content_type": section_type_map.get(chunk.section_id, "")})
            for chunk in chunks
        ]

        type_counts: dict[str, int] = defaultdict(int)
        for chunk in updated_chunks:
            type_counts[chunk.content_type] += 1
        logger.info("Classification complete. Chunk distribution: %s", dict(type_counts))

        return {
            "chunks": updated_chunks,
            "section_classifications": section_classifications,
        }

    # ------------------------------------------------------------------
    # Node: synthesise_document
    # One LLM call over all section summaries to build DocumentMap.
    # ------------------------------------------------------------------
    def synthesise_document(state: ClassifyState) -> ClassifyState:
        section_classifications = state["section_classifications"]

        if not section_classifications:
            logger.warning("No section classifications available; skipping document synthesis")
            return {
                "document_map": DocumentMap(
                    regulatory_objective="Unknown",
                    rule_title="Unknown",
                )
            }

        summaries = [sc.model_dump(mode="json") for sc in section_classifications]
        prompt = build_document_synthesis_prompt(summaries)
        messages = [HumanMessage(content=prompt)]

        try:
            response = llm.invoke(messages)
            raw = _normalize_content(getattr(response, "content", response))
            parsed = parse_json_object(raw)
            document_map = DocumentMap.model_validate(parsed)
            logger.info(
                "Document synthesis complete. rule_title=%r", document_map.rule_title
            )
        except Exception as exc:
            logger.warning("Document synthesis failed: %s -- building from section data", exc)
            document_map = _build_document_map_from_sections(section_classifications)

        # Override LLM-generated section_id lists with deterministic values computed
        # from content_type. The LLM is good at summarising (regulatory_objective,
        # rule_title) but unreliable at exhaustively enumerating all section_ids.
        document_map = _override_section_id_lists(document_map, section_classifications)
        logger.info(
            "Document map finalised. Compliance sections: %d, cost: %d, definition: %d",
            len(document_map.compliance_section_ids),
            len(document_map.cost_section_ids),
            len(document_map.definition_section_ids),
        )

        return {"document_map": document_map}

    # ------------------------------------------------------------------
    # Node: save_classify_artifacts
    # Overwrites chunks.json, saves section_classifications.json and
    # document_map.json.
    # ------------------------------------------------------------------
    def save_classify_artifacts(state: ClassifyState) -> ClassifyState:
        run_id = state["run_id"]
        chunks = state["chunks"]
        section_classifications = state["section_classifications"]
        document_map = state.get("document_map")

        artifact_dir = os.path.join("artifacts", run_id)
        os.makedirs(artifact_dir, exist_ok=True)

        # Overwrite chunks.json with content_type populated
        chunks_data = [c.model_dump(mode="json") for c in chunks]
        _write_file(artifact_dir, "chunks.json", json.dumps(chunks_data, indent=2))

        # Save section_classifications.json
        sc_data = [sc.model_dump(mode="json") for sc in section_classifications]
        _write_file(
            artifact_dir,
            "section_classifications.json",
            json.dumps(sc_data, indent=2),
        )

        # Save document_map.json
        # Build deterministically if not available (cache-hit path skips synthesise_document)
        if document_map is None:
            document_map = _build_document_map_from_sections(section_classifications)
            document_map = _override_section_id_lists(document_map, section_classifications)
            logger.info(
                "document_map built deterministically (cache path). Compliance sections: %d",
                len(document_map.compliance_section_ids),
            )
        _write_file(
            artifact_dir,
            "document_map.json",
            json.dumps(document_map.model_dump(mode="json"), indent=2),
        )

        logger.info(
            "Classify artifacts written to %s/ (%d sections, %d chunks)",
            artifact_dir, len(section_classifications), len(chunks),
        )
        return {"document_map": document_map}

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    graph = StateGraph(ClassifyState)
    graph.add_node("load_chunks", load_chunks)
    graph.add_node("classify_sections", classify_sections)
    graph.add_node("synthesise_document", synthesise_document)
    graph.add_node("save_classify_artifacts", save_classify_artifacts)

    graph.add_edge(START, "load_chunks")
    graph.add_conditional_edges(
        "load_chunks",
        _route_after_load,
        {
            "classify_sections": "classify_sections",
            "save_classify_artifacts": "save_classify_artifacts",
        },
    )
    graph.add_edge("classify_sections", "synthesise_document")
    graph.add_edge("synthesise_document", "save_classify_artifacts")
    graph.add_edge("save_classify_artifacts", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _override_section_id_lists(
    document_map: DocumentMap,
    section_classifications: List[SectionClassification],
) -> DocumentMap:
    """Replace LLM-generated section_id lists with deterministic values from content_type.

    The LLM synthesis is only trusted for narrative fields (regulatory_objective,
    rule_title, sections_by_type). Section_id task lists are computed here so they
    are complete and reproducible.
    """
    compliance_ids = [
        sc.section_id for sc in section_classifications
        if sc.content_type in COMPLIANCE_CONTENT_TYPES
    ]
    cost_ids = [
        sc.section_id for sc in section_classifications
        if sc.content_type == "economic_analysis"
    ]
    definition_ids = [
        sc.section_id for sc in section_classifications
        if sc.content_type == "definition"
    ]
    return document_map.model_copy(update={
        "compliance_section_ids": compliance_ids,
        "cost_section_ids": cost_ids,
        "definition_section_ids": definition_ids,
    })


def _build_document_map_from_sections(
    section_classifications: List[SectionClassification],
) -> DocumentMap:
    """Fallback: build DocumentMap deterministically from section classification data."""
    sections_by_type: dict[str, list[str]] = defaultdict(list)
    for sc in section_classifications:
        sections_by_type[sc.content_type].append(sc.section_id)

    compliance_ids = (
        sections_by_type.get("final_rule_text", [])
        + sections_by_type.get("obligation", [])
        + sections_by_type.get("definition", [])
    )
    cost_ids = sections_by_type.get("economic_analysis", [])
    definition_ids = sections_by_type.get("definition", [])

    return DocumentMap(
        regulatory_objective="SEC regulatory rule (objective inferred from section classification)",
        rule_title="SEC Rule",
        sections_by_type=dict(sections_by_type),
        compliance_section_ids=compliance_ids,
        cost_section_ids=cost_ids,
        definition_section_ids=definition_ids,
    )


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
