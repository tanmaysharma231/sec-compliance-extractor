"""
sec_interpreter/ingest_graph.py

Purpose: LangGraph pipeline for Stage 1 (IngestGraph) — pure Python, no LLM.
         Fetches the document, segments it into labelled sections, chunks each
         section into size-bounded RichChunk objects, scores hot-zone flags, and
         auto-extracts the SUMMARY section text.

Old flow: fetch_document → chunk_text → save_ingest_artifacts
New flow: fetch_document → segment_document → chunk_sections → score_chunks
                         → extract_summary → save_ingest_artifacts

Called by: IngestModule.run()
Calls:     ingest.fetch_rule_text, segmenter.segment_document,
           scorer.score_chunk / build_index_row, schemas.*
"""
from __future__ import annotations

import json
import re
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .ingest import fetch_rule_text
from .schemas import IngestInput, IngestResult, RichChunk, Section
from .segmenter import segment_document as _segment_document
from .scorer import build_index_row, score_chunk
from .utils import chunk_rule_text, get_logger

# ---------------------------------------------------------------------------
# Chunk sizing constants (token estimates; token_estimate = char_len // 4)
# ---------------------------------------------------------------------------
_TARGET_CHARS = 4000    # ~1000 tokens  (target chunk size)
_HARD_CAP_CHARS = 6400  # ~1600 tokens  (absolute split boundary)
_OVERLAP_CHARS = 600    # ~150 tokens   (overlap carried from prev chunk in section)

_SUMMARY_ANCHORS = {"SUMMARY", "SUPPLEMENTARY INFORMATION"}

# Regex to detect inline label at start of a LINE: "SUMMARY: ..." or "SUPPLEMENTARY INFORMATION: ..."
_INLINE_SUMMARY_LINE_RE = re.compile(
    r"^(SUMMARY|SUPPLEMENTARY INFORMATION)\s*:\s*(.+)",
    re.IGNORECASE,
)
# Label that signals the END of a summary block (next labelled section)
_NEXT_LABEL_RE = re.compile(
    r"^(DATES|EFFECTIVE DATE|COMPLIANCE DATE|FOR FURTHER INFORMATION|ADDRESSES?)\s*:",
    re.IGNORECASE,
)


class IngestState(TypedDict, total=False):
    ingest_input: IngestInput
    run_id: str
    rule_text: str
    sections: List[Section]       # NEW
    chunks: List[RichChunk]       # CHANGED: was List[str]
    summary_text: str             # NEW: auto-extracted SUMMARY section text


def build_ingest_graph():
    logger = get_logger("sec_interpreter.ingest")

    # ------------------------------------------------------------------
    # Node: fetch_document
    # ------------------------------------------------------------------
    def fetch_document(state: IngestState) -> IngestState:
        ingest_input = state["ingest_input"]
        run_id = uuid4().hex[:12]
        logger.info("Ingest run_id=%s  source=%s", run_id, ingest_input.source)
        rule_text = fetch_rule_text(ingest_input.source, page_range=ingest_input.page_range)
        logger.info("Fetched %d chars of rule text", len(rule_text))
        return {"run_id": run_id, "rule_text": rule_text}

    # ------------------------------------------------------------------
    # Node: segment_document
    # ------------------------------------------------------------------
    def segment_document(state: IngestState) -> IngestState:
        sections = _segment_document(state["rule_text"])
        logger.info(
            "Segmented into %d sections (first few: %s)",
            len(sections),
            [s.heading_path for s in sections[:4]],
        )
        return {"sections": sections}

    # ------------------------------------------------------------------
    # Node: chunk_sections
    # ------------------------------------------------------------------
    def chunk_sections(state: IngestState) -> IngestState:
        sections = state["sections"]
        all_chunks: List[RichChunk] = []
        global_idx = 0

        for section in sections:
            section_chunks, global_idx = _chunk_section(
                section,
                start_global_idx=global_idx,
            )
            all_chunks.extend(section_chunks)

        logger.info("Produced %d rich chunks from %d sections", len(all_chunks), len(sections))
        return {"chunks": all_chunks}

    # ------------------------------------------------------------------
    # Node: score_chunks
    # ------------------------------------------------------------------
    def score_chunks(state: IngestState) -> IngestState:
        scored = [score_chunk(c) for c in state["chunks"]]
        obl_count = sum(1 for c in scored if c.has_obligations)
        logger.info(
            "Scored %d chunks; %d have obligation flags", len(scored), obl_count
        )
        return {"chunks": scored}

    # ------------------------------------------------------------------
    # Node: extract_summary
    # ------------------------------------------------------------------
    def extract_summary(state: IngestState) -> IngestState:
        sections = state["sections"]
        rule_text = state["rule_text"]
        summary_parts: List[str] = []

        # Strategy 1: sections whose heading_path[0] exactly matches a known anchor
        for sec in sections:
            top_label = sec.heading_path[0].upper().strip() if sec.heading_path else ""
            if top_label in _SUMMARY_ANCHORS:
                summary_parts.append(sec.section_text)

        # Strategy 2: scan rule_text line-by-line for inline Federal Register labels
        # ("SUMMARY: text..." / "SUPPLEMENTARY INFORMATION: text...") which appear
        # mid-paragraph in the raw extracted PDF text.
        if not summary_parts:
            lines = rule_text.splitlines()
            collecting = False
            collected_lines: List[str] = []
            label_name = ""

            for line in lines:
                stripped = line.strip()

                # Check if this line starts a summary-type block
                m = _INLINE_SUMMARY_LINE_RE.match(stripped)
                if m:
                    # Save previous block if we were collecting something else
                    if collecting and collected_lines:
                        summary_parts.append(
                            f"{label_name}: " + "\n".join(collected_lines).strip()
                        )
                        collected_lines = []
                    label_name = m.group(1).upper()
                    collecting = True
                    collected_lines.append(m.group(2).strip())
                    continue

                if collecting:
                    # Stop collecting when we hit the next labelled section
                    if _NEXT_LABEL_RE.match(stripped):
                        break
                    collected_lines.append(line)

            if collecting and collected_lines:
                summary_parts.append(
                    f"{label_name}: " + "\n".join(collected_lines).strip()
                )

        summary_text = "\n\n".join(summary_parts).strip()
        if summary_text:
            logger.info(
                "Extracted summary text (%d chars) from SUMMARY/SUPPLEMENTARY INFORMATION",
                len(summary_text),
            )
        else:
            logger.warning(
                "No SUMMARY / SUPPLEMENTARY INFORMATION section found; summary_text will be empty"
            )
        return {"summary_text": summary_text}

    # ------------------------------------------------------------------
    # Node: save_ingest_artifacts
    # ------------------------------------------------------------------
    def save_ingest_artifacts(state: IngestState) -> IngestState:
        run_id = state["run_id"]
        rule_text = state["rule_text"]
        sections = state["sections"]
        chunks = state["chunks"]
        summary_text = state.get("summary_text", "")
        ingest_input = state["ingest_input"]

        artifact_dir = os.path.join("artifacts", run_id)
        os.makedirs(artifact_dir, exist_ok=True)

        # input.txt — raw fetched text (unchanged)
        _write_file(artifact_dir, "input.txt", rule_text)

        # sections.json — List[Section] serialized
        sections_data = [s.model_dump(mode="json") for s in sections]
        _write_file(artifact_dir, "sections.json", json.dumps(sections_data, indent=2))

        # chunks.json — List[RichChunk] serialized (new format)
        chunks_data = [c.model_dump(mode="json") for c in chunks]
        _write_file(artifact_dir, "chunks.json", json.dumps(chunks_data, indent=2))

        # summary.txt — auto-extracted SUMMARY section
        _write_file(artifact_dir, "summary.txt", summary_text)

        # shortlist.json — compact index rows for Locator
        index_rows = [build_index_row(c) for c in chunks]
        _write_file(artifact_dir, "shortlist.json", json.dumps(index_rows, indent=2))

        # ingest_log.txt
        timestamp = datetime.now(timezone.utc).isoformat()
        log = (
            f"run_id: {run_id}\n"
            f"source: {ingest_input.source}\n"
            f"page_range: {ingest_input.page_range}\n"
            f"section_count: {len(sections)}\n"
            f"chunk_count: {len(chunks)}\n"
            f"rule_text_chars: {len(rule_text)}\n"
            f"summary_chars: {len(summary_text)}\n"
            f"timestamp: {timestamp}\n"
        )
        _write_file(artifact_dir, "ingest_log.txt", log)
        logger.info("Ingest artifacts written to %s/", artifact_dir)
        return {}

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    graph = StateGraph(IngestState)
    graph.add_node("fetch_document", fetch_document)
    graph.add_node("segment_document", segment_document)
    graph.add_node("chunk_sections", chunk_sections)
    graph.add_node("score_chunks", score_chunks)
    graph.add_node("extract_summary", extract_summary)
    graph.add_node("save_ingest_artifacts", save_ingest_artifacts)

    graph.add_edge(START, "fetch_document")
    graph.add_edge("fetch_document", "segment_document")
    graph.add_edge("segment_document", "chunk_sections")
    graph.add_edge("chunk_sections", "score_chunks")
    graph.add_edge("score_chunks", "extract_summary")
    graph.add_edge("extract_summary", "save_ingest_artifacts")
    graph.add_edge("save_ingest_artifacts", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Section chunking helpers
# ---------------------------------------------------------------------------

def _derive_subsection_role(heading_path: list) -> str:
    """Derive subsection role from heading_path[2] naming patterns."""
    if len(heading_path) < 3:
        return "other"
    level3 = heading_path[2].lower()
    if "proposed" in level3:
        return "proposed"
    if "comment" in level3:
        return "comments"
    if "final" in level3:
        return "final"
    return "other"


def _chunk_section(
    section: Section,
    start_global_idx: int,
    target_chars: int = _TARGET_CHARS,
    hard_cap_chars: int = _HARD_CAP_CHARS,
    overlap_chars: int = _OVERLAP_CHARS,
) -> Tuple[List[RichChunk], int]:
    """Split a Section's text into size-bounded RichChunk objects.

    Returns (list_of_chunks, next_global_idx).
    """
    text = section.section_text
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [], start_global_idx

    chunks: List[RichChunk] = []
    global_idx = start_global_idx
    chunk_idx_in_sec = 0
    current_parts: List[str] = []
    current_len = 0
    last_main_text = ""  # tracks previous chunk's main body for overlap

    def _make_chunk(main_text: str, prev_text: str) -> RichChunk:
        nonlocal chunk_idx_in_sec
        # Only add overlap for non-first chunks within the same section
        if prev_text and chunk_idx_in_sec > 0:
            overlap = prev_text[-overlap_chars:] if len(prev_text) > overlap_chars else prev_text
            full_text = overlap + "\n\n" + main_text
        else:
            full_text = main_text
        chunk = RichChunk(
            src_id=f"src:{global_idx}",
            section_id=section.section_id,
            heading_path=section.heading_path,
            chunk_index_in_section=chunk_idx_in_sec,
            text=full_text,
            char_len=len(full_text),
            token_estimate=len(full_text) // 4,
            section_family=section.heading_path[1] if len(section.heading_path) >= 2 else "",
            subsection_role=_derive_subsection_role(section.heading_path),
        )
        chunk_idx_in_sec += 1
        return chunk

    for para in paragraphs:
        # Handle oversized single paragraph (exceeds hard cap)
        if len(para) > hard_cap_chars:
            # Flush current accumulation first
            if current_parts:
                main = "\n\n".join(current_parts)
                chunks.append(_make_chunk(main, last_main_text))
                last_main_text = main
                global_idx += 1
                current_parts = []
                current_len = 0

            # Slice the large paragraph at hard_cap_chars boundaries
            pos = 0
            while pos < len(para):
                sub = para[pos: pos + hard_cap_chars]
                chunks.append(_make_chunk(sub, last_main_text))
                last_main_text = sub
                global_idx += 1
                pos += hard_cap_chars
            continue

        # If adding this paragraph would exceed target, flush first
        if current_parts and current_len + len(para) + 2 > target_chars:
            main = "\n\n".join(current_parts)
            chunks.append(_make_chunk(main, last_main_text))
            last_main_text = main
            global_idx += 1
            current_parts = []
            current_len = 0

        current_parts.append(para)
        current_len += len(para) + 2

    # Flush remainder
    if current_parts:
        main = "\n\n".join(current_parts)
        chunks.append(_make_chunk(main, last_main_text))
        global_idx += 1

    return chunks, global_idx


def _write_file(directory: str, filename: str, content: str) -> None:
    with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
        f.write(content)
