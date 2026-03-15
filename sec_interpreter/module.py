from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from dotenv import load_dotenv

from .classify_graph import build_classify_graph
from .extract_graph import build_extract_graph
from .ingest_graph import build_ingest_graph
from .schemas import DocumentMap, IngestInput, IngestResult, RuleExtractorInput, RuleExtractorOutput
from .utils import get_logger

# Load .env at import time so env vars are available before anything else runs
load_dotenv()


@dataclass
class FakeLLM:
    """Test double — returns a pre-built response without any API call."""
    response: dict[str, Any] | str | Callable[[list[Any]], dict[str, Any] | str]

    def invoke(self, messages: list[Any]) -> Any:
        payload = self.response(messages) if callable(self.response) else self.response
        content = json.dumps(payload) if isinstance(payload, dict) else str(payload)
        return SimpleNamespace(content=content)


class DeterministicLLM:
    """Offline stub — produces a minimal valid RuleExtractorOutput without any API call.
    Used automatically when no model is configured in .env.
    """
    def invoke(self, messages: list[Any]) -> Any:
        user_prompt = str(messages[-1].content)
        chunk_count = _parse_chunk_count(user_prompt)
        return SimpleNamespace(content=json.dumps(_build_fallback_output(chunk_count)))


# ---------------------------------------------------------------------------
# IngestModule — Stage 1: fetch document → chunk → save artifacts
# ---------------------------------------------------------------------------

class IngestModule:
    """Fetches a regulatory document, chunks it, and saves artifacts to disk.

    Returns an IngestResult containing the run_id that ExtractModule uses
    to load the chunks for extraction.
    """

    def __init__(self):
        self.logger = get_logger("sec_interpreter.ingest")
        self.graph = build_ingest_graph()

    def run(
        self,
        source: str,
        page_range: tuple[int, int] | None = None,
        strict_citations: bool = False,
    ) -> IngestResult:
        ingest_input = IngestInput(
            source=source,
            page_range=page_range,
            strict_citations=strict_citations,
        )
        final_state = self.graph.invoke({"ingest_input": ingest_input})
        run_id = final_state["run_id"]
        chunks = final_state["chunks"]
        return IngestResult(
            run_id=run_id,
            chunk_count=len(chunks),
            artifact_dir=os.path.join("artifacts", run_id),
        )


# ---------------------------------------------------------------------------
# ClassifyModule — classify sections once, store content_type on chunks
# ---------------------------------------------------------------------------

class ClassifyModule:
    """Classifies every section in an ingested document by content type.

    Runs once per run_id. Results are cached in section_classifications.json;
    subsequent calls are instant (no LLM calls).

    Usage:
        ingest_result = IngestModule().run("path/to/rule.pdf")
        classify_result = ClassifyModule().run(ingest_result.run_id)
        # Now ExtractModule will use deterministic content_type filter
        output = ExtractModule().run(ingest_result.run_id)
    """

    def __init__(self, llm: Any | None = None):
        self.logger = get_logger("sec_interpreter.classify")
        self.llm = llm or _load_env_llm() or DeterministicLLM()
        self.graph = build_classify_graph(self.llm, self.logger)

    def run(self, run_id: str) -> dict:
        """Run classification for a previously ingested document.

        Returns a summary dict with section_count and document_map info.
        """
        final_state = self.graph.invoke({"run_id": run_id})
        section_classifications = final_state.get("section_classifications", [])
        document_map: DocumentMap | None = final_state.get("document_map")

        type_counts: dict[str, int] = {}
        for sc in section_classifications:
            type_counts[sc.content_type] = type_counts.get(sc.content_type, 0) + 1

        return {
            "run_id": run_id,
            "section_count": len(section_classifications),
            "type_counts": type_counts,
            "compliance_section_count": len(document_map.compliance_section_ids) if document_map else 0,
            "rule_title": document_map.rule_title if document_map else "",
            "regulatory_objective": document_map.regulatory_objective if document_map else "",
        }


# ---------------------------------------------------------------------------
# ExtractModule — Stage 2: load chunks -> LLM extract -> validate -> save artifacts
# ---------------------------------------------------------------------------

class ExtractModule:
    """Loads chunks from a previous ingest run and extracts compliance intelligence.

    Usage:
        result = IngestModule().run("https://sec.gov/rules/final/2023/33-11216.pdf")
        output = ExtractModule().run(result.run_id, strict_citations=True)
    """

    def __init__(self, llm: Any | None = None):
        self.logger = get_logger("sec_interpreter")
        self.llm = llm or _load_env_llm() or DeterministicLLM()
        self.graph = build_extract_graph(self.llm, self.logger)

    def run(
        self,
        run_id: str,
        strict_citations: bool = False,
        skip_locator: bool = False,
    ) -> RuleExtractorOutput:
        """Extract from a previously ingested document (loads chunks from artifacts)."""
        final_state = self.graph.invoke({
            "run_id": run_id,
            "payload": None,
            "strict_citations": strict_citations,
            "skip_locator": skip_locator,
        })
        return final_state["output"]


# ---------------------------------------------------------------------------
# RuleExtractorModule — combined, backward-compatible module
# Accepts rule_text directly (no separate ingest step).
# Used by tests and the legacy --input CLI path.
# ---------------------------------------------------------------------------

class RuleExtractorModule:
    """Combined pipeline: accepts rule_text directly, chunks in memory, extracts.

    Primarily used for testing and simple single-document runs.
    For large documents use IngestModule + ExtractModule separately.
    """

    def __init__(self, llm: Any | None = None):
        self.logger = get_logger("sec_interpreter")
        self.llm = llm or _load_env_llm() or DeterministicLLM()
        self.graph = build_extract_graph(self.llm, self.logger)

    def run(self, payload: RuleExtractorInput | dict[str, Any]) -> RuleExtractorOutput:
        import uuid
        parsed_input = (
            payload
            if isinstance(payload, RuleExtractorInput)
            else RuleExtractorInput.model_validate(payload)
        )
        run_id = uuid.uuid4().hex[:12]
        final_state = self.graph.invoke({
            "run_id": run_id,
            "payload": parsed_input,
        })
        return final_state["output"]


class InterpretModule:
    """Runs the interpretation pipeline on an already-extracted run_id."""

    def __init__(self, llm: Any | None = None, cheap_llm: Any | None = None):
        self.logger = get_logger("sec_interpreter")
        self.llm = llm or _load_env_llm() or DeterministicLLM()
        # cheap_llm for reference judge -- falls back to same model if not set
        self.cheap_llm = cheap_llm or _load_cheap_llm() or self.llm

    def run(self, run_id: str):
        from .interpret_graph import run_interpret_pipeline
        from .schemas import InterpretationOutput
        return run_interpret_pipeline(run_id, self.llm, self.cheap_llm, self.logger)


# ---------------------------------------------------------------------------
# LLM loading
# ---------------------------------------------------------------------------

def _load_env_llm() -> Any | None:
    model_name = os.getenv("SEC_INTERPRETER_MODEL", "").strip()
    if not model_name:
        return None

    provider = os.getenv("SEC_INTERPRETER_MODEL_PROVIDER", "").strip() or None

    try:
        from langchain.chat_models import init_chat_model
        kwargs: dict[str, Any] = {}
        if provider:
            kwargs["model_provider"] = provider
        llm = init_chat_model(model_name, **kwargs)
        get_logger("sec_interpreter").info(
            "Using LLM: %s (provider: %s)", model_name, provider or "auto-detected"
        )
        return llm
    except Exception as exc:
        get_logger("sec_interpreter").warning(
            "Failed to initialise LLM %r: %s — falling back to DeterministicLLM", model_name, exc
        )
        return None


def _load_cheap_llm() -> Any | None:
    """Load a cheap/fast LLM for routing decisions (reference judge).

    Uses SEC_INTERPRETER_CHEAP_MODEL if set; otherwise derives from main model:
      gpt-4o  -> gpt-4o-mini
      claude-* -> claude-haiku-4-5-20251001
    Falls back to None (caller uses main model) if no suitable cheap model found.
    """
    cheap_model = os.getenv("SEC_INTERPRETER_CHEAP_MODEL", "").strip()

    if not cheap_model:
        main_model = os.getenv("SEC_INTERPRETER_MODEL", "").strip()
        if main_model.startswith("gpt-4o"):
            cheap_model = "gpt-4o-mini"
        elif main_model.startswith("claude-"):
            cheap_model = "claude-haiku-4-5-20251001"
        else:
            return None

    provider = os.getenv("SEC_INTERPRETER_MODEL_PROVIDER", "").strip() or None
    try:
        from langchain.chat_models import init_chat_model
        kwargs: dict[str, Any] = {}
        if provider:
            kwargs["model_provider"] = provider
        llm = init_chat_model(cheap_model, **kwargs)
        get_logger("sec_interpreter").info("Using cheap LLM: %s", cheap_model)
        return llm
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DeterministicLLM helpers
# ---------------------------------------------------------------------------

def _parse_chunk_count(user_prompt: str) -> int:
    matches = re.findall(r"\[src:(\d+)\]", user_prompt)
    if not matches:
        return 1
    return max(int(m) for m in matches) + 1


def _build_fallback_output(chunk_count: int) -> dict[str, Any]:
    citations = [f"src:{i}" for i in range(min(chunk_count, 2))]
    first_citation = citations[0] if citations else "src:0"
    return {
        "rule_metadata": {
            "rule_title": "SEC Regulatory Rule",
            "release_number": None,
            "publication_date": None,
            "effective_date": None,
            "citations": [first_citation],
        },
        "rule_summary": {
            "summary": (
                "This SEC rule establishes regulatory requirements for covered entities. "
                "The rule imposes obligations relating to disclosure, recordkeeping, and governance. "
                "Covered entities should review their current practices against the requirements. "
                "Implementation timelines and compliance gaps should be identified and tracked. "
                "Legal and compliance counsel should be consulted for entity-specific guidance."
            ),
            "citations": citations,
        },
        "key_obligations": [
            {
                "obligation_id": "OBL-001",
                "obligation_text": (
                    "Covered entities should review applicable disclosure requirements and assess "
                    "current practices against the rule's expectations."
                ),
                "cited_sections": [],
                "source_citations": [first_citation],
            },
            {
                "obligation_id": "OBL-002",
                "obligation_text": (
                    "Entities should maintain records sufficient to demonstrate compliance with "
                    "the applicable requirements under the rule."
                ),
                "cited_sections": [],
                "source_citations": [first_citation],
            },
        ],
        "affected_entity_types": [
            {
                "entity_type": "Public companies and registered entities subject to SEC oversight",
                "citation": first_citation,
            }
        ],
        "compliance_impact_areas": [
            {
                "area": "Disclosure",
                "linked_obligation_ids": ["OBL-001"],
                "citations": [first_citation],
            },
            {
                "area": "Recordkeeping",
                "linked_obligation_ids": ["OBL-002"],
                "citations": [first_citation],
            },
        ],
        "assumptions": [
            {
                "assumption_text": "Interpretation is based on the provided rule text only",
                "reason": "No additional regulatory context or agency guidance was provided",
                "citation": None,
            }
        ],
    }
