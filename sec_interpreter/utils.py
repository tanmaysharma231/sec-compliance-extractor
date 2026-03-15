from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterator, List

_BANNED_LANGUAGE = re.compile(
    r"\b(compliant|non-compliant|violation|illegal|penalty exposure|must fix)\b",
    re.IGNORECASE,
)
_CITATION_PATTERN = re.compile(r"^src:(\d+)$")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object")
    return parsed


def repair_json(raw_text: str) -> str:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def chunk_rule_text(text: str) -> List[str]:
    """Split rule text into paragraph-based chunks of at most ~1500 chars each."""
    if not text or not text.strip():
        return [""]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [""]

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for para in paragraphs:
        # If adding this paragraph would exceed the limit and we already have content,
        # flush the current chunk first
        if current_parts and current_len + len(para) + 2 > 1500:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0

        # If a single paragraph is itself > 1500 chars, it gets its own chunk
        if not current_parts and len(para) > 1500:
            chunks.append(para)
            continue

        current_parts.append(para)
        current_len += len(para) + 2  # +2 for the "\n\n" separator

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks if chunks else [""]


def enforce_citation_bounds(output: Any, chunk_count: int) -> None:
    """Ensure every src:<N> index satisfies 0 <= N < chunk_count."""
    for text_value in _iter_text_fields(output):
        for match in re.finditer(r"src:(\d+)", text_value):
            idx = int(match.group(1))
            if idx < 0 or idx >= chunk_count:
                raise ValueError(
                    f"Citation index out of range: src:{idx} (chunk_count={chunk_count})"
                )

    # Also check structured citation fields explicitly
    _check_citation_fields(output, chunk_count)


def _check_citation_fields(output: Any, chunk_count: int) -> None:
    from .schemas import RuleExtractorOutput

    if not isinstance(output, RuleExtractorOutput):
        return

    all_citations: list[str] = []
    all_citations.extend(output.rule_metadata.citations)
    all_citations.extend(output.rule_summary.citations)
    for obl in output.key_obligations:
        all_citations.extend(obl.source_citations)
    for ent in output.affected_entity_types:
        all_citations.append(ent.citation)
    for area in output.compliance_impact_areas:
        all_citations.extend(area.citations)
    for assumption in output.assumptions:
        if assumption.citation:
            all_citations.append(assumption.citation)

    for citation in all_citations:
        m = _CITATION_PATTERN.match(citation)
        if m:
            idx = int(m.group(1))
            if idx < 0 or idx >= chunk_count:
                raise ValueError(
                    f"Citation index out of range: {citation} (chunk_count={chunk_count})"
                )


def enforce_strict_citations(output: Any, payload: Any) -> None:
    """When strict_citations=True: every KeyObligation needs ≥1 source_citation,
    every AffectedEntityType needs a citation."""
    if not payload.strict_citations:
        return

    from .schemas import RuleExtractorOutput

    if not isinstance(output, RuleExtractorOutput):
        return

    for obl in output.key_obligations:
        if not obl.source_citations:
            raise ValueError(
                f"Obligation {obl.obligation_id} must include at least one source_citation "
                "when strict_citations is true"
            )

    for ent in output.affected_entity_types:
        if not ent.citation:
            raise ValueError(
                f"AffectedEntityType {ent.entity_type!r} must include a citation "
                "when strict_citations is true"
            )


def enforce_obligation_links(output: Any) -> None:
    """Every ComplianceImpactArea.linked_obligation_ids must reference a real obligation_id."""
    from .schemas import RuleExtractorOutput

    if not isinstance(output, RuleExtractorOutput):
        return

    known_ids = {obl.obligation_id for obl in output.key_obligations}
    for area in output.compliance_impact_areas:
        for linked_id in area.linked_obligation_ids:
            if linked_id not in known_ids:
                raise ValueError(
                    f"ComplianceImpactArea {area.area!r} references unknown obligation_id: "
                    f"{linked_id!r}. Known IDs: {sorted(known_ids)}"
                )


def enforce_safe_language(output: Any) -> None:
    """Scan all text fields for banned terms."""
    for text_value in _iter_text_fields(output):
        if _BANNED_LANGUAGE.search(text_value):
            raise ValueError(
                f"Output violates safe language rules. Banned term found in: {text_value!r}"
            )


def _iter_text_fields(output: Any) -> Iterator[str]:
    """Yield all string fields from RuleExtractorOutput."""
    from .schemas import RuleExtractorOutput

    if not isinstance(output, RuleExtractorOutput):
        return

    yield output.rule_metadata.rule_title
    if output.rule_metadata.release_number:
        yield output.rule_metadata.release_number
    if output.rule_metadata.publication_date:
        yield output.rule_metadata.publication_date
    if output.rule_metadata.effective_date:
        yield output.rule_metadata.effective_date

    yield output.rule_summary.summary

    for obl in output.key_obligations:
        yield obl.obligation_id
        yield obl.obligation_text
        for sec in obl.cited_sections:
            yield sec

    for ent in output.affected_entity_types:
        yield ent.entity_type

    for area in output.compliance_impact_areas:
        yield area.area
        for linked_id in area.linked_obligation_ids:
            yield linked_id

    for assumption in output.assumptions:
        yield assumption.assumption_text
        yield assumption.reason
