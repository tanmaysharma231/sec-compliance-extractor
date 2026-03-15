from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

CITATION_PATTERN = re.compile(r"^src:\d+$")

VALID_IMPACT_AREAS = {
    "Recordkeeping",
    "Reporting",
    "Disclosure",
    "Internal Controls",
    "Governance",
    "Risk Management",
    "Technology Controls",
}


def _validate_citation_list(value: List[str]) -> List[str]:
    for citation in value:
        if not CITATION_PATTERN.match(citation):
            raise ValueError(f"citation must use src:<index> format, got: {citation!r}")
    return value


class IngestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str                              # URL or local file path
    page_range: Optional[Tuple[int, int]] = None  # 1-based (start, end) for PDFs
    strict_citations: bool = False           # passed through to extraction


class IngestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    chunk_count: int
    artifact_dir: str


class RuleExtractorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_text: str
    strict_citations: bool = False


class RuleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_title: str
    release_number: Optional[str] = None
    publication_date: Optional[str] = None
    effective_date: Optional[str] = None
    citations: List[str] = Field(default_factory=list)

    @field_validator("citations")
    @classmethod
    def validate_citations(cls, value: List[str]) -> List[str]:
        return _validate_citation_list(value)


class RuleSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    citations: List[str] = Field(default_factory=list)

    @field_validator("citations")
    @classmethod
    def validate_citations(cls, value: List[str]) -> List[str]:
        return _validate_citation_list(value)


class KeyObligation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obligation_id: str
    obligation_text: str
    trigger: Optional[str] = None
    deadline: Optional[str] = None
    disclosure_fields: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    cited_sections: List[str] = Field(default_factory=list)
    source_citations: List[str] = Field(default_factory=list)

    @field_validator("source_citations")
    @classmethod
    def validate_source_citations(cls, value: List[str]) -> List[str]:
        return _validate_citation_list(value)


class AffectedEntityType(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_type: str
    citation: str

    @field_validator("citation")
    @classmethod
    def validate_citation(cls, value: str) -> str:
        if not CITATION_PATTERN.match(value):
            raise ValueError(f"citation must use src:<index> format, got: {value!r}")
        return value


class ComplianceImpactArea(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area: str
    linked_obligation_ids: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)

    @field_validator("area")
    @classmethod
    def validate_area(cls, value: str) -> str:
        if value not in VALID_IMPACT_AREAS:
            raise ValueError(
                f"area must be one of: {', '.join(sorted(VALID_IMPACT_AREAS))}. Got: {value!r}"
            )
        return value

    @field_validator("citations")
    @classmethod
    def validate_citations(cls, value: List[str]) -> List[str]:
        return _validate_citation_list(value)


class Assumption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assumption_text: str
    reason: str
    citation: Optional[str] = None

    @field_validator("citation")
    @classmethod
    def validate_citation(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not CITATION_PATTERN.match(value):
            raise ValueError(f"citation must use src:<index> format, got: {value!r}")
        return value


class RuleExtractorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_metadata: RuleMetadata
    rule_summary: RuleSummary
    key_obligations: List[KeyObligation]
    affected_entity_types: List[AffectedEntityType]
    compliance_impact_areas: List[ComplianceImpactArea]
    assumptions: List[Assumption] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Two-pass pipeline models (IngestGraph + locator_pass)
# ---------------------------------------------------------------------------

class Section(BaseModel):
    """A named segment of the source document with hierarchy context."""
    model_config = ConfigDict(extra="forbid")

    section_id: str                  # e.g. "SEC-001"
    heading_path: List[str]          # e.g. ["II.", "A.", "1."]
    level: int                       # heading depth (0 = top-level)
    section_text: str


class RichChunk(BaseModel):
    """A size-bounded text chunk derived from a Section, with scoring metadata."""
    model_config = ConfigDict(extra="forbid")

    src_id: str                      # "src:0", "src:1", ...
    section_id: str
    heading_path: List[str]
    chunk_index_in_section: int
    text: str
    char_len: int
    token_estimate: int              # approx: char_len // 4
    has_dates: bool = False
    has_scope: bool = False
    has_obligations: bool = False
    has_definitions: bool = False
    has_codified_text: bool = False
    has_example: bool = False        # set by scorer; True if chunk contains worked examples/scenarios
    content_type: str = ""           # set by classify step; empty means not yet classified


class LocatorSelection(BaseModel):
    """Output of the Locator LLM pass — src_ids selected by category."""
    model_config = ConfigDict(extra="forbid")

    date_chunks: List[str] = Field(default_factory=list)
    scope_chunks: List[str] = Field(default_factory=list)
    obligation_chunks: List[str] = Field(default_factory=list)
    definition_chunks: List[str] = Field(default_factory=list)
    other_key_chunks: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Classification models (ClassifyGraph)
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

COMPLIANCE_CONTENT_TYPES = {"final_rule_text", "obligation", "definition"}


class SectionClassification(BaseModel):
    """Classification result for one document section."""
    model_config = ConfigDict(extra="ignore")

    section_id: str
    heading_path: List[str]
    content_type: str        # one of VALID_CONTENT_TYPES
    summary: str             # 2-3 sentence plain-English description
    topics: List[str] = Field(default_factory=list)
    useful_for: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Interpretation models (InterpretGraph)
# ---------------------------------------------------------------------------

class ObligationInterpretation(BaseModel):
    """Structured interpretation of a single extracted obligation."""
    model_config = ConfigDict(extra="forbid")

    obligation_id: str
    primary_interpretation: str
    supporting_sections: List[str] = Field(default_factory=list)
    alternative_interpretations: List[str] = Field(default_factory=list)
    ambiguous_terms: List[str] = Field(default_factory=list)
    compliance_implication: str
    confidence_level: Literal["high", "medium", "low"] = "medium"


class InterpretationOutput(BaseModel):
    """Full interpretation run output -- one ObligationInterpretation per obligation."""
    model_config = ConfigDict(extra="forbid")

    run_id: str
    rule_title: str
    interpretations: List[ObligationInterpretation] = Field(default_factory=list)


class DocumentMap(BaseModel):
    """Document-level synthesis produced by the classify pipeline."""
    model_config = ConfigDict(extra="ignore")

    regulatory_objective: str
    rule_title: str
    sections_by_type: dict = Field(default_factory=dict)
    compliance_section_ids: List[str] = Field(default_factory=list)
    cost_section_ids: List[str] = Field(default_factory=list)
    definition_section_ids: List[str] = Field(default_factory=list)
