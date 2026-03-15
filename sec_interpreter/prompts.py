from __future__ import annotations

import json
from typing import List

from .schemas import LocatorSelection, RichChunk, RuleExtractorInput, VALID_IMPACT_AREAS, VALID_CONTENT_TYPES

_SCHEMA = {
    "rule_metadata": {
        "rule_title": "string",
        "release_number": "string or null",
        "publication_date": "string or null (YYYY-MM-DD)",
        "effective_date": "string or null (YYYY-MM-DD)",
        "citations": ["src:0"],
    },
    "rule_summary": {
        "summary": "3-5 sentence plain-English summary of the rule",
        "citations": ["src:0", "src:1"],
    },
    "key_obligations": [
        {
            "obligation_id": "OBL-001",
            "obligation_text": "One specific, atomic requirement imposed by the rule (one obligation per entry -- do not merge)",
            "trigger": "event or condition that activates this obligation, or null if always active",
            "deadline": "timing requirement if stated (e.g. '4 business days', 'annual', 'within 30 days'), or null",
            "disclosure_fields": ["specific data item required to be reported or disclosed"],
            "evidence": ["document or record that proves this obligation was met"],
            "cited_sections": ["17 CFR 229.106", "Form 8-K Item 1.05", "Rule 13a-11(c)"],
            "source_citations": ["src:0"],
        }
    ],
    "affected_entity_types": [
        {
            "entity_type": "Public company, investment adviser, broker-dealer, etc.",
            "citation": "src:0",
        }
    ],
    "compliance_impact_areas": [
        {
            "area": "one of: " + ", ".join(sorted(VALID_IMPACT_AREAS)),
            "linked_obligation_ids": ["OBL-001"],
            "citations": ["src:0"],
        }
    ],
    "assumptions": [
        {
            "assumption_text": "State the assumption clearly",
            "reason": "Why this assumption is necessary",
            "citation": "src:0 or null",
        }
    ],
}

_FORBIDDEN_TERMS = [
    "compliant",
    "non-compliant",
    "violation",
    "illegal",
    "penalty exposure",
    "must fix",
]


def build_system_prompt() -> str:
    forbidden = ", ".join(f'"{t}"' for t in _FORBIDDEN_TERMS)
    return (
        "You are a SEC Regulatory Intelligence Extractor. "
        "Your task is to read SEC regulatory documents and extract structured compliance intelligence. "
        "Output valid JSON only — no markdown fences, no prose, no commentary. "
        "Follow the exact output schema provided in the user prompt. "
        f"FORBIDDEN TERMS — never use these words in any field: {forbidden}. "
        "Instead use: risk, gap, needs review, likely applicable, may require, should consider. "
        "Citations must use the format src:<chunk_id> (e.g., src:0, src:1) "
        "where chunk_id corresponds to the [src:N] labels in the provided text. "
        "Only cite chunk IDs that actually appear in the provided source text. "
        "obligation_id values must follow the format OBL-001, OBL-002, etc. (zero-padded, three digits). "
        "compliance_impact_areas.area must be exactly one of the valid values listed in the schema. "
        "EXHAUSTIVENESS: Extract every distinct requirement as a separate obligation entry. "
        "Do not consolidate or merge separate requirements into one. "
        "Each obligation must describe exactly one atomic requirement."
    )


def build_retry_user_prompt(payload: RuleExtractorInput, chunks: List[str], error: str) -> str:
    """Retry variant — embeds the previous validation error so the model can self-correct."""
    base = build_user_prompt(payload, chunks)
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


def build_locator_prompt(summary_text: str, index_rows: List[dict]) -> str:
    """Cheap Locator LLM pass: sees compact index + summary, picks relevant src_ids.

    Three independent signals per chunk (heading, flags, preview) so the Locator
    works correctly even when any single signal is weak or missing.
    """
    # Build aligned table rows
    table_lines = [
        "src_id  | section | flags | chars | preview (first 200 chars)",
        "flags legend: [dates] [scope] [obl] [def] [codified]",
        "[codified] = legally operative CFR amendment text -- ALWAYS include in obligation_chunks",
    ]
    table_lines.append("-" * 100)
    for row in index_rows:
        flags_str = f"[{', '.join(row['flags'])}]" if row["flags"] else "[]"
        heading_trunc = row["heading"][:40] if len(row["heading"]) > 40 else row["heading"]
        preview_trunc = row["preview"][:100] if len(row["preview"]) > 100 else row["preview"]
        table_lines.append(
            f"{row['src_id']:<8} | {heading_trunc:<42} | {flags_str:<16} | {row['chars']:>5} | {preview_trunc}"
        )
    table_text = "\n".join(table_lines)

    summary_block = summary_text.strip() if summary_text.strip() else "(no summary available)"

    schema_hint = json.dumps(
        {
            "date_chunks": ["src:N"],
            "scope_chunks": ["src:N"],
            "obligation_chunks": ["src:N"],
            "definition_chunks": ["src:N"],
            "other_key_chunks": ["src:N"],
        },
        indent=2,
    )

    return (
        "You are a document chunk selector for an SEC regulatory compliance pipeline.\n\n"
        "DOCUMENT SUMMARY:\n"
        f"{summary_block}\n\n"
        "CHUNK INDEX (src_id | section | flags | chars | preview):\n"
        f"{table_text}\n\n"
        "Select the src_ids most relevant to extracting compliance intelligence:\n"
        "  - date_chunks       -> chunks containing effective/compliance dates\n"
        "  - scope_chunks      -> chunks describing who is covered by the rule\n"
        "  - obligation_chunks -> chunks with specific obligations (must/shall/required)\n"
        "  - definition_chunks -> chunks containing key definitions\n"
        "  - other_key_chunks  -> other important context (max 5 src_ids)\n\n"
        "PRIORITY RULE: Any chunk with [codified] in its flags contains legally operative CFR\n"
        "text (e.g. 'Add paragraph 229.106 to read as follows'). Always include ALL [codified] chunks\n"
        "in obligation_chunks regardless of other signals.\n"
        "Avoid picking chunks whose flags show [obl] but not [codified] unless the preview\n"
        "clearly shows a specific operative requirement (not SEC discussion or commentary).\n\n"
        "Rules:\n"
        "  - Total selected src_ids across all categories must not exceed 60\n"
        "  - obligation_chunks must be non-empty\n"
        "  - Only use src_ids that appear in the CHUNK INDEX above\n"
        "  - Output JSON only, matching this schema:\n"
        f"{schema_hint}\n\n"
        "Generate LocatorSelection JSON now:"
    )


def build_extractor_prompt(payload: RuleExtractorInput, selected_chunks: List[RichChunk]) -> str:
    """Extractor prompt variant — uses RichChunk objects with heading context."""
    source_blocks = []
    for chunk in selected_chunks:
        section_hint = " - ".join(chunk.heading_path) if chunk.heading_path else "UNLABELED"
        source_blocks.append(f"[{chunk.src_id}] (Section: {section_hint})\n{chunk.text}")
    sources_text = "\n\n".join(source_blocks)

    schema_json = json.dumps(_SCHEMA, indent=2)

    strict_note = (
        "STRICT MODE: Every KeyObligation must have at least one source_citation. "
        "Every AffectedEntityType must have a citation."
        if payload.strict_citations
        else "Citations are encouraged wherever the source text supports them."
    )

    return (
        "Extract structured compliance intelligence from the following SEC regulatory document chunks.\n\n"
        f"CITATION RULE: {strict_note}\n\n"
        "SOURCE CHUNKS:\n"
        f"{sources_text}\n\n"
        "OUTPUT SCHEMA (produce exactly this structure as JSON):\n"
        f"{schema_json}\n\n"
        "Rules:\n"
        "- obligation_id format: OBL-001, OBL-002, ... (sequential, zero-padded)\n"
        "- compliance_impact_areas.area must be one of the exact strings listed in the schema\n"
        "- All src:<N> citations must reference chunk IDs present above\n"
        "- Do not include any text outside the JSON object\n"
        "- Do not use forbidden terms\n"
        "- EXHAUSTIVENESS: create one obligation entry per distinct requirement found in the source\n"
        "  chunks — do not merge separate requirements. If you find 8 requirements, produce 8 entries.\n"
        "- cited_sections: list the specific CFR sections, form items, or rule references that impose\n"
        "  this obligation (e.g. '17 CFR 229.106(b)', 'Form 8-K Item 1.05', 'Rule 13a-11(c)').\n"
        "  Leave empty only if no specific section reference appears in the source chunks.\n"
        "- trigger: the event or condition that activates this obligation (e.g. 'incident determined\n"
        "  material', 'end of fiscal year', 'registrant files annual report'). null if always active.\n"
        "- deadline: the timing requirement if stated in the source (e.g. '4 business days', 'annual',\n"
        "  'within 30 days of fiscal year end'). null if no deadline is specified.\n"
        "- disclosure_fields: list the specific data items this obligation requires to be reported\n"
        "  (e.g. ['nature of incident', 'scope', 'timing', 'material impact']). Empty if not a\n"
        "  disclosure obligation.\n"
        "- evidence: list the documents or records that would prove this obligation was met\n"
        "  (e.g. ['Form 8-K filing', 'board meeting minutes', 'incident investigation report']).\n\n"
        "Generate RuleExtractorOutput JSON now:"
    )


def build_user_prompt(payload: RuleExtractorInput, chunks: List[str]) -> str:
    # Build numbered source blocks
    source_blocks = []
    for idx, chunk in enumerate(chunks):
        source_blocks.append(f"[src:{idx}]\n{chunk}")
    sources_text = "\n\n".join(source_blocks)

    schema_json = json.dumps(_SCHEMA, indent=2)

    strict_note = (
        "STRICT MODE: Every KeyObligation must have at least one source_citation. "
        "Every AffectedEntityType must have a citation."
        if payload.strict_citations
        else "Citations are encouraged wherever the source text supports them."
    )

    return (
        "Extract structured compliance intelligence from the following SEC regulatory document chunks.\n\n"
        f"CITATION RULE: {strict_note}\n\n"
        "SOURCE CHUNKS:\n"
        f"{sources_text}\n\n"
        "OUTPUT SCHEMA (produce exactly this structure as JSON):\n"
        f"{schema_json}\n\n"
        "Rules:\n"
        "- obligation_id format: OBL-001, OBL-002, ... (sequential, zero-padded)\n"
        "- compliance_impact_areas.area must be one of the exact strings listed in the schema\n"
        "- All src:<N> citations must reference chunk IDs present above\n"
        "- Do not include any text outside the JSON object\n"
        "- Do not use forbidden terms\n\n"
        "Generate RuleExtractorOutput JSON now:"
    )


# ---------------------------------------------------------------------------
# Gap analysis prompt
# ---------------------------------------------------------------------------

def build_gap_analysis_prompt(extracted_output: dict, company_context: str = "") -> str:
    """Prompt to produce a plain-English gap analysis from extracted obligations."""
    rule_title = extracted_output.get("rule_metadata", {}).get("rule_title", "Unknown Rule")
    summary = extracted_output.get("rule_summary", {}).get("summary", "")
    obligations = extracted_output.get("key_obligations", [])
    entities = [e.get("entity_type", "") for e in extracted_output.get("affected_entity_types", [])]

    obl_lines = []
    for obl in obligations:
        lines = [f"  [{obl['obligation_id']}] {obl['obligation_text']}"]
        if obl.get("trigger"):
            lines.append(f"    Trigger: {obl['trigger']}")
        if obl.get("deadline"):
            lines.append(f"    Deadline: {obl['deadline']}")
        if obl.get("disclosure_fields"):
            lines.append(f"    Must disclose: {', '.join(obl['disclosure_fields'])}")
        if obl.get("evidence"):
            lines.append(f"    Evidence needed: {', '.join(obl['evidence'])}")
        if obl.get("cited_sections"):
            lines.append(f"    CFR: {', '.join(obl['cited_sections'])}")
        obl_lines.append("\n".join(lines))

    obligations_text = "\n\n".join(obl_lines)
    entities_text = ", ".join(entities) if entities else "regulated entities"

    company_block = (
        f"\nCOMPANY CONTEXT:\n{company_context.strip()}\n"
        if company_context.strip()
        else "\nCOMPANY CONTEXT:\nA company expanding into US securities markets and seeking to understand what compliance infrastructure this rule requires.\n"
    )

    return (
        f"You are a compliance advisor helping a company understand what they need to do to comply with a new SEC rule.\n\n"
        f"RULE: {rule_title}\n\n"
        f"SUMMARY: {summary}\n\n"
        f"APPLIES TO: {entities_text}\n"
        f"{company_block}\n"
        f"OBLIGATIONS EXTRACTED FROM THE RULE:\n\n"
        f"{obligations_text}\n\n"
        f"Based on the obligations above, produce a gap analysis report in plain English.\n\n"
        f"The report must have exactly these sections:\n\n"
        f"1. WHAT THIS RULE REQUIRES (2-3 sentences, plain English overview)\n\n"
        f"2. OBLIGATIONS SUMMARY (numbered list -- one line per obligation with deadline if applicable)\n\n"
        f"3. WHAT YOU NEED TO BUILD OR HAVE IN PLACE\n"
        f"   For each process, policy, or system the company needs, write:\n"
        f"   - What it is\n"
        f"   - Which obligations it satisfies\n"
        f"   - What evidence it produces\n\n"
        f"4. PRIORITY ACTIONS (top 3-5 concrete first steps, ordered by urgency)\n\n"
        f"Rules:\n"
        f"- Write for a business audience, not a legal one\n"
        f"- Be specific and actionable -- avoid vague statements like 'ensure compliance'\n"
        f"- Do not use the words: compliant, non-compliant, violation, illegal, penalty exposure\n"
        f"- Plain text only -- no markdown headers, no bullet symbols other than dashes\n"
        f"- Keep the total report under 600 words\n\n"
        f"Generate the gap analysis report now:"
    )


# ---------------------------------------------------------------------------
# Interpretation pipeline prompts
# ---------------------------------------------------------------------------

def build_reference_judge_prompt(
    obligation_text: str,
    context_so_far: str,
    refs_seen: list,
) -> str:
    """
    Cheap routing prompt: decide whether more CFR fetching is needed.

    Returns "SUFFICIENT" or a specific reference string to fetch next.
    Only references that appear verbatim in context_so_far are valid choices.
    """
    refs_text = "\n".join(f"  - {r}" for r in refs_seen) if refs_seen else "  (none found)"
    return (
        "You are helping interpret an SEC regulatory obligation.\n\n"
        f"OBLIGATION:\n{obligation_text}\n\n"
        f"CONTEXT ASSEMBLED SO FAR:\n{context_so_far[:4000]}\n\n"
        f"CFR REFERENCES FOUND IN FETCHED TEXT:\n{refs_text}\n\n"
        "TASK: Decide if the context above is sufficient to interpret this obligation.\n\n"
        "If sufficient, respond with exactly: SUFFICIENT\n\n"
        "If there is one specific reference listed above that is critical to understanding\n"
        "this obligation and is NOT yet covered in the context, respond with that reference\n"
        "string exactly as it appears in the list above (e.g. '17 CFR 240.13a-11').\n\n"
        "Rules:\n"
        "- Only choose from the references listed above -- do not invent new ones\n"
        "- When in doubt, respond SUFFICIENT\n"
        "- Respond with a single line only\n\n"
        "Your response:"
    )


def build_interpretation_prompt(obligation: dict, context_bundle: dict) -> str:
    """
    Main interpretation prompt: produces ObligationInterpretation JSON
    from obligation + assembled context bundle.
    """
    import json as _json

    obl_text = obligation.get("obligation_text", "")
    trigger = obligation.get("trigger") or "not specified"
    deadline = obligation.get("deadline") or "not specified"
    cited = ", ".join(obligation.get("cited_sections", [])) or "none"

    definitions_block = ""
    if context_bundle.get("definitions"):
        defs = context_bundle["definitions"]
        definitions_block = "DEFINITIONS FROM DOCUMENT:\n" + "\n---\n".join(defs) + "\n\n"

    surrounding_block = ""
    if context_bundle.get("surrounding"):
        surrounding_block = "SURROUNDING CONTEXT:\n" + "\n---\n".join(context_bundle["surrounding"]) + "\n\n"

    cfr_block = ""
    if context_bundle.get("cfr_texts"):
        parts = []
        for ref, text in context_bundle["cfr_texts"].items():
            parts.append(f"[{ref}]\n{text}")
        cfr_block = "FETCHED CFR TEXT:\n" + "\n---\n".join(parts) + "\n\n"

    discussion_block = ""
    if context_bundle.get("discussion"):
        discussion_block = "DISCUSSION CONTEXT (SEC commentary and public comments):\n" + "\n---\n".join(context_bundle["discussion"]) + "\n\n"

    schema_hint = _json.dumps(
        {
            "obligation_id": obligation.get("obligation_id", "OBL-001"),
            "primary_interpretation": "What this obligation most likely means in practice",
            "supporting_sections": ["CFR section or definition used to reach this interpretation"],
            "alternative_interpretations": ["Another plausible reading if law is ambiguous"],
            "ambiguous_terms": ["term that is unclear or context-dependent"],
            "compliance_implication": "Concrete action the company needs to take",
            "confidence_level": "high | medium | low",
        },
        indent=2,
    )

    return (
        "You are an SEC regulatory compliance analyst interpreting an extracted obligation.\n\n"
        f"OBLIGATION: {obl_text}\n"
        f"Trigger: {trigger}\n"
        f"Deadline: {deadline}\n"
        f"CFR References: {cited}\n\n"
        f"{definitions_block}"
        f"{surrounding_block}"
        f"{cfr_block}"
        f"{discussion_block}"
        "Produce a structured interpretation using the schema below.\n\n"
        "Rules:\n"
        "- primary_interpretation: what this obligation most likely means in practice,\n"
        "  grounded in the definitions and CFR text provided\n"
        "- alternative_interpretations: list other plausible readings where the law is\n"
        "  genuinely ambiguous -- do not give one definitive answer where uncertainty exists\n"
        "- ambiguous_terms: list terms in the obligation text that are legally unclear\n"
        "  (e.g. 'material', 'promptly', 'reasonable') and briefly note why\n"
        "- compliance_implication: describe what the company must BUILD OR CHANGE -- include:\n"
        "  (1) the process or control that needs to be created or updated,\n"
        "  (2) the teams that must be involved (e.g. cybersecurity, legal, finance, disclosure),\n"
        "  (3) the operational challenge (time pressure, facts still unfolding,\n"
        "      coordination across functions).\n"
        "  Use precise legal language from the rule -- do not paraphrase.\n"
        "  If the discussion context contains a worked example or timeline, reference it\n"
        "  to make the implication concrete.\n"
        "- confidence_level: high if the rule is clear, medium if some ambiguity,\n"
        "  low if significant uncertainty or competing interpretations\n"
        "- supporting_sections: name the CFR sections or definitions you relied on\n"
        "- Do not use forbidden terms: compliant, non-compliant, violation, illegal,\n"
        "  penalty exposure, must fix\n"
        "- Output JSON only, no markdown fences\n\n"
        f"OUTPUT SCHEMA:\n{schema_hint}\n\n"
        "Generate ObligationInterpretation JSON now:"
    )


# ---------------------------------------------------------------------------
# Classify pipeline prompts
# ---------------------------------------------------------------------------

def build_section_classify_prompt(heading_path: List[str], section_text: str) -> str:
    """Prompt to classify one document section by content type."""
    valid_types = " | ".join(sorted(VALID_CONTENT_TYPES))
    heading_str = " > ".join(heading_path) if heading_path else "UNLABELED"
    schema_hint = json.dumps(
        {
            "content_type": "one of: " + valid_types,
            "summary": "2-3 sentence plain-English description of what this section contains",
            "topics": ["topic1", "topic2"],
            "useful_for": ["compliance", "cost", "exemption", "context"],
        },
        indent=2,
    )
    return (
        "You are an SEC regulatory document classifier.\n\n"
        f"SECTION HEADING: {heading_str}\n\n"
        "SECTION TEXT (truncated to 6000 chars):\n"
        f"{section_text[:6000]}\n\n"
        "Classify this section using the schema below.\n\n"
        "content_type must be exactly one of:\n"
        "  final_rule_text  -- legally operative CFR/rule text (e.g. 'Add Section 229.106 to read...')\n"
        "  obligation       -- actionable requirements described in discussion form (must/shall/required)\n"
        "  definition       -- key defined terms and their meanings\n"
        "  commentary       -- SEC reasoning, policy discussion, legislative history\n"
        "  comments         -- industry comment letters and SEC responses\n"
        "  economic_analysis -- cost-benefit analysis, quantitative estimates\n"
        "  procedural       -- administrative boilerplate, table of contents, effective dates notices\n\n"
        "useful_for values (select all that apply):\n"
        "  compliance  -- needed for obligations/disclosure extraction\n"
        "  cost        -- needed for compliance cost analysis\n"
        "  exemption   -- needed for understanding who is exempt\n"
        "  context     -- background, useful for understanding but not for extracting requirements\n\n"
        "OUTPUT SCHEMA:\n"
        f"{schema_hint}\n\n"
        "Output JSON only. No markdown fences. No prose.\n"
        "Generate SectionClassification JSON now:"
    )


def build_document_synthesis_prompt(section_summaries: List[dict]) -> str:
    """Prompt to synthesise a DocumentMap from all section classification summaries.

    Only asks the LLM for narrative fields (regulatory_objective, rule_title,
    sections_by_type). Task-specific section_id lists are computed deterministically
    in save_classify_artifacts -- not by the LLM.
    """
    summaries_text = "\n\n".join(
        f"[{s['section_id']}] ({s['content_type']}) {' > '.join(s.get('heading_path', []))}\n"
        f"Summary: {s['summary']}"
        for s in section_summaries
    )
    schema_hint = json.dumps(
        {
            "regulatory_objective": "One sentence describing what this regulatory document accomplishes",
            "rule_title": "Official title of the rule or release",
            "sections_by_type": {
                "final_rule_text": ["SEC-001"],
                "obligation": ["SEC-005"],
                "definition": ["SEC-010"],
                "commentary": ["SEC-020"],
                "comments": ["SEC-030"],
                "economic_analysis": ["SEC-040"],
                "procedural": ["SEC-002"],
            },
        },
        indent=2,
    )
    return (
        "You are an SEC regulatory document analyst.\n\n"
        "Below are summaries of every section in a regulatory document, "
        "already classified by content type.\n\n"
        "SECTION SUMMARIES:\n"
        f"{summaries_text}\n\n"
        "Based on the section summaries above, produce a DocumentMap that:\n"
        "1. States the regulatory objective in one sentence\n"
        "2. Lists the official rule title\n"
        "3. Groups ALL section_ids by their content_type (include every section_id listed above)\n\n"
        "OUTPUT SCHEMA:\n"
        f"{schema_hint}\n\n"
        "Output JSON only. No markdown fences. No prose.\n"
        "Generate DocumentMap JSON now:"
    )
