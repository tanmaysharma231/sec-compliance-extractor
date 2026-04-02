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
            "rule_provision": "specific CFR section or form item imposing this obligation, e.g. '17 CFR 229.106(b)' or 'Item 1.05 Form 8-K'",
            "obligation_text": "One specific, atomic requirement imposed by the rule (one obligation per entry -- do not merge)",
            "trigger": "event or condition that activates this obligation, or null if always active",
            "deadline": "timing requirement if stated (e.g. '4 business days', 'annual', 'within 30 days'), or null",
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
        "- ADOPTED RULES ONLY: Extract only requirements that the SEC is explicitly ADOPTING.\n"
        "  Skip anything the text describes as proposed but not adopted, considered but rejected,\n"
        "  or prefaced with 'we are not adopting', 'we did not propose', 'we are not requiring'.\n"
        "- DISTINCT REQUIREMENTS: Create one obligation entry per independent top-level requirement --\n"
        "  i.e., something a company must affirmatively do or disclose that stands on its own.\n"
        "  Do NOT create separate entries for: safe harbors, exceptions, carve-outs, timing modifiers,\n"
        "  or procedural consequences of another obligation (e.g. 'late filing does not affect S-3\n"
        "  eligibility' is a modifier of the Form 8-K obligation -- put it in key_details there).\n"
        "  Do NOT split one obligation into multiple entries just because it has several sub-elements.\n"
        "- obligation_text: write a complete, self-contained sentence that includes WHO must do WHAT,\n"
        "  by WHEN, and under WHAT condition. Example: 'Registrants must file Form 8-K disclosing a\n"
        "  material cybersecurity incident within 4 business days of determining it is material.'\n"
        "  Do not leave out the deadline or trigger -- a reader of obligation_text alone must get the\n"
        "  full picture. The separate deadline/trigger fields are for machine use and must also be set.\n"
        "- cited_sections: list the specific CFR sections, form items, or rule references that impose\n"
        "  this obligation (e.g. '17 CFR 229.106(b)', 'Form 8-K Item 1.05', 'Rule 13a-11(c)').\n"
        "  Leave empty only if no specific section reference appears in the source chunks.\n"
        "- trigger: the event or condition that activates this obligation (e.g. 'incident determined\n"
        "  material', 'end of fiscal year', 'registrant files annual report'). null if always active.\n"
        "- deadline: the timing requirement if stated in the source (e.g. '4 business days', 'annual',\n"
        "  'within 30 days of fiscal year end'). null if no deadline is specified.\n\n"
        "Generate RuleExtractorOutput JSON now:"
    )


_SECTION_SCHEMA = {
    "key_obligations": _SCHEMA["key_obligations"],
    "affected_entity_types": _SCHEMA["affected_entity_types"],
    "compliance_impact_areas": _SCHEMA["compliance_impact_areas"],
    "assumptions": _SCHEMA["assumptions"],
}


def build_section_extractor_prompt(
    section_heading: str,
    section_chunks: List[RichChunk],
    prior_obligations: List[dict],
    summary_text: str,
    obligation_id_start: int = 1,
    strict_citations: bool = False,
) -> str:
    """Per-section extractor prompt.

    section_heading      -- full heading text for this obligation section
    section_chunks       -- RichChunk objects belonging to this section
    prior_obligations    -- [{"obligation_id": str, "obligation_text": str}] from previous sections
    summary_text         -- document-level summary (first 500 chars used as context)
    obligation_id_start  -- first OBL-N number to use (avoids ID collisions)
    strict_citations     -- whether to require citations on every obligation
    """
    # Document summary block (first 500 chars)
    summary_snippet = summary_text.strip()[:500] if summary_text.strip() else "(no summary available)"

    # Prior obligations block
    if prior_obligations:
        prior_lines = "\n".join(
            f"  {p['obligation_id']}: {p['obligation_text'][:80]}"
            for p in prior_obligations
        )
        prior_block = "OBLIGATIONS ALREADY EXTRACTED (do not re-extract):\n" + prior_lines
    else:
        prior_block = "OBLIGATIONS ALREADY EXTRACTED (do not re-extract):\n  (none yet -- this is the first section)"

    # Source chunks block
    source_blocks = []
    for chunk in section_chunks:
        section_hint = " - ".join(chunk.heading_path) if chunk.heading_path else "UNLABELED"
        source_blocks.append(f"[{chunk.src_id}] (Section: {section_hint})\n{chunk.text}")
    sources_text = "\n\n".join(source_blocks)

    schema_json = json.dumps(_SECTION_SCHEMA, indent=2)

    strict_note = (
        "STRICT MODE: Every KeyObligation must have at least one source_citation. "
        "Every AffectedEntityType must have a citation."
        if strict_citations
        else "Citations are encouraged wherever the source text supports them."
    )

    start_id = f"OBL-{obligation_id_start:03d}"

    return (
        "Extract structured compliance intelligence from one section of an SEC regulatory document.\n\n"
        "DOCUMENT SUMMARY (overall context):\n"
        f"{summary_snippet}\n\n"
        f"{prior_block}\n\n"
        f"SECTION: {section_heading}\n\n"
        f"CITATION RULE: {strict_note}\n\n"
        "SOURCE CHUNKS:\n"
        f"{sources_text}\n\n"
        "OUTPUT SCHEMA (produce exactly this structure as JSON):\n"
        f"{schema_json}\n\n"
        "Rules:\n"
        f"- Start obligation numbering from {start_id} (continue sequentially from there)\n"
        "- Do not duplicate obligations already listed in the OBLIGATIONS ALREADY EXTRACTED block\n"
        "  IMPORTANT: an obligation that applies to a DIFFERENT entity type (e.g., foreign private\n"
        "  issuers vs. domestic registrants, smaller reporting companies vs. large accelerated filers)\n"
        "  is NOT a duplicate -- extract it as a separate obligation even if the content is similar.\n"
        "- Focus exclusively on this section's content\n"
        "- ADOPTED RULES ONLY: Extract only requirements the SEC is explicitly ADOPTING in this section.\n"
        "  Skip anything described as proposed but not adopted, considered but rejected, or prefaced\n"
        "  with 'we are not adopting', 'we did not propose', 'we are not requiring'.\n"
        "- DISTINCT REQUIREMENTS: Create one obligation entry per independent top-level requirement.\n"
        "  Do NOT create separate entries for safe harbors, exceptions, carve-outs, timing modifiers,\n"
        "  or procedural consequences of another obligation -- put those in key_details of the parent.\n"
        "  Do NOT split one obligation into multiple entries just because it has sub-elements.\n"
        "- obligation_id format: OBL-001, OBL-002, ... (sequential, zero-padded)\n"
        "- compliance_impact_areas.area must be one of the exact strings listed in the schema\n"
        "- All src:<N> citations must reference chunk IDs present above\n"
        "- Do not include any text outside the JSON object\n"
        "- Do not use forbidden terms\n"
        "- obligation_text: write a complete, self-contained sentence with WHO must do WHAT,\n"
        "  by WHEN, and under WHAT condition.\n"
        "- cited_sections: list the specific CFR sections or form items imposing this obligation.\n"
        "- trigger: event or condition activating this obligation, or null if always active.\n"
        "- deadline: timing requirement if stated, or null if none.\n\n"
        "Generate SectionExtractOutput JSON now:"
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

def build_context_linker_prompt(
    obligation_text: str,
    obligation_id: str,
    family_chunks: List[dict],
) -> str:
    """
    Cheap linker LLM pass: classify each family chunk as key | supporting | skip
    per obligation. Returns integer indices into family_chunks (0-based).
    """
    n = len(family_chunks)
    # Build aligned table
    header = f"{'idx':<5} | {'role':<10} | {'heading (40 chars)':<40} | {'preview (120 chars)'}"
    separator = "-" * 100
    rows = [header, separator]
    for i, c in enumerate(family_chunks):
        role = c.get("subsection_role", "")[:10]
        heading = c.get("heading", "")[:40]
        preview = c.get("text", "")[:120].replace("\n", " ")
        rows.append(f"{i:<5} | {role:<10} | {heading:<40} | {preview}")
    table_text = "\n".join(rows)

    schema_hint = json.dumps(
        {"key_indices": [0, 2], "supporting_indices": [1], "skip_indices": [3, 4]},
        indent=2,
    )

    return (
        "You are a context pre-filter for an SEC regulatory interpretation pipeline.\n\n"
        f"OBLIGATION [{obligation_id}]:\n{obligation_text}\n\n"
        f"FAMILY CHUNKS (indices 0 to {n - 1}):\n"
        f"{table_text}\n\n"
        "Classify each chunk index as one of:\n"
        "  key         -- directly clarifies THIS obligation: edge cases, SEC answers, worked\n"
        "                 examples, scope limits, timing guidance specific to this requirement\n"
        "  supporting  -- general section context, useful background but not obligation-specific\n"
        "  skip        -- tangential, belongs to a different obligation, economic analysis, procedural\n\n"
        "Rules:\n"
        f"  - Every index 0 to {n - 1} must appear in exactly one list (exhaustive partition)\n"
        "  - When in doubt classify as supporting, not skip\n"
        "  - Output JSON only, no markdown fences\n\n"
        f"OUTPUT SCHEMA:\n{schema_hint}\n\n"
        "Generate ObligationContextLinks JSON now:"
    )


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


def build_interpretation_prompt(
    obligation: dict,
    context_bundle: dict,
    sibling_obligations: list = None,
) -> str:
    """
    Main interpretation prompt: produces ObligationInterpretation JSON
    from obligation + assembled context bundle.

    sibling_obligations: list of {obligation_id, obligation_text} dicts for all
    other obligations in the same run, so the LLM can express parent relationships.
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

    anchor_context_block = ""
    if context_bundle.get("anchor_context"):
        anchor_context_block = (
            "ANCHOR CONTEXT (source text this obligation was extracted from):\n"
            + "\n---\n".join(context_bundle["anchor_context"])
            + "\n\n"
        )

    lookup_results_block = ""
    if context_bundle.get("lookup_results"):
        parts = []
        for term, passages in context_bundle["lookup_results"].items():
            parts.append(f"[Lookup: {term}]\n" + "\n---\n".join(passages))
        lookup_results_block = (
            "TERM LOOKUP RESULTS (document passages retrieved for requested terms):\n"
            + "\n---\n".join(parts)
            + "\n\n"
        )

    cfr_block = ""
    if context_bundle.get("cfr_texts"):
        parts = []
        for ref, text in context_bundle["cfr_texts"].items():
            parts.append(f"[{ref}]\n{text}")
        cfr_block = "FETCHED CFR TEXT:\n" + "\n---\n".join(parts) + "\n\n"

    discussion_block = ""
    if context_bundle.get("discussion"):
        discussion_block = "BIN FINDINGS (scope modifiers, definitions, edge cases):\n" + "\n---\n".join(context_bundle["discussion"]) + "\n\n"

    siblings_block = ""
    if sibling_obligations:
        lines = [
            f"  {s['obligation_id']}: {s['obligation_text'][:100]}"
            for s in sibling_obligations
            if s.get("obligation_id") != obligation.get("obligation_id")
        ]
        if lines:
            siblings_block = (
                "OTHER OBLIGATIONS IN THIS RULE (for relationship context only):\n"
                + "\n".join(lines)
                + "\n\n"
            )

    schema_hint = _json.dumps(
        {
            "obligation_id": obligation.get("obligation_id", "OBL-001"),
            "primary_interpretation": "What this obligation most likely means in practice",
            "key_details": [
                "Each notable exception, carve-out, or safe harbor that modifies the main rule",
                "Scope extensions (e.g. applies to third-party systems, not just registrant-owned)",
                "Procedural details that affect when/how the obligation applies",
            ],
            "supporting_sections": ["CFR section or definition used to reach this interpretation"],
            "alternative_interpretations": ["Another plausible reading if law is ambiguous"],
            "ambiguous_terms": ["term that is unclear or context-dependent"],
            "compliance_implication": "Concrete action the company needs to take",
            "confidence_level": "high | medium | low",
            "needs_more_context": False,
            "lookup_requests": [],
            "parent_obligation_ids": [],
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
        f"{anchor_context_block}"
        f"{lookup_results_block}"
        f"{cfr_block}"
        f"{discussion_block}"
        f"{siblings_block}"
        "Produce a structured interpretation using the schema below.\n\n"
        "Rules:\n"
        "- primary_interpretation: what this obligation most likely means in practice.\n"
        "  If lookup results were provided, ground your interpretation in those passages.\n"
        "  Reference specific SEC reasoning -- do not just restate the obligation text.\n"
        "- key_details: a list of every important nuance visible in the anchor context,\n"
        "  including: exceptions and carve-outs (e.g. national security delay), scope\n"
        "  extensions (e.g. applies to third-party systems), safe harbors (e.g. no\n"
        "  technical detail required), procedural rules (e.g. staged delay extensions),\n"
        "  and eligibility effects (e.g. late filing does not affect shelf registration).\n"
        "  Each item should be one concrete, specific statement. Do not leave this empty\n"
        "  if the anchor context contains any such details.\n"
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
        "  If any context contains a worked example or timeline, reference it.\n"
        "- confidence_level: high if the rule is clear, medium if some ambiguity,\n"
        "  low if significant uncertainty or competing interpretations\n"
        "- supporting_sections: name the CFR sections or definitions you relied on\n"
        "- lookup_requests: list terms you want to search for in the document to resolve\n"
        "  genuine legal ambiguity (e.g. 'material', 'unreasonable delay'). Only list terms\n"
        "  that are unclear from the anchor context above. Leave [] if context is sufficient.\n"
        "  Do NOT request lookups if lookup results are already provided above.\n"
        "- needs_more_context: set true only if the proposed rule background would\n"
        "  materially change the interpretation even after any lookup results\n"
        "- parent_obligation_ids: if this obligation is a special case, exception, or\n"
        "  sub-requirement of another obligation listed above, include those obligation IDs.\n"
        "  Leave [] if this is a standalone requirement.\n"
        "- Do not use forbidden terms: compliant, non-compliant, violation, illegal,\n"
        "  penalty exposure, must fix\n"
        "- Output JSON only, no markdown fences\n\n"
        f"OUTPUT SCHEMA:\n{schema_hint}\n\n"
        "Generate ObligationInterpretation JSON now:"
    )


def build_bin_pass_prompt(
    flagged_chunks: List[dict],
    known_obligations: List[dict],
) -> str:
    """
    Secondary reviewer pass: classifies flagged chunks that may have been missed
    by the main extractor. Looks for gaps not already captured in known_obligations.

    flagged_chunks: list of dicts with keys src_id, heading, text, and flag booleans.
    known_obligations: list of obligation dicts with obligation_id and obligation_text.
    """
    # Build known obligations block
    obl_lines = []
    for i, obl in enumerate(known_obligations, 1):
        obl_id = obl.get("obligation_id", f"OBL-{i:03d}")
        obl_text = obl.get("obligation_text", "")[:150]
        obl_lines.append(f"  {i}. {obl_id}: {obl_text}")
    known_obls_block = "\n".join(obl_lines) if obl_lines else "  (no obligations extracted yet)"

    # Build flagged chunks table
    header = f"{'idx':<5} | {'src_id':<10} | {'heading (40 chars)':<40} | text (first 300 chars)"
    separator = "-" * 110
    rows = [header, separator]
    for i, chunk in enumerate(flagged_chunks):
        src_id = str(chunk.get("src_id", ""))[:10]
        heading = str(chunk.get("heading", ""))[:40]
        text_preview = str(chunk.get("text", ""))[:300].replace("\n", " ")
        rows.append(f"{i:<5} | {src_id:<10} | {heading:<40} | {text_preview}")
    table_text = "\n".join(rows)

    schema_hint = json.dumps(
        {
            "findings": [
                {
                    "finding_type": "scope_modifier",
                    "text": "...",
                    "related_to": ["OBL-001"],
                    "source_chunks": ["src:5"],
                    "notes": "...",
                }
            ]
        },
        indent=2,
    )

    return (
        "You are a secondary reviewer for an SEC regulatory compliance pipeline.\n\n"
        "KNOWN OBLIGATIONS (look for gaps, not duplicates already captured above):\n"
        f"{known_obls_block}\n\n"
        "FLAGGED CHUNKS TO REVIEW:\n"
        f"{table_text}\n\n"
        "For each chunk above, classify it into exactly one of these types:\n"
        "  missed_obligation  -- a genuine new obligation not already captured in known obligations\n"
        "  scope_modifier     -- limits or extends who or what is covered by the rule\n"
        "  implied_requirement -- obligation implied but not explicitly stated\n"
        "  definition         -- key term with a definition applicable across obligations\n"
        "  edge_case          -- specific scenario, SEC comment response, or edge case guidance\n"
        "  not_relevant       -- skip\n\n"
        "For each finding that is NOT not_relevant, produce:\n"
        "  - finding_type: one of the types above\n"
        "  - text: relevant excerpt (max 300 chars)\n"
        "  - related_to: list of obligation IDs this finding applies to (e.g. [\"OBL-001\"])\n"
        "  - source_chunks: list of src_id values\n"
        "  - notes: optional one-sentence explanation\n\n"
        "OUTPUT SCHEMA:\n"
        f"{schema_hint}\n\n"
        "Rules:\n"
        "  - If in doubt, use not_relevant\n"
        "  - Only use missed_obligation if genuinely new and NOT already captured above\n"
        "  - Output JSON only, no markdown fences\n"
        "  - Omit not_relevant findings from output entirely\n\n"
        "Generate BinPassOutput JSON now:"
    )


def build_case_brief_prompt(
    extraction_output: dict,
    bin_findings: list,
    interpretation_output: dict,
    named_section_texts: List[str],
) -> str:
    """
    Produces a plain-text case brief summarising the full compliance picture.

    bin_findings: list of BinFinding dicts.
    interpretation_output: the InterpretationOutput dict (has 'interpretations' list).
    named_section_texts: list of "[heading]\\ntext" strings from effective dates / scope / exemption sections.
    """
    # Rule metadata
    metadata = extraction_output.get("rule_metadata", {})
    rule_title = metadata.get("rule_title", "Unknown Rule")
    release_number = metadata.get("release_number", "")
    effective_date = metadata.get("effective_date", "")

    meta_lines = [f"Rule: {rule_title}"]
    if release_number:
        meta_lines.append(f"Release: {release_number}")
    if effective_date:
        meta_lines.append(f"Effective Date: {effective_date}")
    meta_block = "\n".join(meta_lines)

    # Obligations block
    obligations = extraction_output.get("key_obligations", [])
    obl_lines = []
    for obl in obligations:
        obl_id = obl.get("obligation_id", "")
        rule_provision = obl.get("cited_sections", [])
        provision_str = ", ".join(rule_provision) if rule_provision else "not specified"
        obl_text = obl.get("obligation_text", "")
        deadline = obl.get("deadline") or "not specified"
        obl_lines.append(
            f"  [{obl_id}] {obl_text}\n"
            f"    Rule provision: {provision_str}\n"
            f"    Deadline: {deadline}"
        )
    obligations_block = "\n\n".join(obl_lines) if obl_lines else "  (none extracted)"

    # Bin findings grouped by type
    finding_groups: dict = {}
    for f in bin_findings:
        ftype = f.get("finding_type", "not_relevant")
        if ftype == "not_relevant":
            continue
        finding_groups.setdefault(ftype, []).append(f)

    findings_lines = []
    for ftype, items in finding_groups.items():
        findings_lines.append(f"  [{ftype}]")
        for item in items:
            text = item.get("text", "")[:300]
            related = ", ".join(item.get("related_to", []))
            notes = item.get("notes", "")
            line = f"    - {text}"
            if related:
                line += f" (related: {related})"
            if notes:
                line += f" -- {notes}"
            findings_lines.append(line)
    findings_block = "\n".join(findings_lines) if findings_lines else "  (none)"

    # Interpretation summaries
    interpretations = interpretation_output.get("interpretations", [])
    interp_lines = []
    for interp in interpretations:
        obl_id = interp.get("obligation_id", "")
        primary = interp.get("primary_interpretation", "")[:200]
        implication = interp.get("compliance_implication", "")[:200]
        interp_lines.append(
            f"  [{obl_id}]\n"
            f"    Interpretation: {primary}\n"
            f"    Implication: {implication}"
        )
    interp_block = "\n\n".join(interp_lines) if interp_lines else "  (none)"

    # Named section texts (effective dates, scope, exemptions)
    named_block_parts = []
    for section_text in named_section_texts:
        named_block_parts.append(section_text[:1500])
    named_block = "\n\n---\n\n".join(named_block_parts) if named_block_parts else "(none provided)"

    return (
        "You are an SEC regulatory compliance analyst. Produce a concise case brief for this rule.\n\n"
        "RULE METADATA:\n"
        f"{meta_block}\n\n"
        "OBLIGATIONS:\n"
        f"{obligations_block}\n\n"
        "BIN FINDINGS (additional findings from secondary review):\n"
        f"{findings_block}\n\n"
        "INTERPRETATION SUMMARIES:\n"
        f"{interp_block}\n\n"
        "NAMED SECTIONS (effective dates, scope, exemptions):\n"
        f"{named_block}\n\n"
        "Produce a plain-text case brief with exactly these sections.\n"
        "Use dashes as section headers (e.g. '--- RULE ---'), not markdown.\n\n"
        "  --- RULE ---\n"
        "  Name and release number of the rule.\n\n"
        "  --- APPLIES TO ---\n"
        "  Who is covered (use scope named sections + scope_modifier findings).\n\n"
        "  --- EXEMPTIONS ---\n"
        "  Who is excluded from the rule.\n\n"
        "  --- OBLIGATIONS ---\n"
        "  Numbered list, one per OBL-ID, include rule_provision and deadline.\n\n"
        "  --- KEY DEFINITIONS ---\n"
        "  From definition findings.\n\n"
        "  --- SCOPE MODIFIERS ---\n"
        "  From scope_modifier findings.\n\n"
        "  --- IMPLIED REQUIREMENTS ---\n"
        "  From implied_requirement findings.\n\n"
        "  --- EFFECTIVE DATES ---\n"
        "  From named section texts.\n\n"
        "  --- EDGE CASES ---\n"
        "  From edge_case findings.\n\n"
        "  --- WHAT IT MEANS IN PRACTICE ---\n"
        "  One paragraph per obligation from interpretation summaries.\n\n"
        "Rules:\n"
        "  - Plain text only, no markdown\n"
        "  - Use ASCII separators only (dashes, equals signs)\n"
        "  - Keep total output under 1500 words\n"
        "  - Do not use forbidden terms: compliant, non-compliant, violation, illegal, penalty exposure\n\n"
        "Generate the case brief now:"
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
