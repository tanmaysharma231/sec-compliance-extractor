# System Overview -- SEC Rule Update Compliance Impact Extractor

## Purpose
Takes a raw SEC Federal Register document and extracts structured compliance
intelligence: obligations, affected entities, impact areas, key dates, assumptions,
and per-obligation interpretations backed by live CFR lookups.

---

## Five-Stage Pipeline

### Stage 1 -- IngestGraph (pure Python, no LLM)

```
fetch_document
      |   (URL or local file -> plain text via ingest.py)
      v
segment_document          <- segmenter.py: heading-aware section tree
      |
      v
chunk_sections            <- size-bounded RichChunk objects per section
      |                      target ~1000 tokens, hard cap 1600 tokens, 150-token overlap
      v
score_chunks              <- scorer.py: regex hot-zone flags (dates/scope/obl/def/examples)
      |
      v
extract_summary           <- pulls SUMMARY / SUPPLEMENTARY INFORMATION section text
      |
      v
save_ingest_artifacts
      |
      +-> artifacts/<run_id>/
              input.txt          raw fetched text
              sections.json      List[Section] with heading_path
              chunks.json        List[RichChunk] with metadata + flags
              summary.txt        auto-extracted SUMMARY section
              shortlist.json     compact index rows (src_id, heading, flags, preview)
              ingest_log.txt     metadata (run_id, source, counts, timestamp)
```

### Stage 2 -- ClassifyGraph (one LLM call per section group)

```
load_chunks
      |
      v
classify_sections   <- groups chunks by section, one LLM call per group
      |                -> SectionClassification (content_type, summary, topics)
      v
synthesise_map      <- one LLM call -> DocumentMap
      |                (regulatory_objective, estimated_obligation_count,
      |                 sections_by_type, important_chunks)
      v
save_classify_artifacts
      |
      +-> artifacts/<run_id>/
              control.json       DocumentMap + per-section SectionClassification list
```

Content types (7): `final_rule_text`, `obligation`, `definition`, `commentary`,
`comments`, `economic_analysis`, `procedural`

### Stage 3 -- ExtractGraph (two LLM calls)

```
load_chunks
      |   (reads chunks.json + summary.txt from artifacts/<run_id>/)
      |   (classify filter: keeps final_rule_text + obligation + definition only)
      |   (or: inline rule_text -> sets skip_locator=True)
      v
locator_pass  <- CHEAP LLM: compact index + summary -> LocatorSelection JSON
      |         (skipped in direct/inline-text mode)
      v
extract_structured_fields  <- FULL LLM: only selected_chunks shown
      |
      v
validate_output            <- Pydantic + 4 business rule checks
      | (retry up to 2x)
      v
save_extract_artifacts
      +-> artifacts/<run_id>/
              locator_selection.json   Locator output (selected src_ids by category)
              raw_model_output.txt     Extractor raw response
              validated_output.json    final RuleExtractorOutput
              run_log.txt              execution metadata
              trace.jsonl              structured events
```

### Stage 4 -- InterpretGraph (per-obligation LLM calls)

```
load_extract_output
      |   (reads validated_output.json from artifacts/<run_id>/)
      v
for each KeyObligation:
      |
      v
  build_context           <- definition lookup + surrounding chunk context
      |                      + search_document() over commentary/comments sections
      v
  fetch_cfr               <- tools.py: live eCFR API call for cited CFR section
      |
      v
  resolve_references      <- agentic loop: cheap LLM judge, max 2 hops
      |                      fetches additional CFR sections if needed
      v
  interpret_obligation    <- full LLM: primary interpretation + alternatives
      |                      + ambiguous terms + compliance implication + confidence
      v
save_interpret_artifacts
      +-> artifacts/<run_id>/
              interpretations.json     List[ObligationInterpretation]
```

### Stage 5 -- Report (report_formatter.py)

```
load_interpret_output
      |
      v
render_report             <- markdown: summary block + per-obligation sections
      |                      (description, deadline, cited passages, interpretation,
      |                       compliance implication, confidence level)
      v
write_report              <- stdout or --output <path>
```

---

## Key Modules

| File | Purpose |
|------|---------|
| `ingest.py` | Fetch URL/file -> clean plain text (PDF, HTML) |
| `segmenter.py` | Heading detection -> flat List[Section] |
| `scorer.py` | Regex hot-zone flags -> RichChunk.has_* booleans + has_example |
| `ingest_graph.py` | LangGraph IngestGraph (Stage 1) |
| `classify_graph.py` | LangGraph ClassifyGraph (Stage 2) -- content type labelling |
| `extract_graph.py` | LangGraph ExtractGraph (Stage 3, two LLM passes) |
| `interpret_graph.py` | LangGraph InterpretGraph (Stage 4) -- per-obligation interpretation |
| `tools.py` | CFR fetch (eCFR API), definition lookup, search_document() |
| `prompts.py` | Prompt builders: system, locator, extractor, interpreter, retry |
| `schemas.py` | Pydantic models: Section, RichChunk, LocatorSelection, SectionClassification, DocumentMap, RuleExtractorOutput, ObligationInterpretation |
| `utils.py` | Chunking, JSON parsing, citation/language validators |
| `module.py` | High-level modules: IngestModule, ExtractModule, RuleExtractorModule |
| `report_formatter.py` | Markdown report renderer |
| `comprehend.py` | Calibration tool: classify every chunk, compare locator vs control |
| `cli.py` | CLI: `ingest`, `extract`, `run`, `classify`, `interpret`, `gap`, `report`, `comprehend` |

---

## Schema Hierarchy

```
RuleExtractorOutput
  +-- rule_metadata: RuleMetadata
  +-- rule_summary:  RuleSummary
  +-- key_obligations[]: KeyObligation  (OBL-001, OBL-002, ...)
  +-- affected_entity_types[]: AffectedEntityType
  +-- compliance_impact_areas[]: ComplianceImpactArea
  +-- assumptions[]: Assumption

LocatorSelection
  +-- date_chunks[]: src_id list
  +-- scope_chunks[]
  +-- obligation_chunks[]
  +-- definition_chunks[]
  +-- other_key_chunks[]

RichChunk
  +-- src_id, section_id, heading_path, chunk_index_in_section
  +-- text, char_len, token_estimate
  +-- has_dates, has_scope, has_obligations, has_definitions, has_example
  +-- content_type  (set by ClassifyGraph)

SectionClassification
  +-- section_id, content_type, summary, topics[]

DocumentMap
  +-- regulatory_objective, estimated_obligation_count
  +-- sections_by_type: dict[content_type -> List[section_id]]
  +-- important_chunks[]: src_id list

ObligationInterpretation
  +-- obligation_id
  +-- primary_interpretation
  +-- alternatives[]
  +-- ambiguous_terms[]
  +-- compliance_implication
  +-- confidence_level  (low / medium / high)

InterpretationOutput
  +-- run_id
  +-- interpretations[]: ObligationInterpretation
```

---

## Citation System

- Format: `src:<N>` (0-based global index across full chunk list)
- Validated: bounds check against `len(chunks)` (full list, not selected)
- Strict mode: every obligation and entity type needs >=1 citation
- `enforce_obligation_links(output)` -- linked_obligation_ids must reference real OBL-* ids
- `enforce_safe_language(output)` -- banned terms: compliant, non-compliant, violation,
  illegal, penalty exposure, must fix

---

## Backward Compatibility

- `RuleExtractorModule.run(payload)` -- unchanged; uses direct mode (skip_locator=True)
- All 55 tests pass without any LLM locator call
- Old `chunks.json` format auto-detected and wrapped in minimal RichChunk on load
