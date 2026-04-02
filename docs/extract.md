# Extraction Pipeline

## Overview

Stage 2 of the SEC compliance extraction system. Takes ingest artifacts from Stage 1
and produces structured compliance intelligence (obligations, entities, impact areas).

Two LLM-intensive paths:
- **Structure-per-call path** (default): one LLM call per obligation section, partial results merged
- **Direct path** (fallback): single LLM call over all selected chunks; used in tests

Four primary components:
1. `structure_scan` -- deterministic heading scan; no LLM; identifies extraction targets
2. `extract_sections_loop` -- section-per-call LLM extraction with cross-section context
3. `validate_output` -- schema + business-rule checks with retry on failure
4. `gap_check` -- post-extraction gap detection against structure scan expected count

---

## Files

| File | Responsibility |
|------|----------------|
| `sec_interpreter/extract_graph.py` | LangGraph state machine, all extraction nodes, merge helpers |
| `sec_interpreter/structure.py` | `structure_scan()`, `gap_check()` -- deterministic doc structure mapping |
| `sec_interpreter/prompts.py` | `build_extractor_prompt`, `build_section_extractor_prompt`, `build_gap_analysis_prompt` |

---

## Graph Topology

```
START -> load_chunks
           |-- skip_locator=True  -> extract_structured_fields -> validate_output
           |-- skip_locator=False -> structure_scan_pass
                                       |-- obligation_sections non-empty -> extract_sections_loop -> validate_output
                                       |-- obligation_sections empty     -> extract_structured_fields -> validate_output

validate_output
  |-- success       -> save_extract_artifacts -> END
  |-- failure, retries left -> increment_retry
                                 |-- came from loop   -> extract_sections_loop
                                 |-- came from single -> extract_structured_fields
```

`MAX_RETRIES = 2`. Each retry injects the previous validation error into the prompt.

---

## extract_graph.py

### `ExtractState` (TypedDict)

```
run_id                   str                     -- ties to ingest artifacts
payload                  RuleExtractorInput       -- rule_text + strict_citations
chunks                   List[RichChunk]           -- full chunk list (citation bounds)
summary_text             str                      -- loaded from artifacts/summary.txt
scan_result              StructureScanResult      -- output of structure_scan_pass
selected_chunks          List[RichChunk]           -- chunks passed to extractor
skip_locator             bool                     -- True in direct/inline-text mode
raw_output               str                      -- raw LLM JSON string
output                   Optional[RuleExtractorOutput]
retry_count              int
last_error               Optional[str]
token_usage              dict                     -- accumulated {locator, extractor} usage
section_partial_outputs  List[dict]               -- raw dicts from per-section calls
```

---

### Node: `load_chunks`

Two modes:

**Artifact mode** (`run_id` provided, `payload=None`):
- Reads `chunks.json` from `artifacts/{run_id}/`
- Handles backward compat: old format `{id, text}` wrapped in minimal `RichChunk`
- Reads `input.txt` (full rule text for payload) and `summary.txt`
- Sets `skip_locator=False` (structure scan will run)

**Direct mode** (`payload.rule_text` provided):
- Calls `chunk_rule_text(rule_text)` to split in memory
- Wraps each chunk in a minimal `RichChunk` with `section_id="SEC-INLINE"`
- Sets `skip_locator=True` (bypasses structure scan -- used in tests)

---

### Node: `structure_scan_pass`

Calls `structure_scan(artifact_dir)` and selects chunks for extraction.

**Chunk selection:**
1. Starts with `scan_result.structured_chunk_ids` (final/codified chunks from obligation sections)
2. Adds any `named_section_chunk_ids` not already included (dates, scope, exemptions)
3. Always includes `src:0` (cover page -- has title, release number, effective date)
4. Sorts by document order (original chunk index)

**Fallback:** If scan returns no `structured_chunk_ids` (old artifacts lacking
`subsection_role`/`section_family`), falls back to all chunks with a warning.

---

### Node: `extract_sections_loop`

Main extraction loop. Runs one LLM call per `ObligationSection` from the scan result.

**First section (index 0) -- full schema:**
- Uses `build_extractor_prompt()` with `RuleExtractorOutput` schema
- Augments section chunks with `named_section_chunk_ids` + `src:0` so `rule_metadata`
  and `rule_summary` get populated from rich context
- Uses `structured_llm` (LangChain `with_structured_output(RuleExtractorOutput)`)
- Falls back to manual JSON parse if structured output fails

**Subsequent sections -- partial schema:**
- Uses `build_section_extractor_prompt()` with `SectionExtractOutput` schema
  (no `rule_metadata`/`rule_summary`)
- Passes `prior_obligations` (ID + first 80 chars of text) as cross-section context
  to prevent re-extraction of already-found obligations
- `obligation_id_start` set to `len(prior_obligations) + 1` to avoid ID collisions
- Uses `section_structured_llm` (LangChain `with_structured_output(SectionExtractOutput)`)

**After loop:**
- Calls `_merge_section_outputs(parsed_partials, chunks)` to combine all partials
- Returns `{"raw_output": merged_json, "output": None}` -- lets `validate_output` do final parse
- Accumulates token usage across all section calls

---

### Node: `extract_structured_fields`

Single-call fallback extractor. Used when:
- `skip_locator=True` (direct/test mode)
- `structure_scan` found no obligation sections

Calls `build_extractor_prompt(payload, selected_chunks)`. Prefers structured output
via `with_structured_output(RuleExtractorOutput)`; falls back to manual parse.

On retry, wraps prompt with `_build_retry_prompt()` which prepends the previous
validation error and instructions to fix it.

---

### Node: `validate_output`

Two paths:

**Pre-parsed path** (structured output gave us a Pydantic object):
- Runs business rule checks only: citation bounds, strict citations, obligation links, safe language

**Manual parse path**:
1. `parse_json_object(raw_output)` -- first attempt
2. `repair_json(raw_output)` + `parse_json_object()` -- repair attempt on failure
3. `RuleExtractorOutput.model_validate(parsed)` -- Pydantic validation
4. Business rule checks

Returns `{"output": None, "last_error": str}` on any failure, triggering retry routing.

---

### Node: `save_extract_artifacts`

Writes artifacts and runs `gap_check`.

**Artifacts written:**

| File | Contents |
|------|----------|
| `raw_model_output.txt` | Raw LLM JSON string |
| `validated_output.json` | `RuleExtractorOutput` model dump |
| `gap_report.json` | Gap check results (if structure scan ran) |
| `run_log.txt` | model, chunk counts, retries, token usage, section_call_count |
| `trace.jsonl` | `extract_complete` event with full metadata |

`structure_scan_result.json` is written earlier by `structure_scan_pass` (via `structure.py`).

---

### Helper Functions

#### `_merge_section_outputs(partial_outputs, all_chunks) -> dict`

Merges per-section partial dicts into one `RuleExtractorOutput`-shaped dict.

1. `rule_metadata` + `rule_summary` from first partial (`_is_first=True`)
2. Collect all `key_obligations` in section order
3. Renumber OBL-001 through OBL-N (calls `_renumber_obligations`)
4. Build `old_id -> new_id` map during renumber
5. `affected_entity_types`: deduplicate by `entity_type`
6. `compliance_impact_areas`: deduplicate by `area`, merge `linked_obligation_ids`
   and `citations`, apply renumber map to obligation IDs
7. `assumptions`: concatenate without dedup

#### `_renumber_obligations(obligations) -> (list, dict)`

Assigns sequential `OBL-001`, `OBL-002`, ... IDs to all obligations in order.
Returns `(renumbered_list, old_to_new_id_map)`.

#### `_try_structured_output(llm, logger) -> Any | None`

Wraps LLM with `with_structured_output(RuleExtractorOutput, include_raw=True)`.
Returns `None` if the LLM doesn't support it.

#### `_try_section_structured_output(llm, logger) -> Any | None`

Same but wraps with `SectionExtractOutput`.

#### `_src_index(src_id) -> int`

Parses `"src:N"` -> integer N. Returns 999999 on parse failure (sorts to end).

#### `_extract_usage(response) -> dict`

Extracts `{prompt_tokens, completion_tokens, total_tokens}` from LangChain response
`response_metadata`. Returns zeros if metadata absent.

---

## structure.py

### `structure_scan(artifact_dir) -> StructureScanResult`

Deterministic heading scan. No LLM calls.

**Step 1: Find discussion prefix**

Scans `sections.json` for a section whose `heading_path` contains
`"discussion of final amendments"` (case-insensitive). Records the full
`heading_path` up to and including that element as `discussion_prefix`.

This handles two segmenter layouts:
- `["II. Discussion of Final Amendments", "A. Disclosure..."]` -- prefix length 1
- `["II.", "Discussion of Final Amendments", "A. Disclosure..."]` -- prefix length 2

Returns empty `StructureScanResult` if no discussion section found.

**Step 2: Collect lettered obligation sections**

For each section whose `heading_path` starts with `discussion_prefix` and has a
capital-letter prefix (e.g. `"A."`) at `letter_depth`:
- Creates an `ObligationSection` entry on first encounter of each letter
- Collects `final`-role chunks and `has_codified_text=True` chunks into
  `structured_chunk_ids` for that section
- Accumulates CFR citations from section text via `extract_references_from_text()`

**Step 3: Find named sections**

Scans all sections for headings matching named keywords:
`["effective", "compliance date", "applicability", "exemption", "codified text"]`

Collects all chunk IDs from matching sections into `named_section_chunk_ids`.
These are passed to the first-section extractor call for date/scope context.

**Step 4: Build aggregate**

Deduplicates all `structured_chunk_ids` across obligation sections into
`StructureScanResult.structured_chunk_ids`.

**Saves:** `artifacts/{run_id}/structure_scan_result.json`

---

### `gap_check(extraction_output, scan_result, logger) -> dict`

Post-extraction gap detector. No LLM calls.

**Checks:**
1. `count_gap = max(0, expected_obligation_count - len(key_obligations))`
2. For each `ObligationSection` that has CFR citations, checks whether any of those
   citations appear in the extracted obligations' `cited_sections`. Flags sections
   where no citations matched.

**Returns:**
```json
{
  "expected_count": 6,
  "extracted_count": 6,
  "count_gap": 0,
  "flagged_sections": []
}
```

`flagged_sections` entries include `section_letter`, `heading`, `cfr_citations`,
and `reason`. Logged as warnings if non-empty.

---

## prompts.py (extraction functions)

### `build_system_prompt() -> str`

System message for all extractor calls. Sets the LLM's role as a
"SEC Regulatory Intelligence Extractor". Key rules embedded:
- Output valid JSON only (no markdown fences, no prose)
- Never use forbidden terms: compliant, non-compliant, violation, illegal,
  penalty exposure, must fix
- Use `src:<N>` citation format matching provided chunk labels
- `obligation_id` format: `OBL-001`, `OBL-002` (zero-padded, three digits)
- Extract every distinct requirement as a separate obligation

### `build_extractor_prompt(payload, selected_chunks) -> str`

Full-schema extractor prompt. Produces `RuleExtractorOutput`.

**Structure:**
1. Instruction header + citation rule
2. Source chunks formatted as `[src:N] (Section: heading > path)\ntext`
3. Full `_SCHEMA` JSON (rule_metadata, rule_summary, key_obligations, ...)
4. Rules block including ADOPTED RULES ONLY and DISTINCT REQUIREMENTS constraints

**ADOPTED RULES ONLY rule:** Skip anything described as proposed but not adopted,
prefaced with "we are not adopting", "we did not propose", "we are not requiring".

**DISTINCT REQUIREMENTS rule:** One entry per independent top-level requirement.
Safe harbors, exceptions, carve-outs, and timing modifiers go in `key_details` of
the parent obligation -- not as separate entries.

**obligation_text rule:** Must be a complete, self-contained sentence with WHO, WHAT,
WHEN, and UNDER WHAT CONDITION. Deadline and trigger fields must also be populated.

### `build_section_extractor_prompt(section_heading, section_chunks, prior_obligations, summary_text, obligation_id_start, strict_citations) -> str`

Per-section partial-schema extractor prompt. Produces `SectionExtractOutput`.

**Structure:**
1. Document summary (first 500 chars of `summary_text`)
2. Prior obligations block (IDs + first 80 chars of text each, or "none yet")
3. Section header
4. Source chunks (same format as full prompt)
5. `_SECTION_SCHEMA` (key_obligations + entity_types + impact_areas + assumptions, no metadata)
6. Rules including:
   - `"Start obligation numbering from OBL-{start:03d}"`
   - `"Do not duplicate obligations in the ALREADY EXTRACTED block"` -- with exception:
     different entity types (FPI vs domestic) are NOT duplicates
   - `"Focus exclusively on this section's content"`

### `build_gap_analysis_prompt(extracted_output, company_context) -> str`

Plain-English gap analysis prompt. Used by the `gap` CLI command.

Takes the structured extraction output and produces a 4-section report:
1. WHAT THIS RULE REQUIRES
2. OBLIGATIONS SUMMARY
3. WHAT YOU NEED TO BUILD OR HAVE IN PLACE
4. PRIORITY ACTIONS

Each obligation formatted with trigger, deadline, and CFR references for context.

---

## Artifacts Produced

```
artifacts/{run_id}/
    validated_output.json      RuleExtractorOutput -- key_obligations, entities, impact areas
    raw_model_output.txt       Raw LLM JSON string
    gap_report.json            Gap check: expected vs extracted count, flagged sections
    structure_scan_result.json StructureScanResult -- obligation sections, chunk IDs, CFR cites
    run_log.txt                model, counts, retries, token usage, section_call_count
    trace.jsonl                extract_complete event (appended)
```

---

## Performance

Structure scan: < 50 ms (pure Python, linear in section count).

LLM calls (section-per-call path):
- 1 call per obligation section (typically 4-8 calls per document)
- All calls use the full context of that section's final-role chunks
- No token waste on proposed/commentary chunks

Single-call fallback: 1 extractor call over all selected chunks (~28 chunks for
the SEC cybersecurity rule, ~7000-9000 tokens prompt).
