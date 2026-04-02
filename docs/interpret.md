# Interpretation Pipeline

## Overview

Stage 4 of the SEC compliance extraction pipeline. Produces structured legal
interpretations for each extracted obligation.

Three LLM models in play:
- `llm` (main model, e.g. gpt-4o): final interpretation call
- `cheap_llm` (e.g. gpt-4o-mini): CFR reference judge, bin pass reviewer

Five sub-steps per obligation, all inside `run_interpret_pipeline()`:
1. **build_initial_context** -- no LLM: definition lookup + section-family final chunks
2. **resolve_references** -- agentic loop: fetch live CFR text, judge decides if more needed
3. **interpret_obligation** -- one main LLM call -> `ObligationInterpretation`
4. **term lookup loop** -- LLM signals ambiguous terms; system fetches, re-interprets once
5. **expansion pass** -- if LLM still needs more, add proposed/other chunks and re-interpret

Bin pass (bin_graph.py) runs before interpret as a secondary scan that pre-links
context snippets to obligations -- the interpreter loads these as "discussion" passages.

---

## Files

| File | Responsibility |
|------|----------------|
| `sec_interpreter/interpret_graph.py` | Main pipeline, per-obligation context assembly, multi-pass loop |
| `sec_interpreter/tools.py` | Non-LLM tools: definition lookup, CFR fetch, chunk search |
| `sec_interpreter/bin_graph.py` | Bin pass: cheap LLM reviewer for flagged out-of-set chunks |

---

## Data Flow

```
artifacts/{run_id}/validated_output.json  (extracted obligations)
    |
[_build_initial_context]   -> context_bundle { definitions, anchor_context, discussion, cfr_texts }
    |
[_resolve_references]      -> context_bundle.cfr_texts populated (agentic, max_depth=2)
    |
[_interpret_obligation]    -> ObligationInterpretation  (pass 1)
    |
[term lookup loop]         -> search_chunks_for_term -> re-interpret if found (pass 2)
    |
[expansion pass]           -> get_section_family_chunks(proposed+other) -> re-interpret if needed (pass 3)
    |
artifacts/{run_id}/interpretation.json
```

Events logged to `trace.jsonl` at each step.

---

## interpret_graph.py

### `run_interpret_pipeline(run_id, llm, cheap_llm, logger) -> InterpretationOutput`

Main entry point. Loads `validated_output.json`, optionally loads `bin_findings.json`,
then iterates each obligation through the full pipeline.

**Bin findings integration:**
- Loads `BinPassOutput` from `bin_findings.json` if present
- For each obligation, finds findings where `obl_id in f.related_to` and
  `finding_type != "not_relevant"`
- Places them in `context_bundle["discussion"]` as `"[{finding_type}] {text}"` strings
- The interpreter prompt includes discussion passages as context for interpretation

Saves `InterpretationOutput` to `artifacts/{run_id}/interpretation.json`.

---

### Step 1: `_build_initial_context(obl, artifact_dir, logger)`

Assembles a context bundle with no LLM calls.

**Section-family final chunks (primary path):**
1. Resolves the obligation's `source_citations[0]` to a `section_family` via `chunks.json`
2. Calls `_load_section_final_chunks(section_family, artifact_dir)` which loads all chunks
   where `section_family` matches AND `subsection_role == "final"`
3. This gives the interpreter the complete adopted rule text for the obligation's section,
   not just the 1-2 chunks the extractor happened to cite

**Fallback (no section_id):**
- Falls back to `_load_source_chunks()` which loads only the extractor-cited chunks

**Definition lookup:**
- Calls `detect_ambiguous_terms(obligation_text)` on the raw obligation text
- For each found term, calls `lookup_definition(term, artifact_dir)` to search
  classified definition sections
- Definitions added to `context_bundle["definitions"]`

**Context bundle structure:**
```python
{
    "definitions": List[str],         # "Definition of 'material':\n..."
    "anchor_context": List[str],       # "[Heading > Path]\ntext..."
    "discussion": [],                  # filled by bin findings in caller
    "cfr_texts": {},                   # filled by resolve_references
    "fetched_refs": set(),             # dedup tracking
    "lookup_results": {},              # term -> List[str], filled by term lookup loop
}
```

---

### Step 2: `_resolve_references(obl, context_bundle, cheap_llm, logger)`

Agentic loop to fetch live CFR text. Max `MAX_REFERENCE_DEPTH = 2` iterations.

**Initial fetch:**
- Takes all `cited_sections` from the obligation
- Calls `fetch_cfr(ref)` for each not already fetched
- Stores text in `context_bundle["cfr_texts"]`

**Agentic loop (up to 2 rounds):**
1. Finds all CFR references in the already-fetched text via `extract_references_from_text()`
2. Filters to refs not yet fetched
3. Calls cheap LLM with `build_reference_judge_prompt(obligation_text, context_summary, refs_seen)`
4. If judge says `"SUFFICIENT"` -> stop
5. Otherwise judge returns one reference to fetch next
6. Validates returned ref is in the candidate list (prevents hallucination)
7. Fetches and loops

**Result:** `context_bundle["cfr_texts"]` contains fetched CFR section texts, each
truncated to 3000 chars.

---

### Step 3: `_interpret_obligation(obl, context_bundle, llm, logger, sibling_obligations)`

Calls `build_interpretation_prompt()` then invokes the main LLM.

Returns `ObligationInterpretation`. On any failure, returns a fallback with
`confidence_level="low"` and the raw obligation text as `primary_interpretation`.

The interpreter is given sibling obligations (all obligations from the same run)
as context, so it can express `parent_obligation_ids` links.

---

### Multi-pass logic

**Term lookup loop (pass 2):**
- Merges `interpretation.lookup_requests` and `interpretation.ambiguous_terms` into a
  single dedup'd lookup queue
- Strips explanation suffixes from ambiguous_terms (LLM may use `"material: unclear standard"`)
- Calls `search_chunks_for_term(term, artifact_dir, top_n=5)` for each term
- If any matches found, adds to `context_bundle["lookup_results"]` and re-interprets

**Expansion pass (pass 3):**
- Only runs if `interpretation.needs_more_context == True` and `section_id` is available
- Calls `get_section_family_chunks(section_id, artifact_dir, subsection_roles=["proposed", "other"])`
- Adds returned chunks to `context_bundle["anchor_context"]`
- Re-interprets with the expanded context

**Total passes:** typically 1; up to 3 if term lookup + expansion both trigger.

---

### `_trace(artifact_dir, event)`

Appends a JSON line to `artifacts/{run_id}/trace.jsonl`.

Events written:
- `interpret_obligation_start`: obligation_id, anchor_chunks, bin_findings_count, section_id
- `interpret_pass`: pass number, confidence, lookup_requests, needs_more_context
- `term_lookup`: term, chunks_found
- `expand_context`: roles, chunks_added
- `interpret_obligation_complete`: total_passes, final_confidence

---

## tools.py

Non-LLM utility functions. No LLM calls in this file.

---

### `detect_ambiguous_terms(obligation_text) -> List[str]`

Scans obligation text against a fixed list of known legally ambiguous terms:
```
material, materiality, promptly, reasonable, reasonably, significant,
substantial, timely, appropriate, adequate, effective, necessary,
principal, affiliated, control
```
Uses word-boundary regex. Returns matched terms (may be empty).

---

### `lookup_definition(term, artifact_dir) -> Optional[str]`

Searches document sections classified as `"definition"` content_type for a term.

1. Loads `section_classifications.json` -> finds section_ids where `content_type == "definition"`
2. Loads `sections.json` -> searches only the classified definition sections
3. Returns `section_text[:2000]` for the first matching section, or `None`

Requires `section_classifications.json` to be present (produced by classify pipeline).
Returns `None` gracefully if the file is missing.

---

### `get_section_family_chunks(section_id, artifact_dir, subsection_roles) -> List[dict]`

Structural lookup: returns all chunks in the same `section_family` as the given
`section_id`, filtered to the requested `subsection_roles`.

**Default roles:** `["comments", "final"]` -- covers SEC reasoning (final) and
industry Q&A (comments), the primary interpretation context.

**Lookup logic:**
1. Load `chunks.json`
2. Find the `section_family` for the given `section_id`
3. Return all chunks where `section_family` matches AND `subsection_role` is in the role set

Returns list of dicts with keys: `src_id`, `subsection_role`, `heading`, `text`.

This function is deterministic (no LLM, no embeddings). Works on `heading_path`-derived
fields set at ingest time -- no classify stage required.

---

### `search_chunks_for_term(term, artifact_dir, top_n=5) -> List[dict]`

Keyword search over `chunks.json` for a term.

**Scoring per chunk:**
- `hit_count` = word-boundary occurrences of term in chunk text
- Bonus +2 if `has_definitions == True`
- Bonus +1 if `has_example == True`

Returns top `top_n` chunks sorted by score as dicts with keys: `src_id`, `heading`, `text`.
Returns `[]` if no matches or `chunks.json` missing.

---

### `search_document(query, artifact_dir, content_types, top_n, prefer_examples) -> List[str]`

Keyword search across classified sections.

**Default content_types:** `["commentary", "comments"]`

**Algorithm:**
1. Load `section_classifications.json`, filter to target `content_types`
2. Score each section against query keywords (stopwords removed):
   - Count keyword hits in section summary + heading
   - Bonus +2 if any chunk in section has `has_example == True`
3. Sort by score, take top `top_n`
4. Load full `section_text` from `sections.json`
5. Return `"[heading]\ntext[:2000]"` strings

Gracefully returns `[]` if classification artifacts are missing.

---

### `fetch_cfr(citation, date="current") -> Optional[str]`

Fetches live CFR section text from the eCFR public API.

**Supported citation formats:**
- `"17 CFR 229.106(b)"` -> part=229, section=229.106
- `"CFR 229.106"` -> same
- `"Rule 13a-11"` -> no CFR pattern, returns None

**API endpoint:** `https://www.ecfr.gov/api/renderer/v1/content/enhanced/{date}/title-17?part={part}&section={section}`

Returns HTML, stripped to plain text (block tags converted to newlines, HTML entities
decoded, excess whitespace collapsed). Truncated to 3000 chars. Returns None on
HTTP error, network failure, or unparseable citation.

---

### `extract_references_from_text(text) -> List[str]`

Finds CFR citation strings in a block of text. Deduplicates by insertion order.
Only returns citations verbatim present in the text (prevents hallucination downstream).

Pattern matches: `CFR NNN.NNN` and variants with `17 C.F.R.`, `Part`, and subsection
suffixes like `(a)`.

---

## Bin Pass: bin_graph.py

### `run_bin_pass(run_id, extraction_output, cheap_llm, logger) -> BinPassOutput`

Secondary reviewer pass. Runs after primary extraction.

**Purpose:** Find missed obligations, scope modifiers, definitions, edge cases, and
implied requirements in the chunks that the main extractor did not see.

**Steps:**
1. Load `chunks.json`
2. Load `structure_scan_result.json` -> get `structured_chunk_ids` (the chunks the extractor saw)
3. Compute `remaining` = all chunks NOT in `structured_chunk_ids`
4. Compute `flagged` = remaining chunks where `has_obligations OR has_codified_text OR has_scope`
5. If no flagged chunks -> save empty `BinPassOutput`, return
6. Build prompt via `build_bin_pass_prompt(flagged, known_obligations)`
7. Call `cheap_llm`, parse JSON -> list of `BinFinding`
8. Log `missed_obligation` findings as warnings
9. Save `bin_findings.json`

**Finding types** (validated by schema):
- `missed_obligation`: a requirement the main extractor missed
- `scope_modifier`: text that changes who an obligation applies to
- `implied_requirement`: obligation implied by the rule but not stated explicitly
- `definition`: a definition relevant to understanding obligations
- `edge_case`: a carved-out scenario or exception
- `not_relevant`: chunk is not useful (bin pass may classify chunks as irrelevant)

**Integration with interpret:**
The interpret pipeline loads `bin_findings.json` and routes each finding to the
obligation matching `finding.related_to`. Findings typed `not_relevant` are skipped.

---

## Artifacts Produced

```
artifacts/{run_id}/
    interpretation.json    InterpretationOutput -- one ObligationInterpretation per obligation
    bin_findings.json      BinPassOutput -- missed obligations, scope modifiers, edge cases
    trace.jsonl            Structured event log (interpret events appended)
```

---

## Output Schema: `ObligationInterpretation`

```
obligation_id             str           -- matches extracted OBL-NNN
primary_interpretation    str           -- plain-English statement of what the rule requires
key_details               List[str]     -- 3-5 specific implementation details
supporting_sections       List[str]     -- CFR section numbers referenced
alternative_interpretations List[str]   -- edge-case or disputed readings
ambiguous_terms           List[str]     -- terms needing definition lookup
compliance_implication    str           -- what the company needs to build/do
confidence_level          Literal[high, medium, low]
needs_more_context        bool          -- signals interpreter wants expansion pass
lookup_requests           List[str]     -- terms to search in document
parent_obligation_ids     List[str]     -- OBL-NNN links for related obligations
```

---

## Performance

Typical LLM call count per run (10 obligations):
- 10 `interpret_obligation` calls (main LLM)
- 0-2 CFR judge calls per obligation (cheap LLM)
- 0-1 term-lookup re-interpret per obligation (main LLM, if terms found)
- 0-1 expansion re-interpret per obligation (main LLM, if needs_more_context)
- 1 bin pass call total (cheap LLM)

Total: ~10-30 main LLM calls + ~0-20 cheap LLM calls per run.
