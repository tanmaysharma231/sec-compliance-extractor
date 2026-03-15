# Changelog

## [Unreleased] -- 2026-03-15

### Added: Commentary search + has_example flag

**Why:** Interpretation prompts lacked access to the SEC's own discussion of how rules
should apply. Adding commentary search and example detection gives the interpreter
richer context without expanding the extraction context window.

- `scorer.py`: EXAMPLE_KEYWORDS + `has_example` flag on RichChunk
- `tools.py`: `search_document()` -- keyword search over commentary/comments sections
- `interpret_graph.py`: `search_document()` called in `_build_initial_context`
- `prompts.py`: DISCUSSION CONTEXT block in interpreter prompt + richer
  compliance_implication guidance

---

## [Unreleased] -- 2026-03-10

### Added: Interpretation pipeline

**Why:** Extraction gives you the obligation text. Interpretation gives you what it
means for a regulated firm -- the CFR section it modifies, how ambiguous terms have
been read, and what operational changes it requires.

- `interpret_graph.py`: per-obligation pipeline
  - definition lookup + surrounding chunk context
  - live CFR fetch via eCFR API (`tools.py: fetch_cfr_section()`)
  - agentic reference resolution loop (max 2 hops, cheap LLM judge)
  - full LLM call -> ObligationInterpretation per obligation
- `tools.py`: `fetch_cfr_section()`, `lookup_definitions()`, `search_document()`
- `schemas.py`: ObligationInterpretation, InterpretationOutput
- `report_formatter.py`: markdown report with interpretation blocks per obligation
- `cli.py`: `interpret` subcommand + `report` subcommand + `gap` subcommand

---

## [Unreleased] -- 2026-02-28

### Added: Classify pipeline + test suite expansion

**Why:** Real SEC releases are ~60% noise for compliance purposes (economic analysis,
public comments, SEC reasoning). Running extraction over the full document wastes
tokens and introduces irrelevant text. Content-type classification solves both.

**Calibration finding (run_id 065ae71a69b6, SEC Cybersecurity Disclosure Rule):**
- 161 total chunks; 65 are final_rule_text + obligation + definition (40%)
- economic_analysis: 42 chunks (26%), comments: 26 (16%), commentary: 23 (14%)
- Filtering to operative chunk types cuts locator input by 60%

**ClassifyGraph:**
- One LLM call per section group -> SectionClassification (content_type, summary, topics)
- Synthesis call -> DocumentMap (regulatory_objective, sections_by_type, important_chunks)
- 7 content types: `final_rule_text`, `obligation`, `definition`, `commentary`,
  `comments`, `economic_analysis`, `procedural`
- Deterministic extract filter post-classify (bypasses LLM locator when classify has run)

**Comprehend calibration tool (`comprehend.py`):**
- One LLM call per chunk: content_type + summary + important flag
- Synthesis call: document understanding + important_chunks list
- Saves `artifacts/<run_id>/control.json`
- Prints locator vs control comparison for calibration

**Test suite:**
- 7 tests -> 55 tests (all passing)
- Added: `test_new_schemas.py`, `test_segmenter.py`, `test_tools.py`

**New CLI subcommands:** `classify`, `comprehend`

---

## [Unreleased] -- 2026-02-27

### Added: Intelligent Two-Pass Extraction Pipeline

**Why:** The single-pass pipeline sent all chunks to the Extractor LLM.
For large documents (e.g. 281-page 2023 Cybersecurity Rule -> 200+ chunks),
this wastes tokens and degrades extraction quality by flooding the context.

**Old IngestGraph flow:**
```
fetch_document -> chunk_text -> save_ingest_artifacts
```

**New IngestGraph flow:**
```
fetch_document -> segment_document -> chunk_sections -> score_chunks
               -> extract_summary  -> save_ingest_artifacts
```

**Old ExtractGraph flow:**
```
load_chunks -> extract_structured_fields -> validate_output
            ^ (retry) ^                 -> save_extract_artifacts
```

**New ExtractGraph flow:**
```
load_chunks -> [locator_pass] -> extract_structured_fields -> validate_output
                               ^ (retry) ^                 -> save_extract_artifacts
```
*(locator_pass skipped in direct/inline-text mode for backward compat)*

### New files
- `sec_interpreter/segmenter.py` -- heading-aware section tree builder
- `sec_interpreter/scorer.py` -- hot-zone flag scoring (regex-based, no LLM)
- `CHANGELOG.md` -- this file
- `SYSTEM_OVERVIEW.md` -- architecture overview

### Modified files
- `sec_interpreter/schemas.py` -- added `Section`, `RichChunk`, `LocatorSelection` models
- `sec_interpreter/ingest_graph.py` -- full rewrite with 5-node pipeline
- `sec_interpreter/extract_graph.py` -- added `locator_pass` node, updated state/nodes
- `sec_interpreter/prompts.py` -- added `build_locator_prompt`, `build_extractor_prompt`

### New artifacts per run
```
artifacts/<run_id>/
  sections.json          <- List[Section] with heading_path
  summary.txt            <- auto-extracted SUMMARY section text
  shortlist.json         <- compact index rows for Locator
  locator_selection.json <- Locator LLM output (selected src_ids by category)
  trace.jsonl            <- structured events with timestamps
  chunks.json            <- updated: List[RichChunk] (was List[{id, text}])
  run_log.txt            <- updated: + selected_chunk_count field
```

### Backward compatibility
- `RuleExtractorModule.run(payload)` -- unchanged; locator skipped in direct mode
- All 7 existing tests pass (RuleExtractorModule bypasses locator_pass)
- Old `chunks.json` format (plain `{id, text}` entries) detected and wrapped on load
