# Schemas, Evaluation, CLI, and Utilities

## Overview

Supporting modules used across all pipeline stages.

| Module | Role |
|--------|------|
| `schemas.py` | All Pydantic models used as stage inputs/outputs |
| `eval.py` | LLM-as-judge evaluation of interpretation coverage |
| `cli.py` | Command-line interface (10+ subcommands) |
| `utils.py` | JSON parsing, validation enforcement, chunking, logging |
| `module.py` | Module orchestrators that wire stages together |

---

## schemas.py

All Pydantic models. Imported by every pipeline stage.

### Ingest schemas

```
IngestInput     { source: str, page_range: Optional[Tuple[int,int]], strict_citations: bool }
IngestResult    { run_id: str, chunk_count: int, artifact_dir: str }
```

### Section / chunk schemas (produced by ingest, consumed by extract + interpret)

```
Section         { section_id, heading_path: List[str], level: int, section_text }
RichChunk       { src_id, section_id, heading_path, chunk_index_in_section,
                  text, char_len, token_estimate,
                  has_dates, has_scope, has_obligations, has_definitions,
                  has_codified_text, has_example,
                  content_type, section_family, subsection_role }
```

`section_family` = `heading_path[1]` (the lettered obligation section, e.g.
`"A. Disclosure of Cybersecurity Incidents"`). Set at ingest time.

`subsection_role` = `"final"` | `"proposed"` | `"comments"` | `"other"`. Derived
from heading_path by scanning deepest-first for the keywords. Set at ingest time.

### Extraction schemas

```
RuleExtractorInput   { rule_text: str, strict_citations: bool }

RuleMetadata         { rule_title, release_number, publication_date,
                       effective_date, citations: List[str] }

RuleSummary          { summary: str, citations: List[str] }

KeyObligation        { obligation_id, rule_provision, obligation_text,
                       trigger, deadline,
                       cited_sections: List[str], source_citations: List[str] }

AffectedEntityType   { entity_type: str, citation: str }

ComplianceImpactArea { area: str, linked_obligation_ids: List[str], citations: List[str] }

Assumption           { assumption_text: str, reason: str, citation: Optional[str] }

RuleExtractorOutput  { rule_metadata, rule_summary, key_obligations,
                       affected_entity_types, compliance_impact_areas, assumptions }

SectionExtractOutput { key_obligations, affected_entity_types,
                       compliance_impact_areas, assumptions }
```

`SectionExtractOutput` is used by `extract_sections_loop` for non-first sections.
It omits `rule_metadata` and `rule_summary` (document-level fields only needed once).

**Valid `ComplianceImpactArea.area` values:**
`Recordkeeping`, `Reporting`, `Disclosure`, `Internal Controls`,
`Governance`, `Risk Management`, `Technology Controls`

**Citation format:** `src:<N>` where N is the 0-based chunk index.
Validated by `field_validator` on `source_citations`, `citations`, and `citation` fields.

**`obligation_id` format:** `OBL-001`, `OBL-002`, ... (zero-padded, three digits)

### Locator / structure scan schemas

```
LocatorSelection    { date_chunks, scope_chunks, obligation_chunks,
                      definition_chunks, other_key_chunks -- all List[str] }

ObligationSection   { section_letter, heading, section_id,
                      cfr_citations: List[str], structured_chunk_ids: List[str] }

StructureScanResult { run_id, obligation_sections, named_section_chunk_ids,
                      expected_obligation_count, structured_chunk_ids }
```

### Classification schemas

```
SectionClassification  { section_id, heading_path, content_type, summary,
                          topics: List[str], useful_for: List[str] }

DocumentMap            { regulatory_objective, rule_title, sections_by_type,
                          compliance_section_ids, cost_section_ids, definition_section_ids }
```

**Valid `content_type` values:** `final_rule_text`, `obligation`, `definition`,
`commentary`, `comments`, `economic_analysis`, `procedural`

`COMPLIANCE_CONTENT_TYPES = {"final_rule_text", "obligation", "definition"}`

### Interpretation schemas

```
ObligationInterpretation  { obligation_id, primary_interpretation,
                             key_details: List[str], supporting_sections: List[str],
                             alternative_interpretations: List[str],
                             ambiguous_terms: List[str],
                             compliance_implication: str,
                             confidence_level: Literal["high","medium","low"],
                             needs_more_context: bool,
                             lookup_requests: List[str],
                             parent_obligation_ids: List[str] }

ObligationContextLinks    { key_indices, supporting_indices, skip_indices -- all List[int] }

InterpretationOutput      { run_id, rule_title,
                             interpretations: List[ObligationInterpretation] }
```

### Bin pass schemas

```
BinFinding     { finding_type, text, related_to: List[str],
                  source_chunks: List[str], notes: Optional[str] }

BinPassOutput  { run_id, findings: List[BinFinding] }
```

**Valid `finding_type` values:** `missed_obligation`, `scope_modifier`,
`implied_requirement`, `definition`, `edge_case`, `not_relevant`

`finding_type` is validated by `field_validator`.

### Citation validation constant

```python
CITATION_PATTERN = re.compile(r"^src:\d+$")
```

Used by all field validators. Raises `ValueError` with `"src:<index> format"` message.

---

## eval.py

LLM-as-judge evaluation of `interpretation.json` against reference criteria.

### `run_eval(run_id, criteria_path, llm) -> dict`

**Input files:**
- `artifacts/{run_id}/interpretation.json` -- interpretations to evaluate
- `criteria_path` -- JSON file with structure:
  ```json
  {
    "sources": ["url1", "url2"],
    "obligations": {
      "OBL-001": {
        "description": "...",
        "criteria": ["criterion 1", "criterion 2", ...]
      }
    }
  }
  ```

**ID-agnostic evaluation:**
The judge sees ALL interpretations concatenated into one text block -- not filtered
by obligation ID. This means criteria are checked against the collective set of all
interpretations. An obligation the pipeline split into OBL-002 and OBL-003 will still
pass criteria for the original OBL-002 as long as either interpretation covers it.

**Per criterion:** calls `_judge(all_interp_text, criterion, llm)` which returns
`("PASS"|"FAIL", explanation)`. Defaults to FAIL on LLM error.

**Judge prompt:** Instructs the LLM to respond with JSON `{"result": "PASS"|"FAIL", "explanation": "..."}`.
PASS = "at least one interpretation contains this concept, either explicitly or by clear implication".

**Saves:** `artifacts/{run_id}/eval_report.json`

**Returns dict:**
```json
{
  "run_id": "...",
  "rule_title": "...",
  "sources": [...],
  "summary": { "total_criteria": 38, "passed": 36, "failed": 2, "coverage_pct": 94.7 },
  "obligations": {
    "OBL-001": {
      "description": "...",
      "criteria_results": [{"criterion": "...", "result": "PASS", "explanation": "..."}],
      "pass_count": 14,
      "total_count": 14
    }
  }
}
```

### `print_report(report)`

Prints coverage table to stdout. FAILed criteria include the judge's explanation.
Format:
```
========================================================================
EVAL REPORT  run_id=abc123
Rule: SEC Cybersecurity Disclosure Rule
Coverage: 36/38 criteria passed (94.7%)
========================================================================

OBL-001  Material cybersecurity incident disclosure ...  (14/14)
------------------------------------------------------------------------
  [PASS] The 4-day clock starts at the materiality determination...
  [FAIL] ERM integration requirement...
         -> Interpretation does not explicitly address ERM integration...
```

---

## cli.py

Entry point: `python -m sec_interpreter.cli <command> [args]`

### Commands

| Command | Description | Key args |
|---------|-------------|----------|
| `ingest` | Fetch and chunk a document | `--url` or `--input`, `--pages`, `--strict` |
| `extract` | Run extraction on ingested artifacts | `--run-id`, `--output`, `--strict` |
| `run` | Full pipeline: ingest + extract + bin + interpret + brief | `--url`/`--input`, `--pages`, `--output` |
| `classify` | Classify every section by content type | `--run-id` |
| `interpret` | Interpret extracted obligations | `--run-id`, `--output` |
| `gap` | Generate plain-English gap analysis | `--run-id`, `--output`, `--company` |
| `report` | Format markdown compliance report | `--run-id`, `--output` |
| `scan` | Run structure scan, print sections (no LLM) | `--run-id` |
| `bin` | Run bin pass (secondary scan) | `--run-id` |
| `brief` | Generate case brief | `--run-id`, `--output` |
| `eval` | LLM-as-judge eval against criteria | `--run-id`, `--criteria` |
| `comprehend` | Calibration: classify every chunk | `--run-id` |

### Source input (`--url` vs `--input`)

- `--url`: URL of SEC PDF or HTML document
- `--input path/to/file.pdf`: local PDF
- `--input path/to/file.txt`: local plain text
- `--input path/to/file.json`: JSON with `"rule_text"` key; extracted to temp `.tmp.txt`

`--pages START-END`: 1-based page range for PDFs (e.g. `--pages 5-30`)

### `_cmd_run()` -- full pipeline

Stages run sequentially:
1. `IngestModule().run(source, page_range)` -- fetch, chunk, save artifacts
2. `ExtractModule().run(run_id)` -- structure scan + section-per-call LLM extraction
3. `run_bin_pass(run_id, extraction_dict, cheap_llm)` -- secondary scan
4. `InterpretModule().run(run_id)` -- per-obligation interpretation
5. `build_case_brief_prompt(...)` + LLM call -- final case brief

Outputs: `--output path` gets `validated_output.json`; case brief written to `path.brief.md`.

### `_load_named_section_texts(artifact_dir) -> list`

Loads `structure_scan_result.json` -> `named_section_chunk_ids`, then fetches
corresponding chunk texts from `chunks.json`. Returns formatted `"[heading]\ntext"` strings.
Used to inject effective dates / scope / exemptions into the case brief prompt.

---

## utils.py

### `get_logger(name) -> logging.Logger`

Returns a logger with a `StreamHandler` at `INFO` level. Idempotent -- does not
add duplicate handlers if called multiple times with the same name.

---

### `parse_json_object(raw_text) -> dict`

Parses JSON from an LLM response string. Two strategies:
1. `json.loads(text.strip())` -- direct parse
2. Find `{...}` substring and parse that -- handles leading/trailing prose

Raises `ValueError` if result is not a dict.

---

### `repair_json(raw_text) -> str`

Cleans malformed LLM JSON. Transformations applied:
1. Strip markdown code fences (` ```json ` and ` ``` `)
2. Remove trailing commas before `}` or `]`
3. Extract `{...}` substring

Used as a second-chance parse attempt in `validate_output`.

---

### `chunk_rule_text(text) -> List[str]`

Splits rule text into paragraph-based chunks of at most ~1500 chars each.
Used only in direct/inline mode (tests and `RuleExtractorModule`).

Algorithm:
1. Split on blank lines (`\n\s*\n`) to get paragraphs
2. Accumulate paragraphs until adding the next would exceed 1500 chars -- flush
3. A single paragraph > 1500 chars gets its own chunk

---

### Validation functions

All operate on `RuleExtractorOutput` instances. No-ops on other types.

#### `enforce_citation_bounds(output, chunk_count)`

Two checks:
1. Scans all string fields via `_iter_text_fields()` for `src:N` patterns; validates N < chunk_count
2. Explicitly checks every `source_citations`, `citations`, and `citation` field

Raises `ValueError` with offending citation if out of range.

#### `enforce_strict_citations(output, payload)`

Only active when `payload.strict_citations=True`.

Checks:
- Every `KeyObligation.source_citations` is non-empty
- Every `AffectedEntityType.citation` is set

#### `enforce_obligation_links(output)`

Checks that every `ComplianceImpactArea.linked_obligation_ids` value matches a
known `obligation_id` in `output.key_obligations`.

#### `enforce_safe_language(output)`

Scans all string fields for banned terms:
`compliant`, `non-compliant`, `violation`, `illegal`, `penalty exposure`, `must fix`

These terms create legal risk in automated compliance output. The prompt instructs
the LLM to use `risk`, `gap`, `needs review`, `likely applicable`, `may require`,
`should consider` instead.

---

### `_iter_text_fields(output) -> Iterator[str]`

Yields all user-visible string values from a `RuleExtractorOutput`:
`rule_title`, `release_number`, `publication_date`, `effective_date`,
`summary`, all `obligation_id`, `obligation_text`, `cited_sections`,
`entity_type`, `area`, `linked_obligation_ids`, `assumption_text`, `reason`.

Used by both `enforce_citation_bounds` and `enforce_safe_language`.

---

## module.py

### LLM loaders

#### `_load_env_llm() -> Any | None`

Loads the main LLM from environment. Reads `SEC_INTERPRETER_MODEL` (e.g. `"gpt-4o"`).
Loads API keys from `.env` via `dotenv`. Returns `None` if no model configured.

#### `_load_cheap_llm() -> Any | None`

Loads cheap LLM from `SEC_INTERPRETER_CHEAP_MODEL` (e.g. `"gpt-4o-mini"`).
Falls back gracefully to `None`.

### `DeterministicLLM`

Offline LLM stub used in tests and as a last-resort fallback.

- `invoke(messages)` -> scans the last `HumanMessage` for `[src:N]` patterns
- Counts unique `src:N` references, builds minimal valid `RuleExtractorOutput` JSON
- Always produces a single placeholder obligation citing all found source IDs
- Used in tests to verify pipeline structure without API calls

### `FakeLLM`

Simple test double. `invoke(messages)` returns a fixed string set at construction.
Used for unit tests that need predictable LLM output without the deterministic
heuristics of `DeterministicLLM`.

### Module classes

#### `IngestModule`

```python
IngestModule().run(source, page_range=None, strict_citations=False) -> IngestResult
```

- Calls `run_ingest_graph(source, page_range, strict_citations)`
- Returns `IngestResult { run_id, chunk_count, artifact_dir }`

#### `ExtractModule`

```python
ExtractModule(llm=None).run(run_id, strict_citations=False) -> RuleExtractorOutput
```

- Loads LLM via `_load_env_llm()` or falls back to `DeterministicLLM`
- Calls `build_extract_graph(llm, logger).invoke(initial_state)`
- Returns `RuleExtractorOutput`

`module.llm` is accessible after construction -- used by `_cmd_run` to reuse the
same LLM instance for the case brief call.

#### `ClassifyModule`

```python
ClassifyModule().run(run_id) -> dict
```

- Classifies every section by content type (calls classify_graph)
- Caches results in `section_classifications.json`
- Returns dict with `run_id`, `section_count`, `compliance_section_count`,
  `rule_title`, `regulatory_objective`, `type_counts`

#### `InterpretModule`

```python
InterpretModule(llm=None, cheap_llm=None).run(run_id) -> InterpretationOutput
```

- Loads LLMs via env or falls back to `DeterministicLLM`
- Calls `run_interpret_pipeline(run_id, llm, cheap_llm, logger)`
- Returns `InterpretationOutput`

#### `RuleExtractorModule`

Legacy combined module for direct text input (no two-stage pipeline).

```python
RuleExtractorModule(llm=None).run(rule_text, strict_citations=False) -> RuleExtractorOutput
```

- Creates a `RuleExtractorInput` payload
- Runs `build_extract_graph(llm, logger)` with `skip_locator=True`
- Returns `RuleExtractorOutput`
- Used by tests to run extraction directly on text without ingest artifacts

---

## Test criteria file format (`tests/eval_criteria.json`)

```json
{
  "sources": ["url1", "url2"],
  "obligations": {
    "OBL-001": {
      "description": "short description",
      "criteria": [
        "specific verifiable statement about what the rule requires",
        "another specific criterion"
      ]
    }
  }
}
```

**Criteria writing guidelines:**
- Each criterion should be a single specific, verifiable fact about the rule
- Test what the rule IS (adopted requirements), not what it considered but dropped
- For requirements that were proposed but not adopted, write a criterion testing that
  the pipeline correctly identifies the absence (e.g. "The rule does NOT require X")
- Obligation IDs in the criteria file are for organization only -- the judge evaluates
  all interpretations collectively regardless of which OBL-NNN they were extracted under
