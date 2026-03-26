# Implementation Plan

Rebuilding the extraction pipeline from locator-based to structure-first.
Work in order -- each step depends on the previous.

---

## Pipeline at a Glance

```
RAW DOCUMENT
     |
     v
INGEST  (no LLM)
  segment + chunk + score flags + set section_family + subsection_role
     |
     v
STRUCTURE SCAN  (no LLM)
  find lettered obligation sections A, B, C... from headings
  collect final + codified chunks per section     <- extraction targets
  collect named sections by heading               <- effective dates, scope, exemptions
  extract CFR citations per section               <- for gap check
  count sections                                  <- expected obligation upper bound
     |
     +-- structured_chunks (~25, 15-20% of doc)
     +-- named_section_chunks (dates, scope, exemptions)
     +-- section_map (section -> CFR citations)
     +-- expected_obligation_count
     |
     v
LLM PASS 1 -- EXTRACT  (1 main LLM call)
  input: structured_chunks only
  output: OBL-001..N with rule_provision + CFR citations
     |
     v
GAP CHECK  (no LLM)
  obligation count vs expected_obligation_count
  obligation CFR cites vs section_map
  -> gap_report.json
     |
     v
FLAG FILTER  (no LLM)
  remaining chunks (not in structured_chunks)
  keep: has_obligations OR has_codified_text OR has_scope
  skip: everything else (~86 chunks -- economic analysis, procedural, footnotes)
     |
     +-- flagged_remainder (~40-50 chunks)
     |
     v
LLM PASS 2 -- BIN  (1 cheap LLM call)
  input: flagged_remainder + known obligations as context
  classify each chunk:
    missed_obligation  -> safety net, updates validated_output
    scope_modifier     -> who/what is covered
    implied_requirement
    definition
    edge_case          -> SEC comment responses, specific scenarios
    not_relevant
  tag each finding: related_to [OBL-001, OBL-002...]
  -> bin_findings.json
     |
     v
INTERPRET  (1 main LLM call per obligation)
  input: obligation + bin_findings[OBL-ID] (pre-linked, no search)
       + live CFR text from eCFR API
  -> interpretation.json
     |
     v
CASE BRIEF  (1 main LLM call)
  input: validated_output + bin_findings + interpretation + named_section_chunks
  output: scope, obligations, definitions, modifiers, dates, edge cases,
          what it means in practice
  -> case_brief.md
```

---

---

## Step 1 -- Schema changes  (schemas.py)

### 1a. Add rule_provision to KeyObligation
Add a dedicated field for the specific CFR provision or form item that
imposes the obligation.

```python
class KeyObligation(BaseModel):
    obligation_id: str
    rule_provision: Optional[str] = None   # "17 CFR 229.106(b)" / "Item 1.05 Form 8-K"
    obligation_text: str
    trigger: Optional[str] = None
    deadline: Optional[str] = None
    disclosure_fields: List[str]
    evidence: List[str]
    cited_sections: List[str]
    source_citations: List[str]
```

### 1b. Add BinFinding schema

```python
VALID_BIN_TYPES = {
    "missed_obligation",
    "scope_modifier",
    "implied_requirement",
    "definition",
    "edge_case",
    "not_relevant",
}

class BinFinding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    finding_type: str              # one of VALID_BIN_TYPES
    text: str                      # relevant excerpt from chunk
    related_to: List[str]          # obligation IDs e.g. ["OBL-001", "OBL-003"]
    source_chunks: List[str]       # src:N references
    notes: Optional[str] = None    # LLM reasoning

class BinPassOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    findings: List[BinFinding] = Field(default_factory=list)
```

### 1c. Add StructureScanResult schema

```python
class ObligationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_letter: str            # "A", "B", "C"...
    heading: str                   # full heading text
    section_id: str                # e.g. "SEC-012"
    cfr_citations: List[str]       # CFR cites found in this section
    structured_chunk_ids: List[str] # src:N ids of final + codified chunks

class StructureScanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    obligation_sections: List[ObligationSection]
    named_section_chunk_ids: List[str]  # effective dates, scope, exemptions
    expected_obligation_count: int       # upper bound from heading count
    structured_chunk_ids: List[str]      # all chunks going to extractor
```

### Tests
- BinFinding validates correctly, ignores extra fields
- BinFinding rejects invalid finding_type
- StructureScanResult validates correctly

---

## Step 2 -- structure_scan()  (new file: sec_interpreter/structure.py)

### What it does
Reads sections.json and chunks.json, identifies obligation sections from
headings, collects final + codified chunks, extracts CFR citations.

### Function signature
```python
def structure_scan(artifact_dir: str) -> StructureScanResult:
```

### Logic
1. Load sections.json, chunks.json
2. Find the "Discussion of Final Amendments" parent section by heading
3. Find lettered subsections (A., B., C.) directly under it
4. For each lettered section:
   a. Collect chunks where subsection_role=final OR has_codified_text=True
   b. Run extract_references_from_text on section text -> CFR citations
   c. Record as ObligationSection
5. Find named sections by heading keywords:
   "effective", "compliance date", "applicability", "exemption", "codified text"
   -> collect their chunk IDs as named_section_chunk_ids
6. expected_obligation_count = count of lettered sections
   (sections whose Final Amendments say "not adopting" still count
    toward the heading count but will produce 0 obligations -- gap check handles)
7. structured_chunk_ids = union of all final + codified chunk IDs

### Tests
- finds lettered sections correctly on sections.json fixture
- collects final chunks per section, skips proposed/comments
- extracts CFR citations from section text
- finds named sections by heading keyword matching
- returns correct expected_obligation_count
- handles document with no "Discussion of Final Amendments" section (graceful empty result)

---

## Step 3 -- gap_check()  (sec_interpreter/structure.py)

### What it does
Compares extraction output against structure scan result.
Flags potential misses without modifying any output.

### Function signature
```python
def gap_check(
    extraction_output: dict,
    scan_result: StructureScanResult,
    logger: logging.Logger,
) -> dict:   # returns gap_report dict
```

### Logic
1. Count extracted obligations vs expected_obligation_count
   -> flag if extracted < expected
2. Collect all CFR citations from extracted obligations (cited_sections field)
3. For each ObligationSection in scan_result:
   - Check if any of its cfr_citations appear in extracted obligation cites
   - If none match -> flag that section as potentially missed
4. Return gap_report:
   {
     expected_count: int,
     extracted_count: int,
     count_gap: int,
     flagged_sections: [{ section_letter, heading, cfr_citations, reason }]
   }

### Tests
- returns no gaps when counts match and all CFR cites covered
- flags count gap when extraction has fewer obligations than expected
- flags section when none of its CFR cites appear in extracted obligations
- handles empty cfr_citations on sections gracefully

---

## Step 4 -- bin pass prompt  (sec_interpreter/prompts.py)

### Add build_bin_pass_prompt()

```python
def build_bin_pass_prompt(
    flagged_chunks: List[dict],   # {src_id, text, heading, flags}
    known_obligations: List[dict], # extracted obligations for context
) -> str:
```

### Prompt structure
- Show known obligations as context (so LLM looks for gaps not dupes)
- Show flagged chunks as a numbered list with src_id + heading + full text
- Ask LLM to classify each chunk into one of the bin types
- Ask for related_to obligation IDs for each finding
- Schema hint with BinFinding structure
- Rules: if in doubt use not_relevant, only flag missed_obligation if
  genuinely new and not already covered by known obligations

### Tests
- prompt contains known obligation IDs
- prompt contains chunk text
- prompt contains valid bin type list

---

## Step 5 -- bin pass runner  (new file: sec_interpreter/bin_graph.py)

### What it does
Filters remaining chunks by flags, runs bin pass LLM call, saves findings.

### Function signature
```python
def run_bin_pass(
    run_id: str,
    extraction_output: dict,
    cheap_llm: Any,
    logger: logging.Logger,
) -> BinPassOutput:
```

### Logic
1. Load chunks.json, get structured_chunk_ids from structure_scan artifact
2. remaining = all chunks NOT in structured_chunk_ids
3. flagged_remainder = [c for c in remaining if
       c.has_obligations OR c.has_codified_text OR c.has_scope]
4. If flagged_remainder empty -> return empty BinPassOutput
5. Build prompt via build_bin_pass_prompt(flagged_remainder, known_obligations)
6. Call cheap_llm
7. Parse -> BinPassOutput.model_validate
8. For any finding with type=missed_obligation:
   -> log warning, these need human review
9. Save bin_findings.json

### Tests
- filters correctly: only flagged chunks reach LLM
- returns empty output when no flagged remainder
- parses LLM output into BinPassOutput correctly
- handles LLM failure gracefully (empty output, log warning)
- missed_obligation findings are logged as warnings

---

## Step 6 -- case brief prompt  (sec_interpreter/prompts.py)

### Add build_case_brief_prompt()

```python
def build_case_brief_prompt(
    extraction_output: dict,
    bin_findings: BinPassOutput,
    interpretation_output: dict,
    named_section_texts: List[str],
) -> str:
```

### Output sections
1. Rule name, release number, CFR provisions
2. Scope -- who it applies to (from named sections + scope_modifier findings)
3. Exemptions -- who is excluded
4. Core obligations -- one per OBL-ID with rule_provision and deadline
5. Key definitions -- from bin_findings type=definition
6. Scope modifiers -- from bin_findings type=scope_modifier
7. Implied requirements -- from bin_findings type=implied_requirement
8. Effective and compliance dates -- from named_section_texts
9. Edge cases -- from bin_findings type=edge_case
10. What it means in practice -- from interpretation_output summaries

---

## Step 7 -- wire into extract_graph.py

### Changes
- Remove locator pass entirely
  (remove locator_pass function, LocatorSelection loading, locator_selection.json write)
- Add structure_scan() call after loading chunks
- Use scan_result.structured_chunk_ids to filter chunks sent to extractor
- Add gap_check() call after extraction, save gap_report.json
- Add rule_provision to extractor prompt schema hint

### Keep
- skip_locator flag for backward compat with direct/inline mode tests
  (when skip_locator=True, send all chunks to extractor as before)

---

## Step 8 -- wire into interpret_graph.py

### Changes
- Load bin_findings.json if it exists
- Replace _build_initial_context context assembly with bin_findings lookup:
  filter bin_findings by related_to containing this obligation_id
- Remove _link_family_context function (linker -- replaced by bin pass)
- Remove get_section_family_chunks call from per-obligation loop
- Keep CFR fetching (still useful, independent of bin pass)
- Keep definition lookup as fallback if bin_findings has no definitions

---

## Step 9 -- CLI updates  (sec_interpreter/cli.py)

### New subcommands
- `scan`   -- run structure scan on an existing ingest run, print scan result
- `bin`    -- run bin pass on an existing extract run
- `brief`  -- generate case brief from existing extract + interpret + bin outputs

### Update `run` subcommand
- run: ingest -> scan -> extract -> gap check -> bin -> interpret -> brief

### Update `extract` subcommand
- use structure scan instead of locator

---

## Step 10 -- tests

### New test files
- tests/test_structure.py   (structure_scan, gap_check)
- tests/test_bin.py         (bin pass runner, BinPassOutput schema)
- tests/test_case_brief.py  (build_case_brief_prompt)

### Update existing tests
- test_tools.py -- no changes needed (get_section_family_chunks stays for now)
- test_linker.py -- mark as deprecated but keep passing
  (linker code stays until interpret_graph.py is rewired in step 8)

---

## Build Order

```
Step 1  schemas.py            -- no dependencies
Step 2  structure.py          -- depends on Step 1
Step 3  structure.py          -- depends on Step 2
Step 4  prompts.py            -- depends on Step 1
Step 5  bin_graph.py          -- depends on Steps 1, 4
Step 6  prompts.py            -- depends on Step 1
Step 7  extract_graph.py      -- depends on Steps 2, 3
Step 8  interpret_graph.py    -- depends on Step 5
Step 9  cli.py                -- depends on Steps 2-8
Step 10 tests                 -- depends on all above
```

## Verification at each step

After each step: python -m pytest tests/ -q
All existing tests must pass throughout.
New tests added at each step before moving to next.
