# Ingest Pipeline

## Overview

Stage 1 of the SEC compliance extraction system. Pure Python, no LLM calls.
Takes a raw SEC regulatory document (PDF or HTML URL, or local file) and produces
structured, scored chunks ready for downstream extraction and analysis.

Five transformations in sequence:
1. **Fetch** -- retrieve raw document text from URL or local file
2. **Segment** -- parse heading structures into labeled sections
3. **Chunk** -- split sections into size-bounded text chunks with overlap
4. **Score** -- flag chunks for dates, scope, obligations, definitions, codified text
5. **Summary** -- auto-extract SUMMARY / SUPPLEMENTARY INFORMATION section text

---

## Files

| File | Responsibility |
|------|----------------|
| `sec_interpreter/ingest_graph.py` | LangGraph state machine, chunking logic, artifact writing |
| `sec_interpreter/segmenter.py` | Line-by-line heading parser, Section builder |
| `sec_interpreter/scorer.py` | Regex hot-zone flagging, Locator index rows |
| `sec_interpreter/ingest.py` | `fetch_rule_text()` -- PDF/HTML/text fetch and clean |

---

## Data Flow

```
IngestInput (source, page_range, strict_citations)
    |
[fetch_document]     -> rule_text (str)
    |
[segment_document]   -> List[Section]
    |
[chunk_sections]     -> List[RichChunk]  (src:0, src:1, ...)
    |
[score_chunks]       -> List[RichChunk]  (boolean flags populated)
    |
[extract_summary]    -> summary_text (str)
    |
[save_ingest_artifacts] -> artifacts/{run_id}/
```

---

## Graph: `ingest_graph.py`

### State: `IngestState`

```
ingest_input   IngestInput        -- source, page_range, strict_citations
run_id         str                -- unique 12-char hex (uuid4().hex[:12])
rule_text      str                -- raw fetched document text
sections       List[Section]      -- segmented hierarchical sections
chunks         List[RichChunk]    -- size-bounded chunks with scoring flags
summary_text   str                -- auto-extracted summary block
```

### Chunk Sizing Constants

```python
_TARGET_CHARS   = 4000   # ~1000 tokens -- target chunk size
_HARD_CAP_CHARS = 6400   # ~1600 tokens -- absolute split boundary
_OVERLAP_CHARS  = 600    # ~150 tokens  -- carried from previous chunk
```

Target 4000 balances context window usage with granularity. Hard cap 6400
handles oversized single paragraphs. Overlap 600 ensures continuity across
chunks (a sentence referencing something from the previous chunk stays in scope).

---

### Node: `fetch_document`

Generates `run_id`. Calls `ingest.fetch_rule_text(source, page_range)` which
handles PDF extraction (pdfminer), HTML scraping, and plain text files.
Returns `rule_text` and `run_id`.

---

### Node: `segment_document`

Calls `segmenter.segment_document(rule_text)`. Returns flat `List[Section]`
with hierarchy encoded in each section's `heading_path`.

---

### Node: `chunk_sections`

Iterates sections, calls `_chunk_section()` per section with a running global
index so every chunk gets a unique `src_id` (src:0, src:1, ...).

#### `_chunk_section(section, start_global_idx, ...) -> (List[RichChunk], int)`

Splits one section into chunks respecting paragraph boundaries:

1. Split `section_text` on blank lines to get paragraphs
2. Accumulate paragraphs into `current_parts` until adding the next would exceed
   `_TARGET_CHARS` (4000) -- flush as a chunk
3. If a single paragraph exceeds `_HARD_CAP_CHARS` (6400), slice it directly
4. Every non-first chunk within a section prepends the last 600 chars of the
   previous chunk (overlap) for continuity
5. Each chunk records:
   - `src_id`: `"src:{global_idx}"`
   - `section_family`: `heading_path[1]` -- the obligation-level section letter
     (e.g. "A. Disclosure of Cybersecurity Incidents on Current Reports")
   - `subsection_role`: derived by `_derive_subsection_role()` (see below)
   - `chunk_index_in_section`: counter within the section

#### `_derive_subsection_role(heading_path) -> str`

Scans heading_path from deepest level upward (from index 2 onward) looking for:
- `"final"` in any heading -> `"final"`
- `"proposed"` in any heading -> `"proposed"`
- `"comment"` in any heading -> `"comments"`
- Otherwise -> `"other"`

Critical for downstream filtering: the ExtractGraph selects only `final`-role
chunks from obligation sections. This scan goes deepest-first so sections nested
4 levels deep (e.g. "c. Final Amendments" at heading_path[3]) are correctly
classified rather than picking up a shallower heading.

---

### Node: `score_chunks`

Calls `scorer.score_chunk(chunk)` for every chunk. Populates boolean flags.

---

### Node: `extract_summary`

Two strategies to find summary text:

**Strategy 1 -- Structured section match:**
Checks if `section.heading_path[0]` matches `_SUMMARY_ANCHORS`
(`{"SUMMARY", "SUPPLEMENTARY INFORMATION"}`). Collects matching sections' text.

**Strategy 2 -- Inline label scan (fallback):**
Scans `rule_text` line by line for patterns like `"SUMMARY: ..."`. Collects
lines until a next label pattern (`DATES:`, `EFFECTIVE DATE:`, etc.) is hit.

Used when Federal Register text embeds sections inline rather than as headings.

---

### Node: `save_ingest_artifacts`

Writes all artifacts to `artifacts/{run_id}/`:

| File | Format | Contents |
|------|--------|----------|
| `input.txt` | plain text | Raw fetched rule text |
| `sections.json` | JSON array | `List[Section]` model dumps |
| `chunks.json` | JSON array | `List[RichChunk]` model dumps |
| `summary.txt` | plain text | Extracted summary block |
| `shortlist.json` | JSON array | Locator index rows (one per chunk) |
| `ingest_log.txt` | plain text | run_id, source, counts, timestamp |

---

## Segmenter: `segmenter.py`

### `segment_document(text) -> List[Section]`

Parses SEC Federal Register documents line by line into hierarchical sections.

**Heading detection (`_is_heading(line)`)** accepts a line as a heading if:
- It matches a `KNOWN_ANCHOR` (SUMMARY, DATES, BACKGROUND, etc.)
- It starts with a numbered prefix (Roman numeral, uppercase letter, decimal)
  AND contains >= 2 consecutive letters (excludes bare CFR cites like "229.103.")
- It is ALL-CAPS, short (< 80 chars), has no sentence-terminal punctuation,
  does not end in a digit (excludes journal cites), does not start with `(`

**Hierarchy levels (`_heading_level(label)`):**
- Roman numeral prefix (I., II.) -> Level 0 (top)
- Uppercase letter prefix (A., B.) -> Level 1
- Decimal/integer prefix (1., 1.05.) -> Level 2
- Lowercase letter prefix (a., b.) -> Level 3
- ALL-CAPS / known anchor -> Level 0

**Section IDs (`_build_section_id(level_counters, level)`):**
8-character fixed-width string, 2 digits per level x 4 levels.
Example: counters `[2, 1, 3, 0]` -> `"02010300"`.
Enables sort-friendly comparison and decoding.

**Known anchors:**
```
SUMMARY, DATES, EFFECTIVE DATE, COMPLIANCE DATE, APPLICABILITY,
SUPPLEMENTARY INFORMATION, BACKGROUND, DISCUSSION, AMENDMENTS, DEFINITIONS
```

---

## Scorer: `scorer.py`

### `score_chunk(chunk) -> RichChunk`

Applies regex patterns to chunk text (and heading path) to set boolean flags.

| Flag | Patterns detect | Example signal |
|------|----------------|----------------|
| `has_dates` | effective date, compliance date, N days after, transition, phase-in | "effective date", "30 days after" |
| `has_scope` | applies to, registrant, issuer, exempt, does not apply | "applies to all registrants" |
| `has_obligations` | must, shall, is required, we are adopting, CFR, Item, Rule | "registrants shall file" |
| `has_definitions` | means, definition, for purposes of, as defined in | "'material' means" |
| `has_codified_text` | five-asterisk ellipsis, "to read as follows", specific CFR sections | "* * * * *", "to read as follows" |
| `has_example` | for example, for instance, if a company, if the registrant | "for example, if a registrant" |

`has_codified_text` is set if either the chunk text OR its heading path matches
codified patterns. Heading-based patterns: PART 22x/23x/24x, "Add §", "Amend §",
"Revise Form", "GENERAL INSTRUCTIONS", "List of Subjects".

The codified flag is the strongest signal for legally operative amendment text --
five-asterisk ellipsis and "to read as follows" never appear in commentary.

### `build_index_row(chunk) -> dict`

Produces compact metadata row for Locator LLM prompt / `shortlist.json`:
```json
{
  "src_id": "src:15",
  "heading": "II. Discussion -- A. Incident Disclosure",
  "chars": 3847,
  "flags": ["obl", "scope", "codified"],
  "preview": "Registrants shall file Form 8-K within 4 business days..."
}
```

Three independent signals (flags, heading, preview) so the Locator works
correctly even when any single signal is weak.

---

## Artifacts Produced

```
artifacts/{run_id}/
    input.txt          raw rule text
    sections.json      List[Section] -- heading_path, section_id, section_text
    chunks.json        List[RichChunk] -- src_id, text, flags, section_family, subsection_role
    summary.txt        extracted summary block
    shortlist.json     compact index for Locator (src_id, heading, flags, preview)
    ingest_log.txt     run metadata
```

---

## Performance

No LLM calls. End-to-end typically 200--850 ms per document.
Segmentation, chunking, scoring all linear in document length.
Suitable for batch processing.
