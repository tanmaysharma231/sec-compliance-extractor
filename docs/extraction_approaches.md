# Extraction Approaches -- Decision Document

## Current Approach (Locator-based)

### Pipeline
```
ingest -> locator LLM -> extract -> interpret
```

### Step by step

**1. Ingest (no LLM)**
- Fetch document (PDF or TXT)
- Segment into sections using heading patterns (KNOWN_ANCHORS, ALL-CAPS, numbered)
- Chunk each section into ~4000 char pieces with 600 char overlap
- Score every chunk with keyword flags:
  has_obligations, has_dates, has_scope, has_definitions, has_codified_text
- Set section_family (heading_path[1]) and subsection_role (proposed/comments/final/other)
  on every chunk
- Save: sections.json, chunks.json, summary.txt

**2. Locator pass (1 cheap LLM call)**
- Build a compact table: one row per chunk
  src_id | heading (40 chars) | flags | chars | preview (100 chars)
- Feed all 161 rows to cheap LLM (gpt-4o-mini) in one call
- LLM selects src_ids by category:
  date_chunks, scope_chunks, obligation_chunks, definition_chunks, other_key_chunks
- Save: locator_selection.json

**3. Extract (1 main LLM call)**
- Load only the chunks selected by locator (~20-30 chunks)
- Feed to main LLM (gpt-4o) with extraction prompt
- LLM produces structured output:
  obligations, entity types, impact areas, assumptions
- Validate citations, safe language, obligation format
- Save: validated_output.json

**4. Interpret (1 main LLM call per obligation)**
- For each obligation:
  - Look up section_family -> get_section_family_chunks (comments + final)
  - Run linker LLM call (cheap) to filter family chunks per obligation
  - Look up definitions for ambiguous terms
  - Fetch live CFR text from eCFR API
  - Feed obligation + context bundle to main LLM
  - Produce ObligationInterpretation
- Save: interpretation.json

### What is used for extraction
- All chunk types selected by locator LLM judgment
- No structural awareness -- locator decides purely from 100-char previews + flags

### What is used for interpretation
- section_family structural lookup -> comments + final chunks for that section
- Linker LLM filters those down per obligation
- Definitions from classified definition sections
- Live CFR text

### Known problems
- 100-char previews lose signal -- cannot distinguish real obligation from dropped provision
- Locator missed src:43/src:45 (Item 106(c) governance) on real SEC document
- Locator picked src:57 (dropped board expertise provision) as false positive
- No sanity check on obligation count -- no way to know something was missed
- Locator precision on calibration run: 18% (18 false positives out of 22 selected)
- Locator coverage: 67% (missed 2 of 6 truly important chunks)

---

## Planned Approach (Structure-first + Classify for interpretation)

### Pipeline
```
ingest -> structure scan -> extract -> classify comments only -> interpret
```

### Step by step

**1. Ingest (no LLM) -- same as current**
- Fetch, segment, chunk, score, set section_family + subsection_role
- Save: sections.json, chunks.json

**2. Structure scan (no LLM)**
- Read section headings from sections.json
- Find top-level lettered sections under "Discussion of Final Amendments"
  -> these are the obligation candidate sections (A, B, C, D, ...)
- Count them -> this is the expected obligation upper bound
- For each lettered section, collect:
  - subsection_role=final chunks -> operative text
  - has_codified_text=True chunks -> CFR amendment text
- Also collect the Codified Text section at end of document
- Run extract_references_from_text on each section -> store CFR citations per section
- Output: extraction_targets (list of chunks), section_map (section -> CFR citations)

**3. Extract (1 main LLM call)**
- Feed only extraction_targets chunks to main LLM (~15-20% of document)
- LLM produces structured output: obligations, entity types, impact areas, assumptions
- Validate output
- Gap check (no LLM):
  - Compare obligation count against expected count from structure scan
  - Compare CFR citations in obligations against CFR citations found in section_map
  - Flag any sections whose CFR citations never appear in extracted obligations
- Save: validated_output.json, extraction_targets.json

**4. Classify comments/commentary only (cheap LLM, N calls, once per document)**
- Only classify sections with subsection_role=comments or subsection_role=other
  (skip final, proposed, codified -- structure already tells us what those are)
- For each: content_type, summary, topics, useful_for
- Run extract_references_from_text -> store CFR citations per section
- Save: section_classifications.json

**5. Interpret (1 main LLM call per obligation)**
- For each obligation:
  - Look up its CFR citations (e.g. 229.106(c))
  - Find comment sections whose CFR citations overlap -> targeted context
  - Look up definitions for ambiguous terms
  - Fetch live CFR text from eCFR API
  - Feed obligation + context bundle to main LLM
  - Produce ObligationInterpretation
- Save: interpretation.json

### What is used for extraction
- subsection_role=final chunks + has_codified_text chunks only
- Deterministic -- no LLM judgment involved in chunk selection
- Typically 15-20% of document

### What is used for interpretation
- Comment sections whose CFR citations match the obligation's cited sections
- Classify output used to further filter by content_type if needed
- Definitions from definition sections (detected by heading or classify)
- Live CFR text

### Advantages over current
- No false negatives from locator -- if the section exists, its final chunks are included
- No false positives from locator -- only final/codified text, not commentary
- Structural sanity check built in -- know expected obligation count before LLM runs
- CFR gap detection built in -- know if an extracted obligation missed a cited section
- Classify cost reduced -- only runs on comments/commentary, not entire document
- Faster and cheaper per extraction run
- Fully reproducible -- same document always produces same extraction targets

### Risks and open questions
- Depends on subsection_role being correctly assigned at ingest time
  -> need to verify subsection_role coverage on existing run before committing
- Heading pattern variation across documents:
  "Final Amendments" vs "Adopted Rule" vs "Final Rule Text"
  -> subsection_role assignment must handle all variants
- Less generalisable -- assumes SEC Final Rule release format
  -> proposed rules, interpretive releases, other regulators need different approach
- If segmenter mislabels a heading, error is silent
  -> need logging of which chunks were selected as extraction targets

### What needs to be built
1. structure_scan() function -- reads sections.json, identifies obligation sections,
   collects final+codified chunks, runs CFR citation extraction
2. gap_check() function -- compares extraction output against structure scan map
3. Update extract_graph.py -- replace locator pass with structure_scan output
4. Update classify step -- only run on comments/other sections, skip final/codified/proposed
5. Update interpret_graph.py -- use CFR citation matching instead of section_family lookup
   for finding interpretation context; remove linker LLM call

---

## Summary Comparison

```
                    Current (Locator)       Planned (Structure-first)
                    -----------------       -------------------------
Navigation          LLM (1 call)            Deterministic (no LLM)
Extraction input    Locator judgment        Final + codified chunks only
Doc coverage        ~20-30 chunks           ~15-20% of doc
False negatives     Yes (18% miss rate)     Near zero if subsection_role correct
False positives     Yes (82% noise)         Near zero
Sanity check        None                    Built in (heading count + CFR gap)
Classify needed     No (locator replaces)   Yes but only for comments sections
Interpret context   section_family lookup   CFR citation matching
Linker LLM call     Yes (per obligation)    No (deterministic)
Generalisability    Any document            SEC Final Rule releases primarily
Risk                Silent misses           Heading mislabelling
```
