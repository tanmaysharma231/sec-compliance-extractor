# SEC Compliance Pipeline -- Architecture

## Overview

Takes a raw SEC regulatory document (PDF or TXT) and produces three outputs:

```
validated_output.json   structured obligations            machine readable
interpretation.json     per-obligation legal guidance     compliance teams
case_brief.md           full rule summary                 lawyers / executives
```

---

## Full Pipeline

```
RAW DOCUMENT (PDF / TXT)
         |
         v
+-------------------------+
|  INGEST   (no LLM)      |
|                         |
|  fetch document         |
|  segment into sections  |  heading patterns, KNOWN_ANCHORS, ALL-CAPS
|  chunk each section     |  ~4000 chars, 600 char overlap
|  score every chunk      |  keyword flags:
|                         |    has_obligations (must/shall/required)
|                         |    has_codified_text (CFR amendment language)
|                         |    has_scope (who is covered)
|                         |    has_dates (deadlines, effective dates)
|                         |    has_definitions (defined terms)
|  set section_family     |  heading_path[1] -> obligation group
|  set subsection_role    |  proposed / comments / final / other
|                         |
|  save: sections.json    |
|        chunks.json      |
+-------------------------+
         |
         v
+-------------------------+
|  STRUCTURE SCAN (no LLM)|
|                         |
|  read section headings  |
|  find "Discussion of    |
|  Final Amendments"      |
|  find lettered sections |  A, B, C... -> obligation candidates
|  for each section:      |
|    collect final chunks |  subsection_role=final
|    collect codified     |  has_codified_text=True
|    extract CFR cites    |  extract_references_from_text (free)
|  count sections         |  -> expected obligation upper bound
|                         |
|  collect named sections |  "Effective Dates", "Applicability",
|  by heading regardless  |  "Exemptions", "Codified Text"
|  of flags               |  -> for case brief only
|                         |
|  output:                |
|    structured_chunks    |  ~25 chunks, ~15-20% of doc
|    named_section_chunks |  effective dates, scope, exemptions
|    section_map          |  section -> CFR citations
|    expected_obl_count   |  upper bound from heading count
+-------------------------+
         |
         v
+-------------------------+
|  LLM PASS 1 -- EXTRACT  |  1 main LLM call
|                         |
|  input: structured_     |
|  chunks only            |  clean, operative text, low noise
|                         |
|  output per obligation: |
|    obligation_id        |  OBL-001, OBL-002...
|    rule_provision       |  "17 CFR 229.106(b)" / "Item 1.05 Form 8-K"
|    obligation_text      |
|    trigger              |
|    deadline             |
|    disclosure_fields    |
|    evidence             |
|    cited_sections       |
|    source_citations     |
|                         |
|  save: validated_output |  draft, may be updated by bin pass
+-------------------------+
         |
         v
+-------------------------+
|  GAP CHECK  (no LLM)    |
|                         |
|  compare extracted      |
|  obligation count vs    |
|  expected_obl_count     |  flag if fewer than expected
|                         |
|  compare CFR citations  |
|  in obligations vs      |
|  section_map            |  flag any section CFR cites
|                         |  that never appear in obligations
|                         |
|  save: gap_report.json  |
+-------------------------+
         |
         v
+-------------------------+
|  FLAG FILTER  (no LLM)  |
|                         |
|  take all chunks NOT    |
|  in structured_chunks   |  ~136 remaining
|                         |
|  keep if:               |
|    has_obligations OR   |
|    has_codified_text OR |
|    has_scope            |
|                         |
|  flagged_remainder      |  ~40-50 chunks worth checking
|  unflagged              |  ~86 chunks -> skip entirely
|                         |  (economic analysis, procedural,
|                         |   footnotes, comment attribution)
+-------------------------+
         |
         v
+-------------------------+
|  LLM PASS 2 -- BIN      |  1 cheap LLM call
|                         |
|  input:                 |
|    flagged_remainder    |  ~40-50 chunks
|    known obligations    |  OBL-001..N as context
|                         |  (so LLM looks for gaps not dupes)
|                         |
|  for each chunk output: |
|    type:                |
|      missed_obligation  |  safety net -- new obligation found
|      scope_modifier     |  limits or extends who/what is covered
|      implied_requirement|  obligation implied but not explicit
|      definition         |  key term applicable across obligations
|      edge_case          |  SEC comment response, specific scenario
|      not_relevant       |  skip
|    related_to: []       |  which obligation IDs this links to
|    text                 |  relevant excerpt
|    source_chunks: []    |  src:N references
|                         |
|  if missed_obligation:  |  update validated_output.json
|                         |
|  save: bin_findings.json|
+-------------------------+
         |
         v
+-------------------------+
|  INTERPRET  (per obl.)  |  1 main LLM call per obligation
|                         |
|  input per obligation:  |
|    obligation text      |
|    bin_findings tagged  |  pre-linked -- lookup by OBL-ID
|    to this obligation   |  no runtime search needed
|    definitions from     |  bin_findings type=definition
|    bin pass             |
|    live CFR text        |  fetched from eCFR API
|                         |
|  output:                |
|    primary_interp.      |
|    alternative_interps  |
|    ambiguous_terms      |
|    compliance_impl.     |
|    confidence_level     |
|                         |
|  save: interpretation   |
+-------------------------+
         |
         v
+-------------------------+
|  CASE BRIEF  (1 call)   |  1 main LLM call
|                         |
|  input:                 |
|    validated_output     |
|    bin_findings         |
|    interpretation       |
|    named_section_chunks |  effective dates, scope, exemptions
|                         |
|  output sections:       |
|    rule name + release  |
|    scope                |  who it applies to
|    exemptions           |  who is excluded
|    core obligations     |  OBL-001..N with rule_provision
|    key definitions      |  from bin_findings type=definition
|    scope modifiers      |  from bin_findings type=scope_modifier
|    implied requirements |  from bin_findings type=implied_requirement
|    effective dates      |  from named_section_chunks
|    edge cases           |  from bin_findings type=edge_case
|    what it means        |  from interpretation.json
|                         |
|  save: case_brief.md    |
+-------------------------+
```

---

## LLM Call Summary

```
Step            Model       Calls               When
-----------     -------     ----------------    ------------------
Extract         main        1                   per run
Bin pass        cheap       1                   per run
Interpret       main        1 per obligation    per run
Case brief      main        1                   per run
```

Everything else is deterministic -- no LLM, no cost, fully reproducible.

---

## What Each Step Produces

```
sections.json           section segmentation with heading paths
chunks.json             RichChunk format with flags + section_family + subsection_role
structured_chunks       extraction targets from structure scan
named_section_chunks    effective dates, scope, exemptions (for case brief)
section_map             section -> CFR citations (for gap check)
gap_report.json         flagged missing obligations or CFR citations
validated_output.json   structured obligations (updated if bin pass finds misses)
bin_findings.json       tagged findings with obligation links
interpretation.json     per-obligation legal interpretation
case_brief.md           human readable rule summary
```

---

## Key Design Decisions

**No locator.** Replaced by structure scan. Locator had 18% precision and
missed real obligations on the calibration document.

**No classify pass.** Replaced by structure scan (structural understanding)
and bin pass (semantic understanding of flagged remainder). Classify was
doing with N LLM calls what structure + flags can do for free.

**No linker LLM call.** Bin pass pre-tags findings to obligation IDs.
Interpret does a lookup, not a search. Linker was doing per-obligation
what the bin pass already does once for the whole document.

**Flag filter before bin pass.** Flags are not used to select obligations
(too noisy for that) but are used to filter obvious noise before the LLM
sees anything. ~86 chunks never reach an LLM at all.

**Dates by heading not by flag.** has_dates flag fires on historical dates,
economic analysis, comment period dates -- too noisy. Effective dates and
compliance deadlines are collected directly by section heading in structure
scan.

**rule_provision on every obligation.** The specific CFR provision or form
item that imposes the obligation is a first-class field, not buried in a list.
```

---

## SEC Final Rule Document Structure (confirmed by calibration)

```
I.   Background / Introduction
II.  Discussion of Final Amendments
     A. [obligation topic]
        1. Proposed Rule          subsection_role = proposed   skip for extraction
        2. Comments Received      subsection_role = comments   use for interpretation
        3. Final Amendments       subsection_role = final      use for extraction
     B. [next obligation]
        ... same 3-part pattern ...
     D. [dropped provision]       Final Amendments says "not adopting" -- no obligation
III. Effective and Compliance Dates   named section -> collect for case brief
IV.  Economic Analysis                unflagged in most cases -> skip
V.   Codified Text                    has_codified_text=True -> in structured_chunks
```

The lettered sections under II are fixed and countable before any LLM runs.
That count is the expected obligation upper bound used in gap check.
