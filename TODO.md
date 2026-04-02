# TODO

## Open

### Issue 2b (minor) -- OBL-008 missing FPI Form 20-F board oversight criterion
FPI section E now extracted (OBL-014). But one OBL-008 criterion still fails:
"Foreign private issuers must disclose board oversight of cybersecurity risks and
management's role in the Form 20-F annual report."
The extracted OBL-014 covers 6-K incident disclosure; the 20-F annual governance
obligation may need a second FPI obligation entry.

### Issue 2b (minor) -- OBL-002 missing "incident ongoing at time of filing"
One of the 5 OBL-002 criteria is now failing. Low priority.

## Done

### Issue 1c -- FPI (OBL-008) and XBRL (OBL-007) obligations not extracted
Root cause: single LLM call over all 28 chunks; thin sections (1 chunk each) were drowned
out by large sections (section A, 14 chunks). LLM attention concentrated on dominant sections.
Fix: replaced single-call extraction with section-per-call loop (extract_sections_loop node).
Each obligation section now gets its own LLM call with prior obligations as cross-section context.
Result: 15 obligations extracted (was 6), OBL-007 2/2, OBL-008 2/3. Overall 94.7% (36/38).

### Issue 2a -- OBL-001 missing 120-day staged delay detail
Resolved automatically by section-per-call extraction: section A now has dedicated context.
Result: OBL-001 14/14 (perfect).

### Issue 1a -- OBL-006 governance obligations never extracted (root cause: subsection_role bug)
_derive_subsection_role checked heading_path[2] only. Section C chunks have "c. Final
Amendments" at heading_path[3], so they got role=other instead of role=final. The structure
scan only feeds final-role chunks to the extractor, so section C was skipped.
Fix: changed _derive_subsection_role to scan from deepest heading level upward.
Result: OBL-006 eval coverage 5/11 -> 11/11 (perfect).

### Issue 1b -- OBL-006 interpretation depth gaps (third-party assessors, ERM, etc.)
LLM was summarizing the main rule and omitting exceptions, scope extensions, and safe harbors.
Fix: added key_details field to ObligationInterpretation schema + explicit prompt rule to
enumerate all exceptions, carve-outs, scope extensions, and safe harbors.
Result: OBL-001 13/14, OBL-006 11/11. Overall 84.2% (32/38).
