# SEC Rule Update Compliance Impact Extractor

A multi-stage Python pipeline that takes a raw SEC regulatory release and produces
structured compliance intelligence: obligations, affected entities, key dates, and
per-obligation interpretations with cited source passages.

Built to handle the full complexity of real SEC releases (100-300 pages, 150-200 chunks).
Uses document heading structure -- not keyword search or embeddings -- to identify
extraction targets. Verified on SEC Release 33-11216 (Cybersecurity Disclosure Rule, 2023).
Achieves 97.0% criterion coverage on 38 reference evaluation criteria.

---

## Pipeline Stages

```
[PDF or HTML URL]
       |
  [1. Ingest]       pure Python, no LLM -- fetch, segment, chunk, score
       |
  [2. Extract]      structure scan (no LLM) + one LLM call per obligation section
       |
  [3. Bin pass]     cheap LLM -- secondary scan over flagged remaining chunks
       |
  [4. Interpret]    LLM -- per-obligation context assembly + live CFR lookup
       |
  [validated_output.json, interpretation.json, bin_findings.json, *.brief.md]
```

| Stage | Module | What it does |
|-------|--------|-------------|
| **Ingest** | `ingest_graph.py` | Fetches URL or local file (PDF/HTML), segments into heading-aware sections, chunks to ~1000 tokens, scores each chunk with regex hot-zone flags. No LLM. |
| **Extract** | `extract_graph.py` + `structure.py` | Deterministic structure scan identifies obligation sections from headings; one LLM call per section with prior obligations as cross-section context. |
| **Bin pass** | `bin_graph.py` | Cheap LLM reviews flagged chunks the main extractor did not see; finds missed obligations, scope modifiers, edge cases. |
| **Interpret** | `interpret_graph.py` | Per-obligation: loads all final-rule chunks from the obligation's section family, looks up definitions, fetches live CFR text, produces structured interpretation. |
| **Report** | `report_formatter.py` | Renders a markdown compliance report with interpretation blocks and compliance implications per obligation. |

---

## Key Design Decisions

**Structure-first extraction.** Document heading structure (segmented at ingest) is used
to deterministically identify which chunks belong to which obligation section. No LLM
locator call, no keyword search, no embeddings.

**Section-per-call.** One LLM call per lettered obligation section (A through F). Thin
sections like XBRL (1 chunk) get their own call with full context rather than being
drowned out by larger sections (14 chunks).

**Final-only extraction.** The extractor only sees `subsection_role="final"` chunks and
codified text -- not proposed/comments sub-sections. This prevents extraction of
requirements the SEC considered but did not adopt.

**Section-family interpretation.** The interpreter loads ALL final-role chunks from an
obligation's section, not just the extractor-cited chunks. Ensures nuanced details like
phased timelines and national security exceptions are available to the interpreter.

**Citation traceability.** Every obligation and entity must cite source chunks via
`src:<N>` format. Citations are bounds-checked against the full chunk list. All claims
are auditable back to the original document.

**Pydantic validation throughout.** All LLM outputs are parsed into typed schemas.
Four business rule checks run after parse: citation bounds, strict citations, obligation
link validity, safe language (no "compliant/violation/penalty exposure" in output).

---

## Quick Start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment**

Create `.env`:
```
SEC_INTERPRETER_MODEL=gpt-4o
SEC_INTERPRETER_CHEAP_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

Without an API key the pipeline falls back to `DeterministicLLM` (offline stub).

**3. Run the full pipeline**

```bash
# Full pipeline in one command (ingest + extract + bin + interpret + brief)
python -m sec_interpreter.cli run \
  --url https://www.sec.gov/.../33-11216.pdf \
  --pages 5-60 \
  --output artifacts/output.json
```

Or stage by stage:

```bash
python -m sec_interpreter.cli ingest --input sec_rule.pdf --pages 1-50
# -> prints run_id

python -m sec_interpreter.cli extract   --run-id <run_id> --output out.json
python -m sec_interpreter.cli bin       --run-id <run_id>
python -m sec_interpreter.cli interpret --run-id <run_id> --output interp.json
```

---

## CLI Reference

```bash
python -m sec_interpreter.cli <subcommand> [options]
```

| Subcommand | Purpose | Key options |
|-----------|---------|------------|
| `ingest` | Fetch and chunk a document (no LLM) | `--url` or `--input`, `--pages 1-50` |
| `extract` | Structure scan + LLM extraction | `--run-id`, `--output`, `--strict` |
| `run` | Full pipeline: ingest + extract + bin + interpret + brief | `--url`/`--input`, `--output` |
| `scan` | Run structure scan, print obligation sections (no LLM) | `--run-id` |
| `classify` | Classify every section by content type | `--run-id` |
| `bin` | Secondary scan over flagged remaining chunks | `--run-id` |
| `interpret` | Per-obligation interpretation with CFR lookup | `--run-id`, `--output` |
| `gap` | Plain-English gap analysis vs company context | `--run-id`, `--company "..."` |
| `report` | Format markdown compliance report | `--run-id`, `--output` |
| `brief` | Generate case brief | `--run-id`, `--output` |
| `eval` | LLM-as-judge evaluation against reference criteria | `--run-id`, `--criteria` |
| `comprehend` | Calibration: classify every chunk | `--run-id` |

---

## Output Schema

The extractor produces `validated_output.json` (`RuleExtractorOutput`):

| Field | Type | Description |
|-------|------|-------------|
| `rule_metadata` | RuleMetadata | Rule title, release number, effective date, citations |
| `rule_summary` | RuleSummary | 3-5 sentence plain-language summary |
| `key_obligations` | List[KeyObligation] | Adopted requirements (OBL-001, OBL-002, ...) |
| `affected_entity_types` | List[AffectedEntityType] | Entity types in scope |
| `compliance_impact_areas` | List[ComplianceImpactArea] | Impact area tags |
| `assumptions` | List[Assumption] | Gaps filled by the model, not the document |

Each `KeyObligation`:
- `obligation_text` -- complete self-contained sentence with WHO, WHAT, WHEN, CONDITION
- `trigger` -- event or condition activating the obligation
- `deadline` -- timing requirement if stated
- `cited_sections` -- CFR sections / form items imposing this obligation
- `source_citations` -- list of `src:<N>` chunk references

The interpret stage adds `interpretation.json` (`InterpretationOutput`):
- `primary_interpretation` -- what the obligation means in practice
- `key_details` -- 3-5 specific implementation details
- `alternative_interpretations` -- plausible alternative readings
- `ambiguous_terms` -- terms needing definition lookup
- `compliance_implication` -- concrete operational impact
- `confidence_level` -- `"high"` | `"medium"` | `"low"`

---

## Test Suite

```bash
pytest tests/ -q
```

| File | What it covers |
|------|---------------|
| `test_validate_output.py` | Output validation, safe language enforcement |
| `test_strict_citations.py` | Citation bounds and strict citation rules |
| `test_no_sources_behavior.py` | Behavior when no source chunks are provided |
| `test_new_schemas.py` | KeyObligation, ObligationInterpretation, InterpretationOutput schemas |
| `test_segmenter.py` | Heading detection and section tree construction |
| `test_tools.py` | get_section_family_chunks, definition lookup, chunk search |
| `test_linker.py` | ObligationContextLinks schema and context linker prompt |

---

## Documentation

Detailed component documentation:

- [docs/ingest.md](docs/ingest.md) -- fetch, segment, chunk, score (Stage 1)
- [docs/extract.md](docs/extract.md) -- structure scan, section-per-call extraction (Stage 2)
- [docs/interpret.md](docs/interpret.md) -- context assembly, CFR lookup, bin pass (Stages 3-4)
- [docs/schemas_eval_cli.md](docs/schemas_eval_cli.md) -- Pydantic schemas, eval, CLI, utilities

---

## Artifacts

Each run produces `artifacts/{run_id}/`:

```
input.txt                  raw fetched rule text
sections.json              segmented document structure (List[Section])
chunks.json                scored RichChunk list
summary.txt                auto-extracted SUMMARY section
shortlist.json             compact index (src_id, heading, flags, preview)
ingest_log.txt             ingest run metadata

structure_scan_result.json obligation sections, chunk IDs, CFR citations (no LLM)
validated_output.json      RuleExtractorOutput (obligations, entities, areas)
raw_model_output.txt       raw LLM JSON from extraction
gap_report.json            expected vs extracted obligation count check
run_log.txt                extraction metadata and token usage
trace.jsonl                structured event log (all stages append)

bin_findings.json          BinPassOutput (missed obligations, edge cases)
interpretation.json        InterpretationOutput (one interpretation per obligation)
eval_report.json           LLM-as-judge coverage report
```
