# SEC Rule Update Compliance Impact Extractor

A multi-stage Python pipeline that takes a raw SEC regulatory release and produces
structured compliance intelligence: obligations, affected entities, key dates, and
per-obligation interpretations with cited source passages.

Built to handle the full complexity of real SEC releases (100-300 pages, 150-200 chunks),
where naive single-pass LLM extraction fails due to context size and noise. The pipeline
filters out ~60% of document noise (economic analysis, public comments, SEC commentary)
before any extraction LLM sees the text.

Verified on SEC Release 33-11216 (Cybersecurity Disclosure Rule, 2023).

---

## Pipeline Stages

```
ingest -> classify -> extract -> interpret -> report
```

| Stage | Graph | What it does |
|-------|-------|-------------|
| **Ingest** | `ingest_graph.py` | Fetches URL or local file (PDF/HTML), segments into heading-aware sections, chunks to ~1000 tokens, scores each chunk with regex hot-zone flags (dates, scope, obligations, definitions). No LLM. |
| **Classify** | `classify_graph.py` | One LLM call per section group -> assigns a content_type (7 types). Synthesises a DocumentMap. Filters chunks to final_rule_text + obligation + definition before extraction. |
| **Extract** | `extract_graph.py` | Two LLM calls: cheap locator narrows to relevant chunks, full extractor produces structured RuleExtractorOutput with Pydantic validation and citation tracing. |
| **Interpret** | `interpret_graph.py` | Per-obligation: definition lookup + surrounding context + live CFR fetch (eCFR API) + LLM interpretation. Agentic reference resolution loop (max 2 hops). |
| **Report** | `report_formatter.py` | Renders a markdown compliance report with interpretation blocks, cited passages, and compliance implications per obligation. |

---

## Key Design Decisions

**Citation traceability.** Every obligation, entity type, and compliance area must cite
at least one source chunk using `src:<N>` format. Citations are validated against the
full chunk list (bounds check + strict mode enforcement). This makes every claim
auditable back to the original document.

**Pydantic validation throughout.** All LLM outputs are parsed into typed schemas with
field validators. Four business rule checks run after parse: citation bounds, strict
citations, obligation link validity, safe language (no "compliant/non-compliant/violation"
language in output).

**No LLM at ingest.** The ingest stage is pure Python. Document segmentation and chunk
scoring use heading detection and regex patterns. This makes ingestion fast, deterministic,
and free.

**Content-type filtering before extraction.** The classify stage labels each chunk with
one of 7 content types. The extractor only sees final_rule_text, obligation, and definition
chunks -- typically 40% of the document. This cuts token cost and improves precision.

---

## Quick Start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment**

```bash
cp .env.example .env
# Edit .env and add your API key
```

The pipeline defaults to `gpt-4o` via OpenAI. Set `SEC_INTERPRETER_MODEL` and
`SEC_INTERPRETER_MODEL_PROVIDER` to use Anthropic or Google models instead.

**3. Run the full pipeline**

```bash
# Ingest a PDF (pages 1-50)
python -m sec_interpreter.cli ingest --input sec_rule.pdf --pages 1-50

# Classify the ingested document
python -m sec_interpreter.cli classify --run-id <run_id>

# Extract obligations and key fields
python -m sec_interpreter.cli extract --run-id <run_id>

# Interpret each obligation
python -m sec_interpreter.cli interpret --run-id <run_id>

# Generate markdown report
python -m sec_interpreter.cli report --run-id <run_id>
```

Or run ingest + extract in a single step:

```bash
python -m sec_interpreter.cli run --input sec_rule.pdf --pages 1-50
```

---

## CLI Reference

All subcommands:

```bash
python -m sec_interpreter.cli <subcommand> [options]
```

| Subcommand | Purpose | Key options |
|-----------|---------|------------|
| `ingest` | Fetch and chunk a document | `--input <path or URL>`, `--pages 1-50` |
| `extract` | Extract structured fields from ingested chunks | `--run-id <id>` |
| `run` | Ingest + extract in one step | `--input <path or URL>`, `--pages 1-50` |
| `classify` | Label chunks by content type, build DocumentMap | `--run-id <id>` |
| `interpret` | Per-obligation CFR lookup + LLM interpretation | `--run-id <id>` |
| `gap` | Gap analysis: obligations vs company description | `--run-id <id>`, `--company "..."` |
| `report` | Render markdown compliance report | `--run-id <id>`, `--output report.md` |
| `comprehend` | Calibration: classify every chunk, compare with locator | `--run-id <id>` |

Example end-to-end run:

```bash
python -m sec_interpreter.cli run --input sec_cybersec.pdf --pages 1-50
python -m sec_interpreter.cli classify --run-id <run_id>
python -m sec_interpreter.cli interpret --run-id <run_id>
python -m sec_interpreter.cli report --run-id <run_id> --output report.md
```

---

## Output Schema

The extractor produces a `RuleExtractorOutput` with these top-level fields:

| Field | Type | Description |
|-------|------|-------------|
| `rule_metadata` | RuleMetadata | Rule name, CFR citation, effective date, trigger |
| `rule_summary` | RuleSummary | One-paragraph plain-language summary |
| `key_obligations` | List[KeyObligation] | Actionable requirements (OBL-001, OBL-002, ...) |
| `affected_entity_types` | List[AffectedEntityType] | Entity types in scope |
| `compliance_impact_areas` | List[ComplianceImpactArea] | Impact area tags |
| `assumptions` | List[Assumption] | Gaps filled by the model, not the document |

Each `KeyObligation` includes:

- `obligation_id` (OBL-001 format)
- `description` -- what the rule requires
- `deadline` -- compliance deadline if stated
- `cited_sections` -- list of `src:<N>` citations
- `compliance_implication` -- plain-language impact on a regulated firm

The interpret stage adds an `ObligationInterpretation` per obligation:

- `primary_interpretation` -- what the obligation means in practice
- `alternatives` -- plausible alternative readings
- `ambiguous_terms` -- terms needing further definition
- `compliance_implication` -- concrete operational impact
- `confidence_level` -- low / medium / high

---

## Test Suite

55 tests, all passing.

```bash
pytest tests/ -q
```

Test files:

| File | What it covers |
|------|---------------|
| `test_validate_output.py` | Output validation, safe language enforcement |
| `test_strict_citations.py` | Citation bounds and strict citation rules |
| `test_no_sources_behavior.py` | Behavior when no source chunks are provided |
| `test_new_schemas.py` | RichChunk, Section, LocatorSelection schema validation |
| `test_segmenter.py` | Heading detection and section tree construction |
| `test_tools.py` | CFR fetch, definition lookup, commentary search |

---

## Status & Roadmap

### What is done
- Full pipeline operational: ingest -> classify -> extract -> interpret -> report
- Document segmentation, chunk scoring, and content-type filtering (no LLM at ingest)
- Structured extraction with Pydantic validation and citation tracing
- Per-obligation interpretation with live CFR lookups and commentary search
- Markdown report generation
- 55 passing tests
- Verified end-to-end on SEC Release 33-11216 (Cybersecurity Disclosure Rule)

### What we are working on now
The infrastructure is in place. The current focus is intelligence quality -- making the
output actually useful for a compliance team, not just structurally valid.

Concretely:
- **Obligation completeness:** the extractor currently finds 3-4 of 6+ obligations in a
  typical release. Improving recall without flooding the context window.
- **Interpretation depth:** per-obligation interpretations need to be specific enough that
  a compliance officer can act on them, not just restate the rule text.
- **Verification on a second rule:** all calibration so far is on one document. Need to
  confirm the pipeline generalises before drawing conclusions about quality.

### The intelligence problem we are solving

SEC regulatory releases follow a consistent 3-part structure within every topic section:

```
A. Some Rule Topic
    a. Proposed Amendments    <- what the SEC originally proposed
    b. Comments               <- industry letters raising edge cases and objections
    c. Final Amendments       <- what the SEC actually decided (the operative rule text)
```

This means the comments and commentary chunks -- which we currently filter out before
extraction -- contain exactly the legal reasoning that makes an interpretation useful:
what ambiguous scenarios industry raised, and how the SEC said it would apply the rule.

The next step is to use the `heading_path` field already present on every chunk to
group chunks by their parent section. For each obligation, we pull the full "section
family" -- proposed text, industry comments, SEC responses, and final rule -- and feed
that as context to the interpretation LLM. This replaces keyword search with a
structurally grounded legal context package, so interpretations are based on how the
rule was actually debated and decided, not just what the final text says.

### What comes next
- Section-family context builder: group chunks by heading_path to link comment/response pairs
- Gap analysis layer: given a company description, map obligations to operational gaps
- Multi-rule comparison: surface conflicts and overlaps across related releases
- Structured output for downstream use (API, dashboard integration)

---

## Architecture

See `SYSTEM_OVERVIEW.md` for the full pipeline diagram, module table, schema hierarchy,
citation system details, and backward compatibility notes.
