# Product Vision -- SEC Compliance Intelligence Pipeline

## What We Are Building

A pipeline that takes a raw SEC regulatory document and produces three
distinct outputs, each serving a different purpose.

---

## The Three Outputs

### 1. validated_output.json -- Structured Obligations
Machine readable. The primary extraction result.

```
{
  obligations: [OBL-001, OBL-002, OBL-003, OBL-004],
  entity_types: [...],
  compliance_impact_areas: [...],
  assumptions: [...]
}
```

Used for: downstream systems, gap analysis, automated compliance checks.

---

### 2. interpretation.json -- Per-Obligation Legal Interpretation
One entry per obligation. What it means in practice.

```
OBL-001:
  primary_interpretation: what this most likely means
  alternative_interpretations: where the law is genuinely ambiguous
  ambiguous_terms: material, promptly...
  compliance_implication: what the company must build
  confidence_level: high / medium / low
```

Used for: compliance teams, legal review, building controls.

---

### 3. case_brief.md -- Human Readable Rule Summary
A complete reference document for the rule. Structured like a legal case brief.
Can be handed to a lawyer, compliance officer, or board without further context.

```
RULE BRIEF: SEC Cybersecurity Disclosure Rule (2023)
Release: 33-11216 / 34-97989
CFR: 17 CFR Parts 229, 232, 239, 240, 249

SCOPE
  Who it applies to: all domestic registrants
  Exemptions: smaller reporting companies (extended deadlines)
  Foreign filers: use Form 20-F instead of Form 8-K

CORE OBLIGATIONS
  OBL-001: 8-K incident disclosure within 4 business days of materiality determination
  OBL-002: periodic updates if new material information emerges
  OBL-003: annual 10-K risk management process disclosure
  OBL-004: annual 10-K governance disclosure (board + management roles)

KEY DEFINITIONS
  material: information a reasonable investor would consider important
  cybersecurity incident: unauthorized occurrence jeopardizing confidentiality...
  information system: ...

SCOPE MODIFIERS
  - materiality clock starts at determination, not at breach
  - national security delay provision: AG can extend deadline
  - smaller reporting companies: 1 year phase-in period

IMPLIED REQUIREMENTS
  - companies must not unreasonably delay materiality determination
  - disclosures must reflect actual processes, not aspirational ones

EDGE CASES (from SEC comment responses)
  - what counts as a series of related incidents
  - third-party incidents: when does it trigger disclosure
  - board expertise: proposed but not adopted (see section D)

WHAT IT MEANS IN PRACTICE
  [from interpretation.json per obligation]
```

Used for: legal review, board briefings, training, due diligence.

---

## The Pipeline That Produces Them

```
RAW DOCUMENT
     |
     v
INGEST  (no LLM)
  segment + chunk + score + set section_family + subsection_role
     |
     v
STRUCTURE SCAN  (no LLM)
  read headings -> find obligation sections A, B, C...
  collect final + codified chunks per section
  count expected obligations (upper bound)
  extract CFR citations per section
     |
     v
LLM PASS 1 -- PRIMARY EXTRACTION
  input: structured chunks only (final + codified, ~15-20% of doc)
  output: draft obligations OBL-001 through OBL-N
  -> validated_output.json (primary obligations)
     |
     v
LLM PASS 2 -- BIN THE REST
  input: remaining chunks + OBL-001..N as context
  prompt: what is new, what modifies existing, what is generally applicable
  output: findings tagged by type and related_to obligation IDs
  types:
    - missed_obligation (new primary or secondary obligation)
    - scope_modifier (limits or extends who/what is covered)
    - implied_requirement (obligation implied but not explicit)
    - definition (key term applicable across obligations)
    - edge_case (SEC comment response, specific scenario)
    - not_relevant
  -> bin_findings.json
  -> also catches any missed primary obligations as safety net
     |
     v
GAP CHECK  (no LLM)
  compare extracted obligation count vs expected from structure scan
  compare CFR citations in obligations vs CFR citations found in sections
  flag any gaps for review
     |
     v
LLM PASS 3 -- INTERPRET  (one call per obligation)
  input: obligation + its tagged findings from bin pass (pre-linked by ID)
  output: interpretation per obligation
  -> interpretation.json
     |
     v
LLM PASS 4 -- CASE BRIEF  (one call)
  input: validated_output + bin_findings + interpretation
  output: case_brief.md
  sections: scope, obligations, definitions, modifiers,
            implied requirements, edge cases, what it means in practice
```

---

## Why This Works

**Structure-first extraction** means we know where the obligations are before
any LLM runs. The LLM does substance, not navigation.

**Bin pass as safety net** means we do not rely solely on document structure.
If an obligation is stated outside the lettered sections it gets caught.
The LLM sees the already-extracted obligations so it looks for gaps, not
duplicates.

**Pre-linked context** means interpretation does not search for context at
runtime. The bin pass already tagged which findings relate to which obligation.
Interpretation just does a lookup.

**Three output types** means the pipeline serves different audiences -- systems,
compliance teams, and executives/lawyers -- without each needing to re-read
the source document.

---

## What Needs To Be Built

1. structure_scan() -- reads sections.json, identifies obligation sections,
   collects final + codified chunks, runs CFR citation extraction

2. bin_pass() -- LLM pass over remaining chunks with extracted obligations
   as context, produces tagged findings with obligation links

3. gap_check() -- compares extracted obligations against structure scan map,
   flags potential misses

4. case_brief generator -- LLM pass that synthesises all outputs into
   human readable brief

5. Updated extract_graph.py -- replace locator with structure_scan output

6. Updated interpret_graph.py -- use pre-linked bin findings instead of
   runtime context assembly, remove linker LLM call

7. New schema -- BinFinding { type, text, related_to[], source_chunks[] }
