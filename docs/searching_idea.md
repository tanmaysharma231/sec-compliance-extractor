Document Processing Strategy – Heading-First Chunking Approach
1. Core Idea

Instead of blindly splitting a large SEC document into fixed-length chunks, we first preserve the document’s natural structure.

Regulatory documents are written in a hierarchical format:

Main headings

Subheadings

Nested subsections

Then detailed explanatory text

The idea is to respect this structure first, and only then apply size-based chunking.

2. Step-by-Step Logic
Step 1 – Extract Headings and Build Outline

We parse the document and detect:

Major section headings (e.g., I., II., III.)

Subsections (A., B., 1., 2., etc.)

Known structural markers (e.g., SUMMARY, DATES, SUPPLEMENTARY INFORMATION)

From this, we build a structured outline:

H1
→ H2
→ H3

Each section becomes a structured block with:

section_id

heading_path

section_text

This preserves context.

Step 2 – Convert Sections Into Chunk Candidates

Each section block is then evaluated for size.

If a section is small:

Keep it as one chunk.

If a section is large:

Split it into multiple token-sized chunks (e.g., 800–1500 tokens)

Preserve metadata on each chunk:

src_id

heading_path

chunk_index_in_section

This ensures:

No chunk loses its structural identity.

Citations remain meaningful.

Large sections are manageable for LLM processing.

Step 3 – Deterministic Hot-Zone Detection (Before LLM)

Instead of dumping everything into the LLM, we first identify likely important chunks.

We score chunks based on keyword signals:

Obligation indicators: "must", "shall", "required"

Scope indicators: "applies to", "covered", "registrant"

Date indicators: "effective date", "compliance date"

Rule language: "we are adopting", "amending", "Item", "Rule", "§"

Using Python-based scoring, we shortlist high-scoring chunks.

This reduces noise and improves precision.

Step 4 – LLM Locator Pass

We send to the LLM:

Chunk metadata

Heading paths

Possibly previews

Keyword flags

The LLM selects which chunks actually contain:

Rule metadata

Scope/applicability

Key obligations

Definitions (optional)

The output is just selected src_ids.

This avoids sending the entire document blindly.

Step 5 – LLM Extraction Pass

We then send only the selected chunks’ full text to the LLM.

The LLM extracts:

Rule metadata

Key obligations

Affected entities

Impact areas

All with strict src:<id> citations.

3. Why This Approach Is Better Than Dumping Everything

Even if SEC rules are not frequent:

Dumping 283 pages reduces extraction quality.

It increases hallucination risk.

It makes citation tracing harder.

It makes debugging painful.

The heading-first + funnel approach:

Preserves document structure.

Improves traceability.

Reduces noise.

Keeps the system explainable.

Scales to very large regulatory documents.

4. Design Philosophy

This approach balances:

Structure preservation
+
Deterministic filtering
+
LLM reasoning

It avoids:

Overengineering

Blind LLM dumping

Loss of structural context

Black-box behavior

5. Summary

The chunking strategy is:

Structure first (heading-aware segmentation).

Size-based splitting second.

Deterministic shortlist third.

LLM locator fourth.

LLM extraction last.

This keeps the system disciplined, explainable, and suitable for regulatory AI applications.