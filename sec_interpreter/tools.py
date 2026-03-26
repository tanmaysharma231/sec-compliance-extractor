"""
sec_interpreter/tools.py

Non-LLM tool functions used by the interpretation pipeline.

- lookup_definition         : search classified definition sections in the document
- get_surrounding_context   : pull sections adjacent to an obligation's section
- get_section_family_chunks : structural lookup -- all commentary/comments in same heading family
- search_document           : keyword search across commentary/comments (fallback)
- fetch_cfr                 : fetch live CFR text from the eCFR public API
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import List, Optional, Set

logger = logging.getLogger("sec_interpreter")

# ---------------------------------------------------------------------------
# Ambiguous term detection
# ---------------------------------------------------------------------------

# Common legally ambiguous terms in SEC rules worth looking up
AMBIGUOUS_TERMS = [
    "material", "materiality", "promptly", "reasonable", "reasonably",
    "significant", "substantial", "timely", "appropriate", "adequate",
    "effective", "necessary", "principal", "affiliated", "control",
]


def detect_ambiguous_terms(obligation_text: str) -> List[str]:
    """Return known ambiguous terms found in the obligation text."""
    text_lower = obligation_text.lower()
    found = []
    for term in AMBIGUOUS_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", text_lower):
            found.append(term)
    return found


# ---------------------------------------------------------------------------
# lookup_definition
# ---------------------------------------------------------------------------

def lookup_definition(term: str, artifact_dir: str) -> Optional[str]:
    """
    Search classified definition sections in the document for a term.

    Uses section_classifications.json to find definition-type sections,
    then searches sections.json text for the term.

    Returns section_text (truncated to 2000 chars) or None if not found.
    """
    sections_path = os.path.join(artifact_dir, "sections.json")
    classifications_path = os.path.join(artifact_dir, "section_classifications.json")

    if not os.path.exists(sections_path) or not os.path.exists(classifications_path):
        return None

    with open(classifications_path, encoding="utf-8") as f:
        classifications = json.load(f)
    with open(sections_path, encoding="utf-8") as f:
        sections = json.load(f)

    # Find section_ids classified as definition
    definition_ids = {
        sc["section_id"] for sc in classifications
        if sc.get("content_type") == "definition"
    }

    term_lower = term.lower()
    pattern = re.compile(r"\b" + re.escape(term_lower) + r"\b", re.IGNORECASE)

    for section in sections:
        if section.get("section_id") not in definition_ids:
            continue
        text = section.get("section_text", "")
        if pattern.search(text):
            logger.debug("lookup_definition(%r): found in section %s", term, section["section_id"])
            return text[:2000]

    return None


# ---------------------------------------------------------------------------
# get_surrounding_context
# ---------------------------------------------------------------------------

def get_surrounding_context(
    section_id: str,
    artifact_dir: str,
    window: int = 2,
) -> List[str]:
    """
    Return the text of `window` sections before and after the given section_id.

    Provides discussion context surrounding an obligation in the document.
    Returns a list of section texts (may be empty if section not found).
    """
    sections_path = os.path.join(artifact_dir, "sections.json")
    if not os.path.exists(sections_path):
        return []

    with open(sections_path, encoding="utf-8") as f:
        sections = json.load(f)

    ids = [s["section_id"] for s in sections]
    try:
        idx = ids.index(section_id)
    except ValueError:
        return []

    start = max(0, idx - window)
    end = min(len(sections), idx + window + 1)

    context = []
    for i in range(start, end):
        if i == idx:
            continue  # skip the section itself
        text = sections[i].get("section_text", "")
        heading = " > ".join(sections[i].get("heading_path", []))
        context.append(f"[{heading}]\n{text[:1500]}")

    return context


# ---------------------------------------------------------------------------
# search_document
# ---------------------------------------------------------------------------

_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "of", "to", "in", "is", "are", "be",
    "that", "this", "it", "its", "with", "for", "on", "at", "by", "from",
    "as", "was", "were", "has", "have", "had", "not", "but", "if", "any",
    "all", "such", "other", "under", "also", "must", "shall", "may",
}


def get_section_family_chunks(
    section_id: str,
    artifact_dir: str,
    subsection_roles: List[str] = None,
) -> List[dict]:
    """
    Return dict records for all chunks in the same section family as the given
    section_id, filtered to the requested subsection_roles.

    Uses section_family and subsection_role fields on RichChunk -- both set at
    ingest time from heading_path, no classify stage required.

    subsection_roles defaults to ["comments", "final"], which covers industry
    edge-case Q&A and SEC reasoning -- the primary interpretation context.

    Each returned dict has keys: src_id, subsection_role, heading, text.
    Falls back to empty list if chunks.json is missing or chunks predate this field.
    """
    if subsection_roles is None:
        subsection_roles = ["comments", "final"]

    chunks_path = os.path.join(artifact_dir, "chunks.json")
    if not os.path.exists(chunks_path):
        logger.debug("get_section_family_chunks: no chunks.json in %s", artifact_dir)
        return []

    with open(chunks_path, encoding="utf-8") as f:
        raw_chunks = json.load(f)

    # Find the section_family for the given section_id
    family = ""
    for c in raw_chunks:
        if c.get("section_id") == section_id:
            family = c.get("section_family", "")
            break

    if not family:
        logger.debug(
            "get_section_family_chunks: section_id=%r not found or has no section_family",
            section_id,
        )
        return []

    role_set = set(subsection_roles)
    results = []
    for c in raw_chunks:
        if c.get("section_family") != family:
            continue
        if c.get("subsection_role") not in role_set:
            continue
        results.append({
            "src_id": c.get("src_id", ""),
            "subsection_role": c.get("subsection_role", ""),
            "heading": " > ".join(c.get("heading_path", [])),
            "text": c.get("text", ""),
        })

    logger.debug(
        "get_section_family_chunks: family=%r  roles=%s  matched=%d chunks",
        family, subsection_roles, len(results),
    )
    return results


def search_document(
    query: str,
    artifact_dir: str,
    content_types: List[str] = None,
    top_n: int = 3,
    prefer_examples: bool = True,
) -> List[str]:
    """
    Keyword search across classified sections (default: commentary + comments).

    Algorithm (no LLM, no embeddings):
      1. Load section_classifications.json, filter to target content_types.
      2. Score each section against query keywords (stopwords removed).
         - Count keyword hits against the section summary.
         - Bonus +2 if any chunk in that section has has_example=True.
      3. Sort by score descending, take top_n.
      4. Load full section_text from sections.json.
      5. Return list of "[heading] text..." strings (truncated to 2000 chars each).

    Returns empty list if classifications artifact is missing (graceful degradation).
    """
    if content_types is None:
        content_types = ["commentary", "comments"]

    sections_path = os.path.join(artifact_dir, "sections.json")
    classifications_path = os.path.join(artifact_dir, "section_classifications.json")
    chunks_path = os.path.join(artifact_dir, "chunks.json")

    if not os.path.exists(sections_path) or not os.path.exists(classifications_path):
        logger.debug("search_document: missing artifacts in %s -- skipping", artifact_dir)
        return []

    with open(classifications_path, encoding="utf-8") as f:
        classifications = json.load(f)
    with open(sections_path, encoding="utf-8") as f:
        sections_list = json.load(f)

    # Build section_id -> section_text map
    section_text_map = {s["section_id"]: s for s in sections_list}

    # Build section_id -> has_example flag from chunks (optional, best-effort)
    example_section_ids: Set[str] = set()
    if prefer_examples and os.path.exists(chunks_path):
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        for chunk in chunks:
            if chunk.get("has_example"):
                sid = chunk.get("section_id")
                if sid:
                    example_section_ids.add(sid)

    # Tokenise query into keywords
    raw_tokens = re.findall(r"[a-zA-Z]{3,}", query.lower())
    keywords = [t for t in raw_tokens if t not in _STOPWORDS]
    if not keywords:
        return []

    # Filter classifications to target content_types
    target_set = set(content_types)
    candidates = [
        sc for sc in classifications
        if sc.get("content_type") in target_set
    ]

    # Score each candidate
    scored = []
    for sc in candidates:
        summary_lower = sc.get("summary", "").lower()
        heading_lower = " ".join(sc.get("heading_path", [])).lower()
        search_text = summary_lower + " " + heading_lower

        score = sum(1 for kw in keywords if kw in search_text)
        if prefer_examples and sc["section_id"] in example_section_ids:
            score += 2

        if score > 0:
            scored.append((score, sc["section_id"], sc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]

    # Build result strings
    results = []
    for _score, section_id, sc in top:
        section_data = section_text_map.get(section_id)
        if not section_data:
            continue
        heading = " > ".join(sc.get("heading_path", [])) or section_id
        text = section_data.get("section_text", "")
        results.append(f"[{heading}]\n{text[:2000]}")
        logger.debug(
            "search_document: section=%s score=%d example=%s",
            section_id, _score, section_id in example_section_ids,
        )

    logger.info(
        "search_document: query_keywords=%s  candidates=%d  returned=%d",
        keywords[:5], len(candidates), len(results),
    )
    return results


# ---------------------------------------------------------------------------
# fetch_cfr
# ---------------------------------------------------------------------------

_CFR_CITATION_RE = re.compile(
    r"(?:17\s+)?CFR\s+(?:Part\s+)?(\d+)\.(\d+)",
    re.IGNORECASE,
)

_ECFR_RENDERER = "https://www.ecfr.gov/api/renderer/v1/content/enhanced"
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s{2,}")


def fetch_cfr(citation: str, date: str = "current") -> Optional[str]:
    """
    Fetch live CFR section text from the eCFR renderer API.

    Supports formats like:
      "17 CFR 229.106(b)"
      "CFR 229.106"
      "Rule 13a-11"  -> not a CFR citation, returns None

    Returns plain text of the section (truncated to 3000 chars) or None.
    """
    match = _CFR_CITATION_RE.search(citation)
    if not match:
        logger.debug("fetch_cfr: cannot parse citation %r -- skipping", citation)
        return None

    part = match.group(1)
    section_num = match.group(2)
    title = "17"  # SEC rules are always Title 17
    section_id = f"{part}.{section_num}"

    # eCFR renderer endpoint returns HTML for the section
    url = f"{_ECFR_RENDERER}/{date}/title-{title}?part={part}&section={section_id}"

    logger.info("fetch_cfr: GET %s", url)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/xhtml+xml"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        logger.warning("fetch_cfr: HTTP %d for %r", e.code, citation)
        return None
    except Exception as e:
        logger.warning("fetch_cfr: request failed for %r: %s", citation, e)
        return None

    return _extract_text_from_html(html, citation)


def _extract_text_from_html(html: str, citation: str) -> Optional[str]:
    """Strip HTML tags and clean whitespace to get plain text."""
    # Replace block-level tags with newlines for readability
    html = re.sub(r"</?(p|div|h\d|li|br)[^>]*>", "\n", html, flags=re.IGNORECASE)
    text = _HTML_TAG_RE.sub("", html)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#167;", "section")
    text = text.replace("\xa0", " ")
    # Collapse whitespace
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    if not text:
        logger.warning("fetch_cfr: no text extracted for %r", citation)
        return None
    logger.info("fetch_cfr: extracted %d chars for %r", len(text), citation)
    return text[:3000]


# ---------------------------------------------------------------------------
# extract_references_from_text
# ---------------------------------------------------------------------------

def extract_references_from_text(text: str) -> List[str]:
    """
    Find CFR citation strings in a block of text.

    Used by the reference judge to identify what can be fetched next.
    Only returns citations that appear verbatim in the text (prevents hallucination).
    """
    pattern = re.compile(
        r"(?:17\s+)?C\.?F\.?R\.?\s+(?:Part\s+)?[\d]+\.[\d]+(?:\([a-z]\))?",
        re.IGNORECASE,
    )
    return list(dict.fromkeys(m.group() for m in pattern.finditer(text)))
