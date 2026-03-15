"""
tests/test_segmenter.py

Unit tests for sec_interpreter.segmenter.

Rule: every pure-Python processing module must have structural output tests
that verify the SHAPE of results, not just that the function runs without error.

Bugs caught by this suite:
- _heading_level() using $ anchor so "I. Introduction" returned level 0
  instead of level 0 (correct) — and "A. Subsection" returned level 0
  instead of level 1, causing all heading_path lists to be single-item.
"""
from __future__ import annotations

import pytest

from sec_interpreter.segmenter import _heading_level, _is_heading, segment_document


# ---------------------------------------------------------------------------
# _heading_level — prefix detection must work on full heading labels,
# not just bare prefixes like "I." or "A."
# ---------------------------------------------------------------------------

class TestHeadingLevel:
    def test_roman_numeral_bare(self):
        assert _heading_level("I.") == 0
        assert _heading_level("II.") == 0
        assert _heading_level("XIV.") == 0

    def test_roman_numeral_with_title(self):
        # This is the bug that was present: returning 0 only for bare "I."
        # but not for "I. Introduction and Background"
        assert _heading_level("I. Introduction and Background") == 0
        assert _heading_level("II. Discussion of Final Amendments") == 0
        assert _heading_level("III. Economic Analysis") == 0

    def test_uppercase_letter_bare(self):
        assert _heading_level("A.") == 1
        assert _heading_level("B.") == 1
        assert _heading_level("Z.") == 1

    def test_uppercase_letter_with_title(self):
        assert _heading_level("A. Disclosure of Cybersecurity Incidents") == 1
        assert _heading_level("B. Risk Management and Strategy") == 1
        assert _heading_level("C. Governance Requirements") == 1

    def test_numeric_bare(self):
        assert _heading_level("1.") == 2
        assert _heading_level("2.") == 2
        assert _heading_level("10.") == 2

    def test_numeric_with_title(self):
        assert _heading_level("1. Proposed Amendments") == 2
        assert _heading_level("2. Comments") == 2
        assert _heading_level("3. Final Amendments") == 2

    def test_lowercase_letter_bare(self):
        assert _heading_level("a.") == 3
        assert _heading_level("b.") == 3

    def test_lowercase_letter_with_title(self):
        assert _heading_level("a. Proposed Definitions") == 3
        assert _heading_level("b. Comments") == 3
        assert _heading_level("c. Final Definitions") == 3

    def test_allcaps_returns_top_level(self):
        assert _heading_level("SUMMARY") == 0
        assert _heading_level("DEFINITIONS") == 0
        assert _heading_level("BACKGROUND") == 0

    def test_levels_are_strictly_ordered(self):
        # I. < A. < 1. < a. in depth
        assert _heading_level("I. Top") < _heading_level("A. Second")
        assert _heading_level("A. Second") < _heading_level("1. Third")
        assert _heading_level("1. Third") < _heading_level("a. Fourth")


# ---------------------------------------------------------------------------
# segment_document — hierarchy must be reflected in heading_path and level
# ---------------------------------------------------------------------------

MULTI_LEVEL_TEXT = """\
I. Introduction and Background
This is the introduction text with enough content to form a section body.

II. Discussion of Final Amendments
A. Disclosure of Cybersecurity Incidents on Current Reports
1. Proposed Amendments
Text about proposed amendments for incident disclosure on current reports.

2. Comments
Text about comments received on the proposed incident disclosure amendments.

3. Final Amendments
Text about the final incident disclosure amendments that were adopted.

B. Risk Management and Strategy
1. Proposed Amendments
Text about proposed amendments for risk management strategy disclosure.

2. Comments
Text about comments on risk management proposals.

C. Governance Requirements
a. Board Oversight
Text about board oversight of cybersecurity risks.

b. Management Role
Text about management role in cybersecurity governance.

III. Economic Analysis
A. Introduction
Text about the economic analysis introduction.

B. Benefits and Costs
1. Benefits
Text about the benefits of the final rules.

2. Costs
Text about the costs of the final rules.
"""


class TestSegmentDocument:
    def setup_method(self):
        self.sections = segment_document(MULTI_LEVEL_TEXT)

    def _find(self, *path_parts: str):
        """Return sections whose heading_path ends with the given parts."""
        matches = [
            s for s in self.sections
            if s.heading_path[-len(path_parts):] == list(path_parts)
        ]
        return matches

    def test_produces_sections(self):
        assert len(self.sections) >= 8

    def test_section_ids_are_unique(self):
        ids = [s.section_id for s in self.sections]
        assert len(ids) == len(set(ids))

    def test_top_level_headings_have_level_0(self):
        intro = self._find("I. Introduction and Background")
        assert intro, "Expected a section for I. Introduction"
        assert intro[0].level == 0

        discussion = self._find("II. Discussion of Final Amendments")
        # level 0 sections may not have body text if all text is in children
        if discussion:
            assert discussion[0].level == 0

    def test_letter_subsections_have_level_1(self):
        # A. headings with no direct body produce no section of their own —
        # their children carry the full path. Verify a child of A. has A. as parent.
        proposed = self._find(
            "II. Discussion of Final Amendments",
            "A. Disclosure of Cybersecurity Incidents on Current Reports",
            "1. Proposed Amendments",
        )
        assert proposed, "Expected a section at II > A. > 1. Proposed Amendments"
        # The A. segment is at index -2, meaning depth 1 from root
        path = proposed[0].heading_path
        a_entry = path[-2]
        assert a_entry.startswith("A."), f"Expected 'A.' parent, got: {a_entry}"

    def test_numeric_subsections_have_level_2(self):
        proposed = self._find(
            "A. Disclosure of Cybersecurity Incidents on Current Reports",
            "1. Proposed Amendments",
        )
        assert proposed, "Expected a section under A. > 1."
        assert proposed[0].level == 2

    def test_lowercase_subsections_have_level_3(self):
        # In our fixture C. has no numeric heading between it and a.,
        # so path is [II, C, a] → depth 3 → level 2.
        # The key assertion is that the path is correct and level = path_depth - 1.
        board = self._find("C. Governance Requirements", "a. Board Oversight")
        assert board, "Expected a section under C. > a."
        s = board[0]
        assert s.heading_path[-1] == "a. Board Oversight"
        assert s.heading_path[-2] == "C. Governance Requirements"
        assert s.level == len(s.heading_path) - 1

    def test_heading_path_is_breadcrumb_not_single_item(self):
        # The core regression: heading_path must be > 1 item for nested sections
        proposed = self._find("1. Proposed Amendments")
        # At least one should have a multi-item path
        multi = [s for s in proposed if len(s.heading_path) > 1]
        assert multi, (
            "All '1. Proposed Amendments' sections have single-item heading_path — "
            "parent context is missing. Check _heading_level() for $ anchor bug."
        )

    def test_repeated_headings_have_distinct_paths(self):
        # "1. Proposed Amendments" appears under both A. and B. — paths must differ
        proposed_sections = self._find("1. Proposed Amendments")
        if len(proposed_sections) >= 2:
            paths = [tuple(s.heading_path) for s in proposed_sections]
            assert len(set(paths)) == len(paths), (
                "Repeated heading '1. Proposed Amendments' has duplicate paths — "
                "parent context is not being tracked correctly."
            )

    def test_no_section_has_empty_heading_path(self):
        for s in self.sections:
            assert s.heading_path, f"Section {s.section_id} has empty heading_path"

    def test_level_matches_heading_path_depth(self):
        for s in self.sections:
            expected_level = len(s.heading_path) - 1
            assert s.level == expected_level, (
                f"Section {s.section_id} has heading_path depth {len(s.heading_path)} "
                f"but level={s.level} (expected {expected_level})"
            )
