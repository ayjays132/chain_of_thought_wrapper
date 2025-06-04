import os
import sys

import pytest

"""Unit tests for the `normalize_answer` helper.

These checks cover lowercase conversion, punctuation stripping,
article removal, number word conversion, and whitespace cleanup.
"""

# Make package importable when running tests from the repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



@pytest.mark.parametrize(
    "raw,expected",
    [
        # Basic lowercase conversion and punctuation stripping
        ("HeLLo", "hello"),
        ("hello!!!", "hello"),

        # Article and punctuation removal
        ("The apple is red.", "apple is red"),

        # Number word conversion
        ("I have two apples.", "i have 2 apples"),

        # Whitespace normalization
        ("  hello   world  ", "hello world"),

        # Common prefixes combined with number words
        ("The answer is: FOUR.", "4"),
        ("Result- three!", "3"),
        ("Output: nine", "9"),

        # Mixed casing with punctuation
        ("A quick BROWN Fox.", "quick brown fox"),

        # Extra spaces around prefix
        ("Output -   ten.", "10"),

        # Hyphenated words should remain intact
        ("Ninety-nine bottles!", "ninety-nine bottles"),

        # Non-string inputs should return an empty string
        (123, ""),
        (["not", "a", "string"], ""),
    ],
)
def test_normalize_answer(raw, expected, dependency_stubs):
    """Verify various normalization behaviours of normalize_answer."""
    from chain_of_thought_wrapper import normalize_answer

    assert normalize_answer(raw) == expected

