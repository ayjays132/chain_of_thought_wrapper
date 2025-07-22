import os
import sys

import pytest

# Make package importable when running tests from the repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HeLLo", "hello"),
        ("hello!!!", "hello"),
        ("The apple is red.", "apple is red"),
        ("I have two apples.", "i have 2 apples"),
        ("  hello   world  ", "hello world"),
        ("The answer is: FOUR.", "4"),
        ("Result- three!", "3"),
        ("Output: nine", "9"),
        ("A quick BROWN Fox.", "quick brown fox"),
        ("Output -   ten.", "10"),
        ("Ninety-nine bottles!", "99 bottles"),
        ("twenty-one pilots", "21 pilots"),
        ("I have IV apples.", "i have 4 apples"),
        ("Chapter XI", "chapter 11"),
        (123, ""),
        (["not", "a", "string"], ""),
    ],
)
def test_normalize_answer(raw, expected, dependency_stubs):
    """Verify various normalization behaviours of normalize_answer."""
    from chain_of_thought_wrapper import normalize_answer

    assert normalize_answer(raw) == expected
