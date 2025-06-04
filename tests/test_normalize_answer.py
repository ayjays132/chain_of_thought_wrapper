import os
import sys

import pytest

"""Unit tests for the :func:`normalize_answer` helper.

These checks cover lowercase conversion, punctuation stripping,
article removal, number word conversion, and whitespace cleanup.
The tests depend on the ``dependency_stubs`` fixture from ``conftest.py``
to inject lightweight module stubs so the wrapper imports cleanly without
heavy optional packages installed.
"""


# Make package importable when running tests from the repository root
=======
# abf47s-codex/create-pytest-module-for-normalize_answer
=======
# b3svha-codex/create-pytest-module-for-normalize_answer

# main
import pytest

"""Unit tests for the `normalize_answer` helper.

These checks cover lowercase conversion, punctuation stripping,
article removal, number word conversion, and whitespace cleanup.
"""

# Make package importable when running tests from the repository root
# abf47s-codex/create-pytest-module-for-normalize_answer
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

=======
=======
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

=======
import types

import pytest

# Make package importable when running from repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Provide lightweight stubs for optional heavy dependencies so that the
# chain_of_thought_wrapper module can be imported without the real packages
# being installed.  Only the minimal attributes used during import are mocked.
# ---------------------------------------------------------------------------

# torch stub ---------------------------------------------------------------
torch_stub = sys.modules.setdefault("torch", types.ModuleType("torch"))
torch_stub.device = lambda *args, **kwargs: None
torch_stub.Tensor = type("Tensor", (), {})
torch_stub.tensor = lambda *args, **kwargs: None
torch_stub.no_grad = lambda: (lambda f: f)

class DummyCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def empty_cache() -> None:
        pass

torch_stub.cuda = DummyCuda()

# transformers stub -------------------------------------------------------
transformers_stub = sys.modules.setdefault("transformers", types.ModuleType("transformers"))
for attr in (
    "PreTrainedModel",
    "AutoTokenizer",
    "GenerationConfig",
    "GenerationMixin",
    "AutoProcessor",
    "AutoModel",
    "AutoConfig",
):
    setattr(transformers_stub, attr, object)

utils_stub = sys.modules.setdefault("transformers.utils", types.ModuleType("transformers.utils"))
utils_stub.is_accelerate_available = lambda: False
utils_stub.is_bitsandbytes_available = lambda: False

# Additional optional modules -------------------------------------------
sys.modules.setdefault("PIL", types.ModuleType("PIL"))
sys.modules.setdefault("PIL.Image", types.ModuleType("Image"))
for name in (
    "Enhanced_MemoryEngine",
    "NeuroMemoryProcessor",
    "AGIEnhancer",
    "FullAGI_ExpansionModule",
    "SimulatedSelfAssessment",
):
    sys.modules.setdefault(name, types.ModuleType(name))

from chain_of_thought_wrapper import normalize_answer  # noqa: E402


@pytest.mark.parametrize(
    "raw,expected",
    [
=======
# abf47s-codex/create-pytest-module-for-normalize_answer
=======
# b3svha-codex/create-pytest-module-for-normalize_answer
# main
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

=======
# abf47s-codex/create-pytest-module-for-normalize_answer
=======
=======
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
        ("Ninety-nine bottles!", "ninety-nine bottles"),
#main
# main

        (123, ""),
        (["not", "a", "string"], ""),
    ],
)

=======
# abf47s-codex/create-pytest-module-for-normalize_answer
=======
# b3svha-codex/create-pytest-module-for-normalize_answer
# main

def test_normalize_answer(raw, expected, dependency_stubs):
    """Verify various normalization behaviours of normalize_answer."""
    from chain_of_thought_wrapper import normalize_answer

    assert normalize_answer(raw) == expected
=======
# abf47s-codex/create-pytest-module-for-normalize_answer
    assert normalize_answer(raw) == expected
=======
=======
def test_normalize_answer(raw, expected):
    """Verify various normalization behaviours of normalize_answer."""
#main
    assert normalize_answer(raw) == expected

# main
