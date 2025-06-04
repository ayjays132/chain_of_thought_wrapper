import os
import sys
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
        (123, ""),
        (["not", "a", "string"], ""),
    ],
)
def test_normalize_answer(raw, expected):
    """Verify various normalization behaviours of normalize_answer."""
    assert normalize_answer(raw) == expected

