import os
import sys
import types
import pytest

@pytest.fixture
def dependency_stubs():
    """Insert lightweight stubs for heavy optional dependencies."""
    stubs = {}

    torch_stub = types.ModuleType("torch")
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
    stubs["torch"] = torch_stub

    transformers_stub = types.ModuleType("transformers")
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

    utils_stub = types.ModuleType("transformers.utils")
    utils_stub.is_accelerate_available = lambda: False
    utils_stub.is_bitsandbytes_available = lambda: False

    stubs["transformers"] = transformers_stub
    stubs["transformers.utils"] = utils_stub

    stubs["PIL"] = types.ModuleType("PIL")
    stubs["PIL.Image"] = types.ModuleType("Image")

    for name in (
        "Enhanced_MemoryEngine",
        "NeuroMemoryProcessor",
        "AGIEnhancer",
        "FullAGI_ExpansionModule",
        "SimulatedSelfAssessment",
    ):
        stubs[name] = types.ModuleType(name)

    original_modules = {}
    for name, module in stubs.items():
        original_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    yield stubs

    for name, orig in original_modules.items():
        if orig is None:
            del sys.modules[name]
        else:
            sys.modules[name] = orig
