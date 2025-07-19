import os
import sys

import pytest

# Make package importable when running tests from the repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



def test_fallback_when_cuda_unavailable(dependency_stubs, caplog):
    torch = dependency_stubs["torch"]
    torch.cuda.is_available = lambda: False
    import sys
    sys.modules.pop("chain_of_thought_wrapper", None)
    from chain_of_thought_wrapper import validate_device_selection
    result = validate_device_selection("cuda:0")
    assert result == "cpu"


def test_fallback_on_invalid_index(dependency_stubs, caplog):
    torch = dependency_stubs["torch"]
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    import sys
    sys.modules.pop("chain_of_thought_wrapper", None)
    from chain_of_thought_wrapper import validate_device_selection
    result = validate_device_selection("cuda:5")
    assert result == "cpu"


def test_valid_device_kept(dependency_stubs):
    torch = dependency_stubs["torch"]
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 2
    import sys
    sys.modules.pop("chain_of_thought_wrapper", None)
    from chain_of_thought_wrapper import validate_device_selection
    assert validate_device_selection("cuda:1") == "cuda:1"
