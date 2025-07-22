import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_device_metrics_fallback_to_cpu(dependency_stubs):
    torch = dependency_stubs["torch"]
    torch.cuda.is_available = lambda: False
    import sys
    sys.modules.pop("chain_of_thought_wrapper", None)
    from chain_of_thought_wrapper import get_device_metrics
    metrics = get_device_metrics("cuda:0")
    assert "cpu_memory_used_mb" in metrics
    assert "cpu_memory_total_mb" in metrics
