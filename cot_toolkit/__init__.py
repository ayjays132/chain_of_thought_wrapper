from importlib.metadata import version

from .benchmark import benchmark_prompt
from .simple_rag import SimpleRAG

# Functions are imported lazily to avoid circular imports when
# ``chain_of_thought_wrapper`` itself depends on this package.
def ChainOfThoughtWrapper(*args, **kwargs):
    from chain_of_thought_wrapper import ChainOfThoughtWrapper as _COT
    return _COT(*args, **kwargs)

def validate_device_selection(*args, **kwargs):
    from chain_of_thought_wrapper import validate_device_selection as _validate
    return _validate(*args, **kwargs)

def normalize_answer(*args, **kwargs):
    from chain_of_thought_wrapper import normalize_answer as _normalize
    return _normalize(*args, **kwargs)

__all__ = [
    "ChainOfThoughtWrapper",
    "validate_device_selection",
    "normalize_answer",
    "benchmark_prompt",
    "SimpleRAG",
]

try:
    __version__ = version("cot-toolkit")
except Exception:
    __version__ = "0.0.0"

