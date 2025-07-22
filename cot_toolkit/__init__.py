from importlib.metadata import version

from chain_of_thought_wrapper import (
    ChainOfThoughtWrapper,
    validate_device_selection,
    normalize_answer,
)
from .benchmark import benchmark_prompt

__all__ = [
    "ChainOfThoughtWrapper",
    "validate_device_selection",
    "normalize_answer",
    "benchmark_prompt",
]

try:
    __version__ = version("cot-toolkit")
except Exception:
    __version__ = "0.0.0"
