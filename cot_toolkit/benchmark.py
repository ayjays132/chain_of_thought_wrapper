"""Utility for benchmarking ChainOfThoughtWrapper output."""
from __future__ import annotations

from typing import Any, Dict


def benchmark_prompt(wrapper: Any, prompt: str, generation_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Generate with and without chain-of-thought instructions and return metrics.

    Parameters
    ----------
    wrapper : Any
        Instance with ``generate`` method and ``cot_instruction`` attribute.
    prompt : str
        Prompt text to generate from.
    generation_params : dict | None
        Parameters forwarded to ``wrapper.generate``.

    Returns
    -------
    dict
        Dictionary containing generation durations and answers for both modes
        as well as the number of reasoning steps when CoT is enabled.
    """
    generation_params = generation_params or {}

    # Run generation with CoT enabled
    result_cot = wrapper.generate(prompt, generation_params=generation_params)

    # Temporarily disable CoT and run again
    original_instruction = getattr(wrapper, "cot_instruction", "")
    wrapper.cot_instruction = ""
    try:
        result_plain = wrapper.generate(prompt, generation_params=generation_params)
    finally:
        wrapper.cot_instruction = original_instruction

    metrics = {
        "cot_duration": result_cot.get("generation_duration"),
        "plain_duration": result_plain.get("generation_duration"),
        "cot_answer": (result_cot.get("final_answers") or [""])[0],
        "plain_answer": (result_plain.get("final_answers") or [""])[0],
        "cot_steps": len((result_cot.get("reasoning_steps") or [[]])[0]),
    }
    return metrics
