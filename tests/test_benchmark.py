import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def import_benchmark():
    return __import__("cot_toolkit.benchmark", fromlist=["benchmark_prompt"])


class DummyWrapper:
    def __init__(self):
        self.cot_instruction = "Explain step"

    def generate(self, prompt, generation_params=None, **kwargs):
        if self.cot_instruction:
            return {
                "final_answers": ["cot"],
                "reasoning_steps": [["a", "b"]],
                "generation_duration": 0.2,
            }
        return {
            "final_answers": ["plain"],
            "reasoning_steps": [],
            "generation_duration": 0.1,
        }


def test_benchmark_prompt(dependency_stubs):
    wrapper = DummyWrapper()
    sys.modules.pop("cot_toolkit.benchmark", None)
    benchmark_prompt = import_benchmark().benchmark_prompt
    metrics = benchmark_prompt(wrapper, "hi")
    assert metrics["cot_steps"] == 2
    assert metrics["cot_answer"] == "cot"
    assert metrics["plain_answer"] == "plain"
    assert wrapper.cot_instruction == "Explain step"
