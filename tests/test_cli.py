import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from importlib import import_module


def get_cli_module():
    return import_module("cot_toolkit.cli")


class DummyTensor:
    def __init__(self, length):
        self.shape = (1, length)
        self.dtype = 'int64'
        self.device = 'cpu'

    def numel(self):
        return self.shape[0] * self.shape[1]


class DummyBatch(dict):
    def to(self, device):
        return self

class DummyModel:
    def generate(self, *args, **kwargs):
        return types.SimpleNamespace(sequences=["dummy"])

class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def __call__(self, text, return_tensors=None, padding=None, truncation=True, max_length=None):
        return DummyBatch({"input_ids": DummyTensor(1), "attention_mask": DummyTensor(1)})
    def decode(self, ids, skip_special_tokens=True):
        return "Step 1: foo\nFinal Answer: bar"


def test_parse_args(dependency_stubs):
    sys.modules.pop("cot_toolkit.cli", None)
    cli = get_cli_module()
    args = cli.parse_args(["model", "prompt", "--device", "cpu", "--max-new-tokens", "5"])
    assert args.model == "model"
    assert args.prompt == "prompt"
    assert args.device == "cpu"
    assert args.max_new_tokens == 5


def test_main_runs(monkeypatch, dependency_stubs, capsys):
    stubs = dependency_stubs
    sys.modules.pop("cot_toolkit.cli", None)
    monkeypatch.setattr(stubs["transformers"], "AutoModel", types.SimpleNamespace(from_pretrained=lambda m: DummyModel()))
    monkeypatch.setattr(stubs["transformers"], "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda m: DummyTokenizer()))
    cli = get_cli_module()

    class DummyWrapper:
        def __init__(self, *a, **k):
            pass

        def generate(self, *a, **k):
            return {"final_answers": ["bar"], "full_texts": ["Step 1: foo\nFinal Answer: bar"]}

    monkeypatch.setattr(cli, "ChainOfThoughtWrapper", DummyWrapper)

    cli.main(["model", "prompt"])
    captured = capsys.readouterr()
    assert captured.out.strip() == "bar"
