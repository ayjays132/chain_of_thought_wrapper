import os
import sys
import types

class GenerationConfigStub:
    def __init__(self):
        self.do_sample = False
        self.temperature = 1.0
        self.num_return_sequences = 1
        self.max_length = None
        self.max_new_tokens = None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return self.__dict__

# Make package importable when running tests from the repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.mock_length = 0

    def __call__(self, text, return_tensors=None, padding=None, truncation=True, max_length=None):
        length = self.mock_length
        return DummyBatch({"input_ids": DummyTensor(length), "attention_mask": DummyTensor(length)})

    def decode(self, ids, skip_special_tokens=True):
        return "Step 1: foo\nFinal Answer: bar"


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.image_processor = None


class DummyModel:
    def __init__(self):
        self.config = types.SimpleNamespace(max_position_embeddings=100)
        self.called_config = None

    def generate(self, *args, **kwargs):
        self.called_config = kwargs.get("generation_config")
        return types.SimpleNamespace(sequences=["dummy"])


def build_wrapper(max_length=10):
    from chain_of_thought_wrapper import ChainOfThoughtWrapper

    model = DummyModel()
    processor = DummyProcessor()
    wrapper = ChainOfThoughtWrapper(model=model, processor=processor, device="cpu", max_length=max_length)
    return wrapper, model, processor


def test_max_new_tokens_within_limit(dependency_stubs):
    sys.modules.pop("chain_of_thought_wrapper", None)
    dependency_stubs["transformers"].GenerationConfig = GenerationConfigStub
    wrapper, model, processor = build_wrapper(max_length=10)
    processor.tokenizer.mock_length = 5
    wrapper.generate("hello", generation_params={"max_new_tokens": 3})
    cfg = model.called_config
    assert cfg.max_new_tokens == 3
    assert cfg.max_length == 8


def test_max_new_tokens_exceeds_limit(dependency_stubs):
    sys.modules.pop("chain_of_thought_wrapper", None)
    dependency_stubs["transformers"].GenerationConfig = GenerationConfigStub
    wrapper, model, processor = build_wrapper(max_length=10)
    processor.tokenizer.mock_length = 8
    wrapper.generate("hello", generation_params={"max_new_tokens": 5})
    cfg = model.called_config
    assert cfg.max_new_tokens == 2
    assert cfg.max_length == 10


def test_long_input_ids_edge_case(dependency_stubs):
    sys.modules.pop("chain_of_thought_wrapper", None)
    dependency_stubs["transformers"].GenerationConfig = GenerationConfigStub
    wrapper, model, processor = build_wrapper(max_length=10)
    processor.tokenizer.mock_length = 12
    wrapper.generate("hello", generation_params={"max_new_tokens": 5})
    cfg = model.called_config
    assert cfg.max_new_tokens == 0
    assert cfg.max_length == 12
