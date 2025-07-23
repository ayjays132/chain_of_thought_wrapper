import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_record_history_methods(dependency_stubs):
    sys.modules.pop("chain_of_thought_wrapper", None)
    from chain_of_thought_wrapper import ChainOfThoughtWrapper

    wrapper = ChainOfThoughtWrapper(model=None, processor=None, device="cpu")
    wrapper.add_record("note1")
    wrapper.add_record("note2")
    assert wrapper.get_records() == ["note1", "note2"]
    wrapper.clear_records()
    assert wrapper.get_records() == []
