# 🚀 NeuroReasoner Chain-of-Thought Toolkit

NeuroReasoner wraps any Hugging Face model with automatic chain-of-thought prompting. It provides a Streamlit GUI, benchmarking utilities, and optional memory helpers.

## ⚙️ Installation
```bash
pip install cot-toolkit
# or from source
pip install -r requirements.txt
```

## ✨ Features
- **Always-on CoT prompting** with optional self-consistency
- **Streamlit GUI** for interactive chats
- **Simple RAG helper** for lightweight retrieval
- **Saved memories** that persist across prompts

## 👩‍💻 Quick Start
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from cot_toolkit import ChainOfThoughtWrapper

model_id = "sshleifer/tiny-gpt2"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
wrapper = ChainOfThoughtWrapper(model=model, processor=tok, device="cpu")
wrapper.remember("My name is Alice")
wrapper.rag_helper.add_document("Jupiter is the largest planet.")
result = wrapper.generate("Who am I and what is the largest planet?", generation_params={"max_new_tokens": 16})
print(result["final_answers"][0])
```

## 🖥️ Launch the GUI
```bash
streamlit run chain_of_thought_gui.py
```
Use the sidebar to pick a model and toggle self-consistency.

## ⏳ Example GUI Session
```
▶ Prompt: What causes rainbows?
▶ Chains: 3 (self-consistency)
…generating…
1. Sunlight is made of many colours.
2. Water droplets split the light.
3. The observer sees the separated colours.
Final Answer: Rainbows appear when light refracts through droplets.
```

## 📊 Benchmarking
Run the helper to compare CoT prompting vs plain generation:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from cot_toolkit import ChainOfThoughtWrapper, benchmark_prompt

model_id = "sshleifer/tiny-gpt2"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
wrapper = ChainOfThoughtWrapper(model=model, processor=tok, device="cpu")
wrapper.remember("benchmark demo")
metrics = benchmark_prompt(wrapper, "What is the largest planet?", {"max_new_tokens": 16})
print(metrics)
```
### 📈 Example Output
```
{'cot_duration': 0.25, 'plain_duration': 0.14,
 'cot_answer': 'factors factors …',
 'plain_answer': 'stairs stairs …', 'cot_steps': 0}
```

## 📚 Memory & RAG Tips
- `wrapper.remember(text)` stores a phrase.
- `wrapper.get_memories()` lists stored items.
- `wrapper.rag_helper.add_document(text)` adds retrieval context.
- `wrapper.rag_search(query)` returns top matching docs.

## 📜 License
MIT
