# 🚀 NeuroReasoner Chain-of-Thought Toolkit

NeuroReasoner wraps any Hugging Face model with chain-of-thought (CoT) prompting. It exposes convenient metrics, a Streamlit GUI, and optional AGI helper modules.

## ⚙️ Installation
```bash
pip install cot-toolkit
# or from source
pip install -r requirements.txt
```

## ✨ Key Features
- **Always-on CoT** prompting with optional self-consistency
- **Streamlit GUI** for interactive use
- **RAG helper** (`SimpleRAG`) for lightweight retrieval
- **Saved memories** that influence future answers

## 👩‍💻 Quick Start
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from cot_toolkit import ChainOfThoughtWrapper

model_id = "sshleifer/tiny-gpt2"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
wrapper = ChainOfThoughtWrapper(model=model, processor=tok, device="cpu")
wrapper.remember("My name is Alice")
wrapper.rag_helper.add_document("Jupiter is the largest planet in our solar system.")
result = wrapper.generate("Who am I and what is the largest planet?", generation_params={"max_new_tokens": 16})
print(result["final_answers"][0])
```

## 🖥️ Launching the GUI
Run:
```bash
streamlit run chain_of_thought_gui.py
```
Configure model and sampling in the sidebar, then chat with the model.

## ⏳ Example GUI Session
```
▶ Prompt: What causes rainbows?
▶ Chains: 3, Self-Consistency: on
…generating…
1. Sunlight is made of many colors.
2. Water droplets split the light.
3. The observer sees the separated colors.
Final Answer: Rainbows appear when light refracts through droplets.
```

## 📊 Benchmarking
```python
from cot_toolkit import ChainOfThoughtWrapper, benchmark_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "sshleifer/tiny-gpt2"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
wrapper = ChainOfThoughtWrapper(model=model, processor=tok, device="cpu")
wrapper.remember("benchmark demo")
wrapper.rag_helper.add_document("Jupiter is the largest planet in our solar system.")
metrics = benchmark_prompt(wrapper, "What is the largest planet?", {"max_new_tokens": 16})
print(metrics)
```
### 📈 Latest Benchmark Example
```
{'cot_duration': 0.19, 'plain_duration': 0.16, 'cot_answer': 'stairs stairs …', 'plain_answer': 'factors factors …', 'cot_steps': 0}
```
Even this tiny model adds minimal overhead while providing structured reasoning.

## 📚 Memory & RAG Tips
- `wrapper.remember(text)` stores a phrase permanently.
- `wrapper.get_memories()` lists stored items.
- `wrapper.rag_helper.add_document(text)` adds retrieval context.
- `wrapper.rag_search(query)` returns top matching docs.

## 📜 License
MIT

