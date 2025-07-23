# 🚀 NeuroReasoner Chain-of-Thought Toolkit

**NeuroReasoner** wraps any Hugging Face language model with automatic *chain-of-thought* prompting and adds quality-of-life utilities such as a RAG helper, persistent memories and a simple Streamlit GUI.

---

## ⚙️ Installation
```bash
pip install cot-toolkit
# or from source
pip install -r requirements.txt
```

## ✨ Features
- **Always-on CoT prompting** with optional self-consistency
- **Streamlit GUI** for interactive conversations
- **Simple RAG helper** for lightweight retrieval
- **Saved memories** that persist across prompts
- **Benchmark utilities** for measuring CoT vs. plain generation

---

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
result = wrapper.generate("Who am I and what is the largest planet?", generation_params={"max_new_tokens":16})
print(result["final_answers"][0])
```

## 🖥️ Launch the GUI
```bash
streamlit run chain_of_thought_gui.py
```
Use the sidebar to select a model and toggle self-consistency.

---

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

---

## 📊 Benchmarking
Run the helper to compare CoT prompting with plain generation:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from cot_toolkit import ChainOfThoughtWrapper, benchmark_prompt

model_id = "sshleifer/tiny-gpt2"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
wrapper = ChainOfThoughtWrapper(model=model, processor=tok, device="cpu")
wrapper.remember("benchmark demo")
metrics = benchmark_prompt(wrapper, "What is the largest planet?", {"max_new_tokens":16})
print(metrics)
```

### 📈 Latest Benchmark Example
Output from running the above on a CPU instance:
```
{'cot_duration': 0.73, 'plain_duration': 0.30,
 'cot_answer': 'stairs stairs …',
 'plain_answer': 'stairs stairs …', 'cot_steps': 0}
```

---

## 📚 Memory & RAG Tips
- `wrapper.remember(text)` stores a short fact for later reference.
- `wrapper.get_memories()` lists everything stored so far.
- `wrapper.rag_helper.add_document(text)` adds retrieval context.
- `wrapper.rag_helper.retrieve(query)` returns matching docs that can be inserted into prompts.

## 📜 License
MIT
