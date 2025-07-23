# 🚀 NeuroReasoner Chain-of-Thought Toolkit

**NeuroReasoner** wraps any Hugging Face model with always-on chain-of-thought (CoT) prompting, a tiny RAG helper and persistent memory utilities. A lightweight Streamlit GUI is provided for quick experiments.

---

## ⚙️ Installation
```bash
pip install cot-toolkit
# or from source
pip install -r requirements.txt
```

## ✨ Features
- **Self-consistent CoT prompting** out of the box
- **Streamlit GUI** for chat-style interaction
- **Simple RAG helper** for storing and retrieving facts
- **Saved memories** that persist between runs
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
res = wrapper.generate("Who am I and what is the largest planet?", generation_params={"max_new_tokens":16})
print(res["final_answers"][0])
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

### 📈 Latest Benchmark Example
```
{'cot_duration': 0.13, 'plain_duration': 0.16,
 'cot_answer': 'stairs stairs ...',
 'plain_answer': 'factors factors ...', 'cot_steps': 0}
```

---

## 📚 Memory & RAG Tips
- `wrapper.remember(text)` stores a short fact for later reference.
- `wrapper.get_memories()` lists everything stored so far.
- `wrapper.rag_helper.add_document(text)` adds retrieval context.
- `wrapper.rag_helper.retrieve(query)` returns matching docs to insert into prompts.

## 📜 License
MIT
