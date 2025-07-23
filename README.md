# 🚀 NeuroReasoner Chain-of-Thought Toolkit

**NeuroReasoner** is a small but capable playground for chain-of-thought prompting, retrieval augmented generation and persistent memories. Everything runs locally with free Hugging Face models and a lightweight Streamlit GUI.

---

## ⚙️ Installation
```bash
pip install cot-toolkit
# or from source
pip install -r requirements.txt
```

## ✨ Features
- **Self‑consistent CoT prompting** with majority voting
- **Streamlit GUI** for chat‑style interaction
- **Tiny retrieval‑augmented generation** helper
- **Persistent memories** that survive between sessions
- **Reference past conversations and memories** when replying
- **Benchmark utilities** to measure CoT vs. plain generation

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
{'cot_duration': 0.143, 'plain_duration': 0.110, 'cot_answer': 'stairs', 'plain_answer': 'factors', 'cot_steps': 0}
```

## 📊 Benchmarking
Run the built‑in benchmark helper to compare CoT vs. plain generation:

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
from cot_toolkit import ChainOfThoughtWrapper, benchmark_prompt

tok = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')
model = AutoModelForCausalLM.from_pretrained('sshleifer/tiny-gpt2')
wrapper = ChainOfThoughtWrapper(model=model, processor=tok, device='cpu')
print(benchmark_prompt(wrapper, 'What causes rainbows?'))
PY
```
This prints a dictionary with durations and answers for each mode.

---

## 📚 Memory & RAG Tips
- `wrapper.remember(text)` stores a short fact for later reference.
- `wrapper.get_memories()` lists everything stored so far.
- `wrapper.forget_all()` clears them.
- `wrapper.rag_helper.add_document(text)` adds retrieval context.
- `wrapper.rag_helper.retrieve(query)` returns matching docs to insert into prompts.
- The wrapper automatically references previous chat history when answering.

## 📜 License
MIT
