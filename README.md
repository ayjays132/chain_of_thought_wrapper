# 🚀 NeuroReasoner Chain-of-Thought Toolkit

**NeuroReasoner** offers a compact yet capable playground for chain-of-thought prompting, retrieval‑augmented generation (RAG), and lightweight persistent memories. Everything runs locally using free Hugging Face models and a minimal Streamlit GUI.

---

## ⚙️ Installation
```bash
pip install cot-toolkit
# or from source
pip install -r requirements.txt
```
The toolkit is available on PyPI under the name `cot-toolkit`.

## ✨ Features
- **Self‑consistent CoT prompting** with majority voting
- **Streamlit GUI** for an interactive chat experience
- **Mini RAG helper** to inject custom context
- **Persistent memories** that survive between sessions
- **Reference prior conversations** when generating
- **Record history** management for transcripts or notes
- **Benchmark utilities** for CoT vs. plain generation

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
wrapper.add_record("Session started")
wrapper.rag_helper.add_document("Jupiter is the largest planet.")
res = wrapper.generate("Who am I and what is the largest planet?", generation_params={"max_new_tokens":16})
print(res["final_answers"][0])
print(wrapper.get_records())
```

## 🖥️ Launch the GUI
```bash
streamlit run chain_of_thought_gui.py
```
Use the sidebar to select a model and toggle self‑consistency.

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
{'cot_duration': 0.2859, 'plain_duration': 0.1203,
 'cot_answer': 'stairs',
 'plain_answer': 'stairs', 'cot_steps': 0}
```
*(measured with `sshleifer/tiny-gpt2` on CPU)*

## 📊 Benchmarking
Run the built-in benchmark helper to compare CoT vs. plain generation:
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
- `wrapper.add_record(text)` saves a transcript snippet to the record history.
- `wrapper.get_records()` retrieves saved records.
- `wrapper.clear_records()` wipes all record history.
- `wrapper.rag_helper.add_document(text)` adds retrieval context.
- `wrapper.rag_helper.retrieve(query)` returns matching docs to insert into prompts.
- The wrapper automatically references previous chat history when answering.

## 🚀 Releasing to PyPI
This project uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) via GitHub Actions.
Creating a GitHub release triggers the `python-app.yml` workflow which runs the tests,
builds the package and uploads it to PyPI.
Make sure the repository has a `pypi` environment configured as a Trusted Publisher.


## 📜 License
MIT
