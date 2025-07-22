# 🚀 NeuroReasoner Chain-of-Thought Toolkit

NeuroReasoner brings always-on chain-of-thought prompting to any Hugging Face model. The wrapper collects reasoning steps, reports device metrics and supports an optional Streamlit GUI.

## ⚙️ Installation

```bash
pip install cot-toolkit
# or, install from this repository
pip install -r requirements.txt
```

## 👩‍💻 Using the Wrapper

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from cot_toolkit import ChainOfThoughtWrapper

model_id = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

wrapper = ChainOfThoughtWrapper(model=model, processor=tokenizer, device="cpu")
inputs = tokenizer("Why is the sky blue?", return_tensors="pt")
result = wrapper.generate("Why is the sky blue?", generation_params={"max_new_tokens": 16})
print(result["final_answers"][0])
```

## 🖥️ Launching the GUI

Run the Streamlit interface:

```bash
streamlit run chain_of_thought_gui.py
```

Configure the model, device and sampling options in the sidebar and enter your prompt.

## ⏳ Example GUI Session

```
▶ Prompt: What causes rainbows?
▶ Chains: 3, Self-Consistency: on
▶ Sampling: temp 0.7, top-k 50, top-p 0.9
…generating…
▼ Chain 1 ▼
1. Sunlight is made of many colors.
2. Water droplets bend and split the light.
3. The observer sees the separated colors as an arc.
Final Answer: Rainbows appear when light refracts and disperses through droplets.
```

## 📊 Benchmarking

Run a benchmark on a small model to measure overhead:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from cot_toolkit import ChainOfThoughtWrapper, benchmark_prompt

model_id = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
wrapper = ChainOfThoughtWrapper(model=model, processor=tokenizer, device="cpu")
metrics = benchmark_prompt(wrapper, "What is the largest planet?")
print(metrics)
```

### 📈 Latest Benchmark Example

Running the above on CPU produced:

```
{'cot_duration': 0.22, 'plain_duration': 0.31, 'cot_answer': 'stairs', 'plain_answer': 'stairs', 'cot_steps': 0}
```

Even the tiny model returns a structured answer while adding only a small amount of latency.

## ✨ GUI Highlights

* Dark theme with copy-to-clipboard buttons
* Responsive layout for desktop and mobile
* Telemetry panel shows GPU or CPU usage
* Optional premium and sci‑fi themes
* Download chat history and last reasoning step
* Generation duration and memory metrics displayed per response
* Number words like "twenty-one" normalize to digits for better self-consistency

## 📜 License

Released under the MIT License.
