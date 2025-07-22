# 🚀 NeuroReasoner Chain-of-Thought Toolkit

A breakthrough open-source suite providing: **always-on** Chain-of-Thought reasoning, **self-consistency** sampling, and **real-time telemetry**, all packaged as a Python wrapper and a futuristic Streamlit GUI.

The interface ships with a sleek dark theme, smooth hover transitions, and a one-click "Copy" button on every code block for effortless sharing of generated scripts.

## 📂 Included Scripts

- `chain_of_thought_wrapper.py` – the core Python module you import into your own scripts.
- `chain_of_thought_gui.py` – a Streamlit app for interactive, no-code usage.

## ⚙️ Installation

1. Install from PyPI:
   ```bash
   pip install cot-toolkit
   ```
   Or clone this repo and install dependencies manually:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure your model checkpoint (e.g. `ayjays132/NeuroReasoner-1-NR-1`) is accessible or change the name in the GUI script.

## 👩‍💻 Importing & Using the Wrapper

Embed step-by-step reasoning directly in your Python code:

```python
from chain_of_thought_wrapper import ChainOfThoughtWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1) Load your tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("ayjays132/NeuroReasoner-1-NR-1")
model     = AutoModelForCausalLM.from_pretrained("ayjays132/NeuroReasoner-1-NR-1")
device    = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2) Wrap with CoT logic
cot = ChainOfThoughtWrapper(model=model, tokenizer=tokenizer, device=device)

# 3) Prepare your prompt
inputs = tokenizer("Why is the sky blue?", return_tensors="pt").to(device)

# 4) Generate step-by-step reasoning
result = cot.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

# 5) Inspect the output
for i, step in enumerate(result["reasoning_steps"][0], 1):
    print(f"Step {i}:", step)
print("Final Answer:", result["final_answers"][0])
```

## 🖥️ Launching the GUI

No code edits needed—just run:

```bash
streamlit run chain_of_thought_gui.py
```

Then open the local URL in your browser. Adjust model name, device, number of chains, sampling parameters, and enter your prompt.

## 🔧 GUI Configuration Options

- **Model**: Hugging Face repo or local path.
- **Device**: cuda or cpu.
- **# Chains**: Number of reasoning samples.
- **Self-Consistency**: Toggle majority-vote across chains.
- **Max New Tokens**: Length of generated reasoning.
- **Temperature**, **Top-k**, **Top-p** & **No-repeat n-gram**: Sampling controls.

## ✨ Polished User Experience

- **Dark theme** with neon accents and subtle gradients.
- **Copy button** on each code block for instant script copying.
- **Responsive layout** that adapts to desktop and mobile screens.
- **Telemetry panel** displaying GPU stats in real time.
- **Download Chat History** option for saving transcripts.
- **Reset Session** button to quickly clear the interface.
- **Premium theme** option for a high-contrast look.
- **Auto-scroll** feature to always show the latest message.
- **Download Last Reasoning** button for saving the most recent answer.
- **Generation duration** displayed for each response.
- **Roman numeral normalization** ensures outputs like "IV" convert to "4".
- **Hyphenated number words** like "twenty-one" convert to digits for cleaner voting.

## ⏳ Example GUI Session

```text
▶ Prompt: What causes rainbows?
▶ Chains: 3, Self-Consistency: on
▶ Sampling: temp 0.7, top-k 50, top-p 0.9
…generating…
▼ Chain 1 ▼
1. Sunlight is composed of multiple colors.
2. Water droplets refract and disperse each color.
3. Observer sees spectrum as arc.
Final Answer: Rainbows form when sunlight refracts and disperses through droplets, separating into colors.
…
```

## 📜 License

Released under the **MIT License**. Free to use, modify, and share—empower everyone with transparent, step-by-step AI reasoning!

