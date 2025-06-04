<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="description" content="Chain‑of‑Thought wrapper and GUI for Hugging Face causal LMs with step‑by‑step reasoning and telemetry.">
</head>
<body>

  <h1>🚀 NeuroReasoner Chain‑of‑Thought Toolkit</h1>

  <p>
    A breakthrough open‑source suite providing:
    <strong>always‑on</strong> Chain‑of‑Thought reasoning,
    <strong>self‑consistency</strong> sampling, and
    <strong>real‑time telemetry</strong>,
    all packaged as a Python wrapper and a futuristic Streamlit GUI.
  </p>
  <p>
    The interface ships with a sleek dark theme, smooth hover transitions, and a
    one‑click "Copy" button on every code block for effortless sharing of generated
    scripts.
  </p>

  <h2>📂 Included Scripts</h2>
  <ul>
    <li><code>chain_of_thought_wrapper.py</code> – the core Python module you import into your own scripts.</li>
    <li><code>chain_of_thought_gui.py</code> – a Streamlit app for interactive, no‑code usage.</li>
  </ul>

  <h2>⚙️ Installation</h2>
  <ol>
    <li>Clone or unzip this folder.</li>
    <li>Install required packages:
      <pre><code>pip install torch transformers streamlit pynvml</code></pre>
    </li>
    <li>Ensure your model checkpoint (e.g. <code>ayjays132/NeuroReasoner‑1‑NR‑1</code>) is accessible
      or change the name in the GUI script.
    </li>
  </ol>

  <h2>👩‍💻 Importing &amp; Using the Wrapper</h2>
  <p>Embed step‑by‑step reasoning directly in your Python code:</p>
  <pre><code>from chain_of_thought_wrapper import ChainOfThoughtWrapper
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

# 4) Generate step‑by‑step reasoning
result = cot.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

# 5) Inspect the output
for i, step in enumerate(result["reasoning_steps"][0], 1):
    print(f"Step {i}:", step)
print("Final Answer:", result["final_answers"][0])
</code></pre>

  <h2>🖥️ Launching the GUI</h2>
  <p>No code edits needed—just run:</p>
  <pre><code>streamlit run chain_of_thought_gui.py</code></pre>
  <p>Then open the local URL in your browser. Adjust model name, device, number of chains, sampling parameters, and enter your prompt.</p>

  <h2>🔧 GUI Configuration Options</h2>
  <ul>
    <li><strong>Model</strong>: Hugging Face repo or local path.</li>
    <li><strong>Device</strong>: cuda or cpu.</li>
    <li><strong># Chains</strong>: Number of reasoning samples.</li>
    <li><strong>Self‑Consistency</strong>: Toggle majority‑vote across chains.</li>
    <li><strong>Max New Tokens</strong>: Length of generated reasoning.</li>
    <li><strong>Temperature</strong>, <strong>Top‑k</strong>, <strong>Top‑p</strong> &amp; <strong>No‑repeat n‑gram</strong>: Sampling controls.</li>
  </ul>

  <h2>✨ Polished User Experience</h2>
  <ul>
    <li><strong>Dark theme</strong> with neon accents and subtle gradients.</li>
    <li><strong>Copy button</strong> on each code block for instant script copying.</li>
    <li><strong>Responsive layout</strong> that adapts to desktop and mobile screens.</li>
    <li><strong>Telemetry panel</strong> displaying GPU stats in real time.</li>
  </ul>

  <h2>⏳ Example GUI Session</h2>
  <pre><code>▶ Prompt: What causes rainbows?
▶ Chains: 3, Self‑Consistency: on
▶ Sampling: temp 0.7, top‑k 50, top‑p 0.9
…generating…
▼ Chain 1 ▼
1. Sunlight is composed of multiple colors.
2. Water droplets refract and disperse each color.
3. Observer sees spectrum as arc.
Final Answer: Rainbows form when sunlight refracts and disperses through droplets, separating into colors.
…</code></pre>

  <h2>📜 License</h2>
  <p>Released under the <strong>MIT License</strong>. Free to use, modify, and share—empower everyone with transparent, step‑by‑step AI reasoning!</p>

</body>
</html>
