<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NeuroReasoner Chain‑of‑Thought Toolkit</title>
  <meta name="description" content="Open‑source Chain‑of‑Thought inference wrapper on Hugging Face, with step‑by‑step reasoning and telemetry.">
</head>
<body>

  <h1>🚀 NeuroReasoner Chain‑of‑Thought Toolkit</h1>

  <p>
    An open‑source “always‑on” Chain‑of‑Thought inference wrapper for Hugging Face<br>
    models—delivering step‑by‑step reasoning, self‑consistency sampling, and real‑time telemetry.
  </p>

  <h2>🔑 Key Features</h2>
  <ul>
    <li><strong>Strict CoT Injection</strong>: Always formats reasoning as “Step 1: … Step 2: … Final Answer: …”.</li>
    <li><strong>Self‑Consistency</strong>: Sample multiple reasoning chains &amp; take a majority vote.</li>
    <li><strong>Telemetry Stream</strong>: ASCII‑art panels show GPU/CPU utilization &amp; memory in real time.</li>
    <li><strong>Plug & Play</strong>: Works with any <code>AutoModelForCausalLM</code> checkpoint—local or on Hugging Face.</li>
    <li><strong>Minimal Dependencies</strong>: <code>torch</code>, <code>transformers</code>, <code>streamlit</code> (for GUI), optionally <code>pynvml</code>.</li>
  </ul>

  <h2>⚙️ Installation</h2>
  <ol>
    <li>Clone or unzip this folder.</li>
    <li>Install required packages:
      <pre><code>pip install torch transformers rich streamlit pynvml</code></pre>
    </li>
    <li>Ensure you have access to the model  
      <code>ayjays132/NeuroReasoner-1-NR-1</code>  
      or change the path in the scripts to your own checkpoint.
    </li>
  </ol>

  <h2>💡 Included Scripts</h2>
  <ul>
    <li><code>chain_of_thought_wrapper.py</code> — the core CoT wrapper module you can import.</li>
    <li><code>inference_neuroreasoner_1_nr_1_with_cot.py</code> — CLI inference script with telemetry.</li>
    <li><code>gui_neuroreasoner_cot.py</code> — Streamlit‑based GUI for interactive use.</li>
    <li><code>README.md</code> — this documentation file.</li>
  </ul>

  <h2>👩‍💻 Importing in Your Own Code</h2>
  <p>If you’d rather embed CoT into your Python project, simply:</p>
  <pre><code>from chain_of_thought_wrapper import ChainOfThoughtWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
https://github.com/ayjays132/chain_of_thought_wrapper/blob/main/README.md
# 1) Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("ayjays132/NeuroReasoner-1-NR-1")
model     = AutoModelForCausalLM.from_pretrained("ayjays132/NeuroReasoner-1-NR-1")

# 2) Wrap with CoT logic
cot = ChainOfThoughtWrapper(
    model=model,
    tokenizer=tokenizer,
    device="cuda"  # or "cpu"
)

# 3) Tokenize your prompt
inputs = tokenizer("Why is the sky blue?", return_tensors="pt").to(cot.device)

# 4) Generate step-by-step reasoning
out = cot.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

# 5) Access results
for step in out["reasoning_steps"][0]:
    print("Step:", step)
print("Final Answer:", out["final_answers"][0])
</code></pre>

  <h2>⏳ One‑Line CLI Inference</h2>
  <p>Run the example script in a single command (no code edits needed):</p>
  <pre><code>python inference_neuroreasoner_1_nr_1_with_cot.py --prompt "Why is the sky blue?" --self-consistency --num-sequences 3 --max-new-tokens 200</code></pre>

  <h2>🎨 Launch the GUI</h2>
  <p>For an interactive web app:</p>
  <pre><code>streamlit run gui_neuroreasoner_cot.py</code></pre>

  <h2>🔍 Example CLI Output</h2>
  <pre><code>╔════════ 🚀 GENERATION START 🚀 ════════╗
║ Prompt: Why is the sky blue?           ║
╚════════════════════════════════════════╝
║ 1. Sunlight enters the atmosphere.
║ 2. Molecules scatter blue wavelengths.
║ 3. Our eyes perceive the scattered blue.
Final Answer: The sky looks blue because shorter (blue) light is scattered more strongly by air molecules.
</code></pre>

  <h2>📜 License</h2>
  <p>This toolkit is released under the <strong>MIT License</strong>. Feel free to use, modify, and share!</p>

</body>
</html>
