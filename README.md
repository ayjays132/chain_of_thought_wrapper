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
  models, delivering step‑by‑step reasoning, self‑consistency sampling, and real‑time telemetry.
</p>

<h2>🔑 Key Features</h2>
<ul>
  <li><strong>Strict CoT Injection</strong>: Forces “Step 1: … Step 2: … Final Answer:” formatting.</li>
  <li><strong>Self‑Consistency</strong>: Sample multiple reasoning chains (majority vote).</li>
  <li><strong>Telemetry Stream</strong>: ASCII‑art panels showing GPU/CPU and memory usage.</li>
  <li><strong>Plug & Play</strong>: Works with any `AutoModelForCausalLM` checkpoint.</li>
  <li><strong>Minimal Dependencies</strong>: Just <code>torch</code>, <code>transformers</code>, <code>rich</code>, optionally <code>pynvml</code>.</li>
</ul>

<h2>⚙️ Installation</h2>
<ol>
  <li>Clone or unzip this directory.</li>
  <li>Install Python dependencies:
    <pre><code>pip install torch transformers rich pynvml</code></pre>
  </li>
  <li>Ensure your model checkpoint is available as 
    <code>ayjays132/NeuroReasoner-1-NR-1</code> or modify the script accordingly.</li>
</ol>

<h2>💡 Files</h2>
<ul>
  <li><code>chain_of_thought_wrapper.py</code> — the CoT wrapper module</li>
  <li><code>inference_neuroreasoner_1_nr_1_with_cot.py</code> — example inference script with telemetry</li>
  <li><code>README.md</code> — this file</li>
</ul>

<h2>⏳ One‑Line Usage</h2>
<p>Run inference with a single command:</p>
<pre><code>python inference_neuroreasoner_1_nr_1_with_cot.py --prompt "Why is the sky blue?" --self-consistency --num-sequences 3 --max-new-tokens 200</code></pre>

<h2>🔍 Example Output</h2>
<pre><code>╔════ 🚀 GENERATION START 🚀 ════╗
║ Prompt: Why is the sky blue?    ║
╚════════════════════════════════╝
║ 1. Sunlight enters atmosphere.
║ 2. Air molecules scatter blue.
║ 3. Our eyes perceive scattered blue.
Final Answer: The sky appears blue because shorter (blue) wavelengths scatter more than longer ones.
</code></pre>

<h2>⚙️ Optional Arguments</h2>
<ul>
  <li><code>--temperature</code> (float): Sampling temperature, default <code>0.7</code>.</li>
  <li><code>--top-k</code> (int): Top‑k sampling, default <code>50</code>.</li>
  <li><code>--top-p</code> (float): Nucleus sampling, default <code>0.9</code>.</li>
  <li><code>--no-repeat-ngram-size</code> (int): Prevent repeat n‑grams, default <code>3</code>.</li>
  <li><code>--device</code> (str): CUDA or CPU device, default autodetected.</li>
</ul>

<h2>📜 License</h2>
<p>This project is released under the <strong>MIT License</strong>. Feel free to use, modify, and share!</p>

</body>
</html>
