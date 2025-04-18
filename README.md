<!DOCTYPE html>
<html>
<head>
<style>
body { background: #000; color: #0f0; font-family: monospace; margin: 2em; }
h1 { text-shadow: 0 0 10px #0f0; }
.code { background: #111; padding: 1em; border-radius: 5px; overflow-x: auto; }
ul { list-style: none; padding: 0; }
li:before { content: '🚀 '; }
</style>
</head>
<body>
<h1>NeuroReasoner Chain-of-Thought Toolkit</h1>
<p>A futuristic open-source toolkit for step-by-step reasoning with stunning telemetry.</p>
<ul>
<li>Always-on Chain-of-Thought Wrapper</li>
<li>ASCII Telemetry Stream</li>
<li>Self-Consistency Sampling</li>
<li>Plug & Play with Hugging Face Transformers</li>
</ul>
<h2>Usage</h2>
<p>Run the inference script:</p>
<pre class="code">python inference_neuroreasoner_1_nr_1_with_cot.py \
  --prompt "Why is the sky blue?" \
  --self-consistency \
  --num-sequences 3 \
  --max-new-tokens 200
</pre>
<h2>Contents</h2>
<p>
- chain_of_thought_wrapper.py<br>
- inference_neuroreasoner_1_nr_1_with_cot.py<br>
- README.md
</p>
<h2>License</h2>
<p>MIT License — Free to use and modify.</p>
</body>
</html>
