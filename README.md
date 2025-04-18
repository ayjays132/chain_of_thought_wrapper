
<style>
/* General Styles */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');

body {
  font-family: 'Montserrat', sans-serif;
  background-color: #121212;
  margin: 0;
  padding: 20px;
  line-height: 1.6;
  color: #e0e0e0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.05);
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  background: linear-gradient(145deg, rgba(20, 35, 55, 0.95), rgba(15, 25, 45, 0.9), rgba(10, 20, 40, 0.85));
  padding: 60px;
  border-radius: 35px;
  box-shadow: 0 25px 70px rgba(0, 0, 0, 0.8), inset 0 0 25px rgba(255, 255, 255, 0.1);
  position: relative;
  overflow: hidden;
  border: 2px solid rgba(100, 200, 255, 0.2);
}
.container::before {
  content: '';
  position: absolute;
  top: -60%;
  left: -60%;
  width: 220%;
  height: 220%;
  background: radial-gradient(circle, rgba(255, 255, 255, 0.2), transparent);
  animation: pulse 14s infinite;
  pointer-events: none;
}
@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.2); }
  100% { transform: scale(1); }
}
.section {
  margin-bottom: 70px;
  position: relative;
}
.section:hover {
  transform: translateY(-7px);
  transition: all 0.5s ease-in-out;
}
.detail {
  padding: 25px;
  margin-bottom: 25px;
  border: 1px solid rgba(120, 160, 220, 0.3);
  border-radius: 20px;
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.1), rgba(100, 140, 200, 0.2));
  box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(255, 255, 255, 0.2);
  transition: all 0.4s ease;
}
.detail:hover {
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.15), rgba(140, 180, 240, 0.25));
  transform: translateY(-7px);
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7), inset 0 0 20px rgba(255, 255, 255, 0.25);
}
.detail-icon {
  font-size: 1.8em;
  color: #63d2ff;
  margin-right: 20px;
}
.detail:hover .detail-icon {
  color: #a2f4ff;
  transform: scale(1.2);
}
ul {
  list-style: none;
  padding: 0;
}
ul li {
  margin: 20px 0;
  padding: 20px;
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.1), rgba(60, 100, 140, 0.25));
  border-radius: 15px;
  box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.3), 0 8px 25px rgba(0, 0, 0, 0.6);
  transition: all 0.4s ease;
}
ul li:hover {
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.15), rgba(80, 120, 160, 0.3));
  transform: translateX(10px);
  box-shadow: 0 15px 30px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(255, 255, 255, 0.2);
}
a {
  color: #63d2ff;
  text-decoration: none;
  font-weight: bold;
  transition: color 0.3s ease, text-shadow 0.3s ease;
}
a:hover {
  color: #a2f4ff;
  text-shadow: 0 0 12px rgba(255, 255, 255, 0.9), 0 0 18px rgba(100, 200, 255, 0.6);
}
h1, h2, h3 {
  text-transform: uppercase;
  color: #e8f0ff;
  text-shadow: 5px 5px 15px rgba(0, 0, 0, 0.9), 0 0 20px rgba(255, 255, 255, 0.6);
  font-weight: 700;
}
</style>

</head>
<body>
<h1>NeuroReasoner Chain-of-Thought Toolkit</h1>
<p>A futuristic open-source toolkit for step-by-step reasoning with stunning telemetry.</p>
<ul>
<li>Always-on Chain-of-Thought Wrapper</li>
<li>ASCII Telemetry Stream</li>
<li>Self-Consistency Sampling</li>
<li>Plug & Play with Hugging Face Transformers</li>
</ul>
<h2>Usage</h2>
<p>Run the inference script:</p>
<pre class="code">python inference_neuroreasoner_1_nr_1_with_cot.py \
--prompt "Why is the sky blue?" \
--self-consistency \
--num-sequences 3 \
--max-new-tokens 200
</pre>
<h2>Contents</h2>
<ul>
<li>chain_of_thought_wrapper.py</li>
<li>inference_neuroreasoner_1_nr_1_with_cot.py</li>
<li>README.md</li>
</ul>
<h2>License</h2>
<p>MIT License — Free to use and modify.</p>
</body>
