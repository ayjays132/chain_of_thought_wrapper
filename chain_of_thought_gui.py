#!/usr/bin/env python3
"""
NeuroReasoner 1 Chain‑of‑Thought GUI
-------------------------------------------------------------
A futuristic, user‑friendly Streamlit app for step‑by‑step reasoning
using any Hugging Face causal LM.

Features:
 • Load any model by repo name or local path
 • Full control of generation params (Temp, top‑k/p, etc.)
 • Self‑Consistency sampling
 • ASCII telemetry panels
 • Progress indicators and collapsible reasoning details
"""
import os
import time
import torch
import pynvml
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from chain_of_thought_wrapper import ChainOfThoughtWrapper

# Initialize GPU telemetry
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

@st.cache_data(show_spinner=False)
def get_telemetry():
    if not GPU_AVAILABLE or not torch.cuda.is_available():
        return "[No GPU telemetry]"
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return f"GPU: {util.gpu}% | Mem: {mem.used//1024**2}/{mem.total//1024**2} MB"

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
model_name = st.sidebar.text_input(
    "Model (HuggingFace repo or local path)", value="ayjays132/NeuroReasoner-1-NR-1"
)
device = st.sidebar.selectbox("Device", options=["cuda" if torch.cuda.is_available() else "cpu", "cpu"] )
num_sequences = st.sidebar.slider("# Chains", min_value=1, max_value=10, value=3)
self_consistency = st.sidebar.checkbox("Self‑Consistency", value=False)
max_new_tokens = st.sidebar.slider("Max New Tokens", 50, 1024, 256)
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)
top_k = st.sidebar.slider("Top-k", 0, 200, 50)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9)
no_repeat_ngram = st.sidebar.slider("No‑repeat ngram", 0, 10, 3)

# Main interface
st.markdown("# 🌀 NeuroReasoner CoT GUI")
col1, col2 = st.columns([3,1])
with col1:
    prompt = st.text_area("🚀 Enter your prompt", value="Explain why the sky is blue.", height=120)
with col2:
    st.metric("Telemetry", get_telemetry())

if st.button("🪄 Generate Reasoning", type="primary"):
    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()
    # Load model & tokenizer
    try:
        with st.spinner("🌐 Loading model and tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(device)
        st.success("✅ Model loaded.")
    except Exception as e:
        st.error(f"❌ Load error: {e}")
        st.stop()
    # Setup CoT
    cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=(num_sequences if self_consistency else 1),
        no_repeat_ngram_size=no_repeat_ngram,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    cot = ChainOfThoughtWrapper(
        model=model,
        tokenizer=tokenizer,
        generation_config=cfg,
        device=device,
        self_consistency=self_consistency,
        consistency_rounds=(num_sequences if self_consistency else 1)
    )
    # Tokenize & generate
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    start = time.time()
    output = cot.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        num_return_sequences=(num_sequences if self_consistency else 1)
    )
    elapsed = time.time() - start
    st.success(f"✨ Done in {elapsed:.2f}s")
    # Display results
    for idx, (full, steps, ans) in enumerate(zip(output['full_texts'], output['reasoning_steps'], output['final_answers']), 1):
        with st.expander(f"Chain {idx}"):
            st.text_area("Full Text", value=full, height=200)
            if steps:
                st.write("**Steps:**")
                for i, s in enumerate(steps, 1): st.write(f"{i}. {s}")
            else:
                st.warning("No parsed steps.")
            st.markdown(f"**Final Answer:** {ans}")
    st.markdown("---")
    st.write(f"Telemetry: {get_telemetry()}")

# Footer
st.markdown("<sub>Built for a futuristic, seamless reasoning experience.</sub>", unsafe_allow_html=True)
