#!/usr/bin/env python3
"""
NeuroReasoner 1 Inference with Chain-of-Thought and Chronometric Telemetry
----------------------------------------------------------------------------

This script demonstrates SOTA, advanced inference with step-by-step reasoning,
self-consistency, and a fun, ASCII telemetry stream capturing GPU and memory usage
in real-time, using only the open-sourced ChainOfThoughtWrapper.
"""
import argparse
import logging
import sys
import time
import torch
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from chain_of_thought_wrapper import ChainOfThoughtWrapper

# Try to get GPU telemetry
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def banner_line(msg: str, width: int = 80) -> str:
    pad = max((width - len(msg) - 2) // 2, 0)
    return "╔" + "═" * pad + f" {msg} " + "═" * (width - len(msg) - 2 - pad) + "╗"

def footer_line(width: int = 80) -> str:
    return "╚" + "═" * (width - 2) + "╝"

def telemetry_stream() -> str:
    if not GPU_AVAILABLE or not torch.cuda.is_available():
        return "[No GPU telemetry]"
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
    usedMB = mem.used // 1024**2
    totMB  = mem.total // 1024**2
    return f"GPU Util: {util.gpu}% | Mem: {usedMB}/{totMB} MB"
def parse_args():
    p = argparse.ArgumentParser(
        description="Inference with NeuroReasoner 1 + ChainOfThought + Telemetry"
    )
    p.add_argument("--prompt", type=str, default="Explain why the sky is blue.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-new-tokens", type=int,   default=256)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--top-k",          type=int,   default=50)
    p.add_argument("--top-p",          type=float, default=0.9)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--num-sequences",  type=int,   default=3,
                   help="How many CoT sequences to sample (self-consistency).")
    p.add_argument("--self-consistency", action="store_true",
                   help="Enable majority-vote across sampled chains.")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.info

    # --- Telemetry banner ---
    log(banner_line("📡 CHRONOMETRIC TELEMETRY STREAM 📡"))
    log(f"║   Start: {time.strftime('%H:%M:%S')} on {args.device}")
    log(f"║   {telemetry_stream()}")
    log(footer_line())

    # --- Load model & tokenizer directly ---
    try:
        tokenizer = AutoTokenizer.from_pretrained("ayjays132/NeuroReasoner-1-NR-1")
        model = AutoModelForCausalLM.from_pretrained("ayjays132/NeuroReasoner-1-NR-1")
        model.to(args.device)
        log("✅ Model & tokenizer loaded.")
    except Exception as e:
        log(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        log("[Warning] pad_token set to eos_token")

    # --- Chain‑of‑Thought wrapper setup ---
    ensemble = args.num_sequences if args.self_consistency else 1
    cot = ChainOfThoughtWrapper(
        model=model,
        tokenizer=tokenizer,
        generation_config=GenerationConfig(
            max_new_tokens         = args.max_new_tokens,
            temperature            = args.temperature,
            top_k                  = args.top_k,
            top_p                  = args.top_p,
            do_sample              = True,
            num_return_sequences   = ensemble,
            no_repeat_ngram_size   = args.no_repeat_ngram_size,
            eos_token_id           = tokenizer.eos_token_id,
            pad_token_id           = tokenizer.pad_token_id
        ),
        device           = args.device,
        self_consistency = args.self_consistency,
        consistency_rounds=ensemble
    )
    log("✅ ChainOfThoughtWrapper initialized.")

    # --- Tokenize user prompt ---
    enc = tokenizer(
        args.prompt,
        return_tensors   = 'pt',
        padding          = True,
        truncation       = True,
        max_length       = 512
    ).to(args.device)

    # --- Generation start ---
    log(banner_line("🚀 GENERATION START 🚀"))
    log(f"║ Prompt: {args.prompt}")
    log(footer_line())

    out = cot.generate(
        input_ids            = enc['input_ids'],
        attention_mask       = enc['attention_mask'],
        num_return_sequences = ensemble
    )

    # --- Telemetry mid-run ---
    log(banner_line("📡 TELEMETRY UPDATE 📡"))
    log(f"║   {telemetry_stream()}")
    log(footer_line())

    # --- Display each sequence ---
    for i, (full, steps, ans) in enumerate(
            zip(out['full_texts'], out['reasoning_steps'], out['final_answers']), 1):
        log(banner_line(f"===== Sequence {i} ====="))
        log("║ Full Text:")
        for line in full.splitlines():
            log(f"║   {line}")
        log("║")
        if steps:
            log("║ Reasoning Steps:")
            for idx, st in enumerate(steps, 1):
                log(f"║   {idx}. {st}")
        else:
            log("║ [No distinct reasoning parsed]")
        log(f"║ Final Answer: {ans}")
        log(footer_line())

    # --- Final telemetry ---
    log(banner_line("📡 FINAL TELEMETRY STREAM 📡"))
    log(f"║ Completed: {time.strftime('%H:%M:%S')}   {telemetry_stream()}")
    log(footer_line())


if __name__ == "__main__":
    main()
