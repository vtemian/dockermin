# scripts/smoke_lora_hotswap.py
"""Verify vLLM 0.7.3 LoRA hotswap on Qwen 2.5 Coder 7B before committing to GRPO pipeline."""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
PROMPT = "Write a Python function that returns the nth Fibonacci number."

def train_tiny_lora(seed: int, out_dir: Path) -> None:
    """Train a 50-step LoRA on a single random batch so adapters differ."""
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    text = "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n" * 4
    enc = tok(text, return_tensors="pt").to("cuda")
    for _ in range(50):
        out = model(**enc, labels=enc["input_ids"])
        out.loss.backward(); opt.step(); opt.zero_grad()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    del model; torch.cuda.empty_cache()

def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="lora_smoke_"))
    a_dir, b_dir = workdir / "a", workdir / "b"
    print(f"Training LoRA A in {a_dir}")
    train_tiny_lora(seed=1, out_dir=a_dir)
    print(f"Training LoRA B in {b_dir}")
    train_tiny_lora(seed=2, out_dir=b_dir)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=BASE, enable_lora=True, max_loras=4, max_lora_rank=32,
              dtype="bfloat16", gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=128, seed=0)

    t0 = time.perf_counter()
    base_out = llm.generate([PROMPT], sp)[0].outputs[0].text
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    a1 = llm.generate([PROMPT], sp, lora_request=LoRARequest("a", 1, str(a_dir)))[0].outputs[0].text
    t_a1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    b1 = llm.generate([PROMPT], sp, lora_request=LoRARequest("b", 2, str(b_dir)))[0].outputs[0].text
    t_b1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    a2 = llm.generate([PROMPT], sp, lora_request=LoRARequest("a", 1, str(a_dir)))[0].outputs[0].text
    t_a2 = time.perf_counter() - t0

    print(f"latency base={t_base:.3f}s a1={t_a1:.3f}s b1={t_b1:.3f}s a2={t_a2:.3f}s")
    if base_out == a1:
        print("FAIL: base output == LoRA A output (adapter not applied)"); return 1
    if a1 == b1:
        print("FAIL: LoRA A output == LoRA B output (swap not effective)"); return 1
    if a1 != a2:
        print("FAIL: same LoRA + greedy + same seed produced different output (nondeterminism)"); return 1
    print("PASS: base != A != B, A reproducible after swap")
    return 0

if __name__ == "__main__":
    sys.exit(main())
