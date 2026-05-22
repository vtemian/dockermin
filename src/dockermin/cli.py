"""dockermin <dockerfile_path> --test '<cmd>' --expect <substring> --model <hf_id>"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from .reward.prompts import format_messages, extract_dockerfile

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dockerfile", type=Path)
    p.add_argument("--test", required=True)
    p.add_argument("--expect", required=True)
    p.add_argument("--model", default="vladtemian/dockermin-qwen7b-lora-v1")
    args = p.parse_args()
    df = args.dockerfile.read_text()
    msgs = format_messages(df, args.test.split(), args.expect)
    # Inference: load base Qwen + LoRA via vllm or HF transformers. For CLI ship the simpler HF path.
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", torch_dtype="bfloat16", device_map="auto")
    model = PeftModel.from_pretrained(base, args.model)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=1024, temperature=0.2, do_sample=True)
    text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    new_df = extract_dockerfile(text)
    if new_df is None:
        print("ERROR: model output had no fenced dockerfile block", file=sys.stderr)
        return 1
    print(new_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
