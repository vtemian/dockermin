"""dockermin <dockerfile_path> --test '<cmd>' --expect <substring> --model <hf_id>"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from .reward.prompts import extract_dockerfile, format_messages

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def _generate(msgs: list[dict[str, str]], model_id: str) -> str:
    """Load base Qwen + LoRA via HF transformers and decode one completion.

    For the CLI we ship the simpler HF path rather than vllm.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="bfloat16",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, model_id)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=1024, temperature=0.2, do_sample=True)
    text: str = tok.decode(out[0][enc.input_ids.shape[1] :], skip_special_tokens=True)
    return text


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dockerfile", type=Path)
    p.add_argument("--test", required=True)
    p.add_argument("--expect", required=True)
    p.add_argument("--model", default="vladtemian/dockermin-qwen7b-lora-v1")
    args = p.parse_args()
    df = args.dockerfile.read_text()
    msgs = format_messages(df, shlex.split(args.test), args.expect)
    text = _generate(msgs, args.model)
    new_df = extract_dockerfile(text)
    if new_df is None:
        print("ERROR: model output had no fenced dockerfile block", file=sys.stderr)
        return 1
    print(new_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
