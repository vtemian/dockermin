"""Dockermin verifiers Environment for prime-rl. Single-turn: prompt -> Dockerfile -> reward."""
from __future__ import annotations
import verifiers as vf
from datasets import load_dataset
from dockermin.reward.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from dockermin.reward.dockermin_reward import dockermin_reward

def load_environment(**kwargs) -> vf.Environment:
    ds = load_dataset("vladtemian/dockermin-v0", split="train")
    def fmt(ex):
        return {
            "prompt": [
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":USER_TEMPLATE.format(
                    dockerfile=ex["dockerfile"],
                    test_cmd=" ".join(ex["test_cmd"]),
                    expected=ex["expected_substring"],
                )},
            ],
            "info": {
                "baseline_size": ex["baseline_size"],
                "test_cmd": ex["test_cmd"],
                "expected_substring": ex["expected_substring"],
            },
            "answer": "",  # not used for free-form reward
        }
    ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c not in ("prompt","info","answer")])
    return vf.SingleTurnEnv(
        dataset=ds,
        rubric=vf.Rubric(funcs=[dockermin_reward], weights=[1.0]),
    )
