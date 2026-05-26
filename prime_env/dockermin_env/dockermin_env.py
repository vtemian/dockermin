"""Dockermin verifiers Environment for prime-rl. Single-turn: prompt -> Dockerfile -> reward."""

from __future__ import annotations

import verifiers as vf
from datasets import load_dataset

from dockermin.reward.dockermin_reward import dockermin_reward
from dockermin.reward.prompts import SYSTEM_PROMPT, USER_TEMPLATE


def load_environment(**_kwargs) -> vf.Environment:
    train_ds = load_dataset("vtemian/dockermin-v0", split="train")
    eval_ds = load_dataset("vtemian/dockermin-v0", split="test")

    def fmt(ex):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(dockerfile=ex["dockerfile"]),
                },
            ],
            "info": {
                "baseline_size": ex["baseline_size"],
                "test_cmd": ex["test_cmd"],
                "expected_substring": ex["expected_substring"],
            },
            "answer": "",  # not used for free-form reward
        }

    def _project(ds):
        return ds.map(fmt, remove_columns=[c for c in ds.column_names if c not in ("prompt", "info", "answer")])

    return vf.SingleTurnEnv(
        dataset=_project(train_ds),
        eval_dataset=_project(eval_ds),
        rubric=vf.Rubric(funcs=[dockermin_reward], weights=[1.0]),
    )
