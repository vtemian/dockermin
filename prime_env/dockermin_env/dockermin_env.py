"""Dockermin verifiers Environment for prime-rl. Single-turn: prompt -> Dockerfile -> reward."""

from __future__ import annotations

import os

import verifiers as vf
from datasets import load_dataset

from dockermin.dataset.annotate import parse_gate
from dockermin.reward.dockermin_reward import dockermin_reward
from dockermin.reward.prompts import SYSTEM_PROMPT, USER_TEMPLATE

DATASET_ID = os.environ.get("DOCKERMIN_DATASET_ID", "vtemian/dockermin-v1")
# v3 default: vtemian/dockermin-v1 (198 train, 37 frozen-from-v0 test, 14 unique
# bases). Override via DOCKERMIN_DATASET_ID env var (e.g. for a v0 baseline run).


def load_environment(**_kwargs) -> vf.Environment:
    train_ds = load_dataset(DATASET_ID, split="train")
    eval_ds = load_dataset(DATASET_ID, split="test")

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
                # baseline_command_count caps the pad-the-Dockerfile reward hack:
                # the reward function uses min(rewrite.cmd_count, baseline+2) as
                # the effective cmd count for partial credit on failure rungs.
                "baseline_command_count": parse_gate(ex["dockerfile"]).command_count,
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
