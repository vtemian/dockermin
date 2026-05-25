"""Push best LoRA checkpoint to HF as vladtemian/dockermin-qwen7b-lora-v1."""
from __future__ import annotations

import sys

from huggingface_hub import HfApi


def main(ckpt_dir: str) -> None:
    api = HfApi()
    api.create_repo("vladtemian/dockermin-qwen7b-lora-v1", repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=ckpt_dir, repo_id="vladtemian/dockermin-qwen7b-lora-v1",
                      repo_type="model", commit_message="initial LoRA checkpoint")

if __name__ == "__main__":
    main(sys.argv[1])
