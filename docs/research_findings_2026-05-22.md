# Research findings, 2026-05-22

Three open verifications resolved via subagent reading prime-rl + verifiers source. These were flagged as launch-blocking unknowns before the Saturday pod session.

## Q1: prime-rl PR #1392 (auto_setup_lora NCCL crash)

**MERGED 2025-12-07.** Merge SHA `91182b7d647285a3e9e32f7959fdc3ff044d9330`.

Action Saturday: pull latest prime-rl main, or pin >= that SHA. Verify with `git log --oneline | head -5` in the prime-rl clone.

**Caveat that survived the PR:** `validate_lora_broadcast` in `packages/prime-rl-configs/src/prime_rl/configs/trainer.py:641-644` still raises "NCCL weight broadcast does not support LoRA yet." when both are set. The PR fixed the auto-configure bug; the underlying NCCL+LoRA combination is still unsupported. Use filesystem broadcast.

In our pilot/full TOMLs:
```toml
[trainer.weight_broadcast]
type = "filesystem"
```

## Q2: prime-rl TOML schema (model + LoRA + wandb)

Reference: `prime-rl/examples/alphabet_sort/rl.toml`.

Correct shape verbatim:
```toml
[model]
name = "Qwen/Qwen3-4B-Instruct-2507"

[trainer.model.lora]
rank = 32
alpha = 64
```

Key facts:
- `[model]` is a **table**, key is `name` (string). NOT a flat `model = "..."`.
- LoRA lives under `[trainer.model.lora]`. NOT under `[model.experimental]` (which does not exist).
- `max_lora_rank` lives under `[inference]` and is **auto-set** by `auto_setup_lora` from `trainer.model.lora.rank` (see `inference.py:222` and `rl.py:460`). Do not set it manually.
- `[trainer.experimental]` and `[inference.experimental]` exist; `[model.experimental]` does not.
- `[wandb]` with `project / entity / name / group / tags / offline` is valid per `shared.py:197-216`.

Our `configs/dockermin_{pilot,full}.toml` now match this shape.

## Q3: verifiers entry-point group is decorative

The `[project.entry-points."verifiers.environments"]` block in `prime_env/dockermin_env/pyproject.toml` is **harmless but not load-bearing**. Only 1 of 39 in-tree verifiers environments declares one, and it is unused at runtime.

What actually matters: verifiers resolves an `env_id` to a Python module via `package_module_name(env_id)`:
- `package_module_name("dockermin")` -> `"dockermin"` (mismatch with our `dockermin_env` module)
- `package_module_name("dockermin-env")` -> `"dockermin_env"` (matches)

Fix already applied in `configs/dockermin_{pilot,full}.toml`: `[[env]] id = "dockermin-env"`.

`vf-install` is a thin wrapper around `uv pip install -e <path>` (see `verifiers/utils/install_utils.py:211`). No registry lookup. Just install the package and verifiers finds it by importable module name.

## Q4: dockerfile.GoParseError symbol is stable in v3.3.1

Defined in `asottile/dockerfile/pylib/support.c:52` as `PyErr_NewException("dockerfile.GoParseError", PyExc_ValueError, NULL)`. Subclass of `ValueError`. Documented in README.md:45.

**The repo is archived upstream** (`gh api repos/asottile/dockerfile` returns `archived: True`). No deprecation banner in the README, but archive status is the canonical "no future updates" signal.

Action: safe to use against 3.3.1 as pinned. Plan a longer-term migration to a pure-Python parser if the project survives past v0.

## Action items for Saturday morning

- [ ] `git pull origin main` in your prime-rl clone, confirm PR #1392 merge SHA `91182b7` is in history
- [ ] `prime-rl/examples/alphabet_sort/rl.toml` - diff against ours to confirm we haven't missed any required keys
- [ ] First call after pod is up: `python -c "from prime_rl.configs import RLConfig; RLConfig.from_toml('configs/dockermin_pilot.toml')"` to validate config parses against prime-rl's schema
- [ ] `vf-install <prime_env/dockermin_env>` on the pod, then `python -c "import dockermin_env; print(dockermin_env.dockermin_env.load_environment.__module__)"` to confirm the resolution path
