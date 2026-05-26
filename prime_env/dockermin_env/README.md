# dockermin-env

Verifiers Environment package wrapping the Dockermin dataset + reward for prime-rl.

This package exposes a single-turn environment named `dockermin` that prime-rl can
discover via the `verifiers.environments` entry-point group. It loads the
`vtemian/dockermin-v0` HuggingFace dataset, formats each row with the
Dockermin prompt template, and grades rollouts with `dockermin.reward.dockermin_reward`.

## Layout

```
prime_env/dockermin_env/
  __init__.py
  dockermin_env.py    # load_environment() entry point
  pyproject.toml      # registers verifiers.environments -> dockermin
```

## Install (on the training pod)

The parent `dockermin` package must be installed first, since this environment
imports `dockermin.reward.prompts` and `dockermin.reward.dockermin_reward`.

```bash
# from repo root
pip install -e .

# then install this environment package
cd prime_env/dockermin_env
uv pip install -e .

# make it discoverable by prime-rl / verifiers
vf-install dockermin
```

After install, prime-rl configs can reference it as:

```toml
[[env]]
id = "dockermin"
```

## Dependencies

- `verifiers` - provides `SingleTurnEnv`, `Rubric`, `Environment` base classes
- `datasets` - loads `vtemian/dockermin-v0` from HF Hub
- `dockermin` - the parent package (prompts + reward), install from repo root with `pip install -e ../..`
