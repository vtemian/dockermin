"""Dockermin verifiers Environment for prime-rl.

verifiers resolves an env id to a module and reads ``load_environment`` off the
imported package, so the entrypoint must be exposed at the package root (not only
in the same-named submodule).
"""

from dockermin_env.dockermin_env import load_environment

__all__ = ["load_environment"]
