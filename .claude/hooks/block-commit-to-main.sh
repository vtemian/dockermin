#!/usr/bin/env bash
# PreToolUse(Bash) guard: refuse to let a git commit land on main.
# Project rule (CLAUDE.md): never commit directly to main; branch per change.
# The git pre-commit hook already runs the quality gate, so this guard only
# closes the gap pre-commit hooks ignore — the current branch.

set -euo pipefail

input=$(cat)
command=$(printf '%s' "$input" | python3 -c "import sys, json; print(json.load(sys.stdin).get('tool_input', {}).get('command', ''))")

case "$command" in
  *"git commit"*) ;;
  *) exit 0 ;;
esac

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
if [ "$branch" = "main" ]; then
  echo "Blocked: committing directly to main violates the project rule. Branch first: git checkout -b feat/<description>" >&2
  exit 2
fi
exit 0
