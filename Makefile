# dockermin -- ergonomic developer commands
# Plan: docs/plans/2026-05-22-dockermin-implementation.md
#
# All targets shell out to the project venv at .venv/ so they work without
# `source .venv/bin/activate`. Run `make` (no args) for a help listing.

PY := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
MYPY := .venv/bin/mypy

.DEFAULT_GOAL := help

.PHONY: help venv install install-light test test-pure lint fmt typecheck quality scrape annotate smoke-reward leaderboard clean

help: ## Show this help (targets and their descriptions)
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create .venv with python3.11 and install dev deps
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

install: ## pip install -e .[dev] into existing .venv
	$(PIP) install -e .[dev]

install-light: ## Install minimal runtime deps (no torch/vllm) for dev laptop
	$(PIP) install pytest dockerfile docker tqdm tenacity huggingface_hub anthropic openai datasets
	$(PIP) install --no-deps -e .

test: ## Run full pytest suite
	$(PYTEST) tests/ -v

test-pure: ## Run pytest excluding docker-daemon-gated tests
	$(PYTEST) tests/ -v -m "not docker"

lint: ## Run ruff check (with --fix) on src tests scripts prime_env
	$(RUFF) check --fix src tests scripts prime_env

fmt: ## Run ruff format on src tests scripts prime_env
	$(RUFF) format src tests scripts prime_env

typecheck: ## Run mypy strict on src tests
	$(MYPY) src tests

quality: fmt lint typecheck test-pure ## Run all quality gates (format, lint, typecheck, pure tests)

scrape: ## Run the dataset scrape script
	$(PY) scripts/run_scrape.py

annotate: ## Run the annotate pipeline
	$(PY) scripts/run_annotate.py

smoke-reward: ## Smoke-test the reward model wiring
	$(PY) scripts/smoke_reward.py

leaderboard: ## Build leaderboard from eval results
	$(PY) scripts/leaderboard.py --in data/eval/results.jsonl --out docs/leaderboard.md

clean: ## Remove .venv, generated data, caches
	rm -rf .venv data/ wandb/ checkpoints/ outputs/ .pytest_cache .ruff_cache __pycache__
