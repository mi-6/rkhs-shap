SHELL := /bin/bash

.PHONY: uv format lint lint-fix check test all

uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "uv installed. Use 'uv' to manage your Python environment."

format:
	uv run ruff format
	uv run ruff check --fix

lint:
	uv run ruff format --check
	uv run ruff check

test:
	uv run pytest tests

all: format lint test
