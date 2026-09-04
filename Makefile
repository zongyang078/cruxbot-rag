.PHONY: install install-all test lint fmt check clean

install:  ## Core + dev tooling only (no torch); what CI runs.
	pip install -e ".[dev]"

install-all:  ## Everything, including the retrieval and serving stacks.
	pip install -e ".[rag,serve,ingest,eval,dev]"

test:
	pytest -m "not integration"

test-all:  ## Includes tests needing a built index or a running LLM.
	pytest

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff check --fix src tests
	ruff format src tests

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
