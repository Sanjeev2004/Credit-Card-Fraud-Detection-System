.PHONY: install lint test run

install:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e ".[dev]"

lint:
	.venv/bin/python -m ruff check src scripts tests

test:
	.venv/bin/python -m pytest -q

run:
	./run.sh
