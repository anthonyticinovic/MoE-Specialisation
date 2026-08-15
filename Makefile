.PHONY: help lint format test demo check clean-demo

PY ?= uv run

help:
	@echo "make lint        Ruff lint (correctness everywhere, style on the core) + mypy"
	@echo "make format      Apply ruff formatting"
	@echo "make test        CPU-only pytest suite"
	@echo "make demo        Run the whole pipeline on CPU against synthetic fixtures"
	@echo "make check       lint + test + demo"
	@echo "make clean-demo  Remove demo_output/"

lint:
	$(PY) ruff check --select F821,F811,F822 .
	$(PY) ruff check models/ data/ tests/
	$(PY) ruff format --check .
	$(PY) mypy

format:
	$(PY) ruff format .

test:
	$(PY) pytest -q

demo:
	$(PY) python demo/run_demo.py

check: lint test demo

clean-demo:
	rm -rf demo_output/
