.PHONY: help lint format test demo figures check clean-demo

PY ?= uv run

# Committed Stage 3 metrics (inputs) and where the PNGs are written (generated).
METRICS ?= paper_metrics/stage3
FIGURES ?= results/figures
LAYERS ?= all_layers

help:
	@echo "make lint        Ruff lint (correctness everywhere, style on the core) + mypy"
	@echo "make format      Apply ruff formatting"
	@echo "make test        CPU-only pytest suite"
	@echo "make demo        Run the whole pipeline on CPU against synthetic fixtures"
	@echo "make figures     Regenerate the paper's figures from results/ (no GPU)"
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

figures:
	@ls $(METRICS)/expert_metrics/expert_metrics_epoch_*.json >/dev/null 2>&1 || { \
		echo "No Stage 3 metrics in $(METRICS)/expert_metrics/."; \
		echo "They are not committed yet — see paper_metrics/README.md for what to add."; \
		exit 1; }
	$(PY) python analysis_scripts/plot_expert_metrics.py \
		--metrics_dir "$(METRICS)/expert_metrics" \
		--output_dir "$(FIGURES)" \
		--layers "$(LAYERS)" \
		--training_metrics "$(METRICS)/training_metrics_stage3.json"

check: lint test demo

clean-demo:
	rm -rf demo_output/
