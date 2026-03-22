# Calgary Open Data Portfolio — Makefile
# Usage: make <target>

PYTHON  ?= python
PIP     ?= pip
VENV    ?= venv

PROJECTS := $(sort $(wildcard project_*/))

.PHONY: help install install-all venv run-portfolio run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv: ## Create a virtual environment
	$(PYTHON) -m venv $(VENV)
	@echo "Activate with: source $(VENV)/bin/activate  (or $(VENV)\\Scripts\\activate on Windows)"

install: ## Install shared (root) dependencies
	$(PIP) install -r requirements.txt

install-all: install ## Install shared + every project's dependencies
	@for proj in $(PROJECTS); do \
		if [ -f "$$proj/requirements.txt" ]; then \
			echo "Installing $$proj dependencies..."; \
			$(PIP) install -r "$$proj/requirements.txt"; \
		fi; \
	done

run-portfolio: ## Run the portfolio landing page (Streamlit)
	streamlit run portfolio_app.py

run-%: ## Run a specific project app, e.g. make run-project_01_building_permit_cost_predictor
	streamlit run $*/app.py

clean: ## Remove cached data, models, and __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned __pycache__ and .ipynb_checkpoints directories"
