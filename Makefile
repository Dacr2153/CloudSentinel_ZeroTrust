# FILE: cloudsentinel-zero-trust/Makefile
# CloudSentinel Zero-Trust — Developer Makefile
# Usage: make <target>

.PHONY: help install install-dev lint format typecheck test test-cov security \
        cfn-lint package clean run train evaluate

PYTHON    ?= .venv/bin/python
PIP       ?= .venv/bin/pip
PYTEST    ?= .venv/bin/pytest
RUFF      ?= .venv/bin/ruff
MYPY      ?= .venv/bin/mypy
BANDIT    ?= .venv/bin/bandit

SRC_DIR   := src
TEST_DIR  := tests
BUILD_DIR := build
COV_MIN   := 80

# ────────────────────────────────────────────────────────────────────
# Default target
# ────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ────────────────────────────────────────────────────────────────────
# Setup
# ────────────────────────────────────────────────────────────────────

install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install ## Install all dependencies (prod + dev)
	$(PIP) install -r requirements-dev.txt

# ────────────────────────────────────────────────────────────────────
# Code Quality
# ────────────────────────────────────────────────────────────────────

lint: ## Run Ruff linter
	$(RUFF) check $(SRC_DIR)/ $(TEST_DIR)/

format: ## Auto-format code with Ruff
	$(RUFF) format $(SRC_DIR)/ $(TEST_DIR)/

typecheck: ## Run mypy type checking
	$(MYPY) $(SRC_DIR)/ --ignore-missing-imports

# ────────────────────────────────────────────────────────────────────
# Testing
# ────────────────────────────────────────────────────────────────────

test: ## Run all unit tests
	$(PYTEST) $(TEST_DIR)/unit/ -v

test-cov: ## Run tests with coverage report
	$(PYTEST) $(TEST_DIR)/ \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=$(COV_MIN) \
		-v

# ────────────────────────────────────────────────────────────────────
# Security
# ────────────────────────────────────────────────────────────────────

security: ## Run Bandit security scan (medium+ severity)
	$(BANDIT) -r $(SRC_DIR)/ -ll

# ────────────────────────────────────────────────────────────────────
# Infrastructure (requires AWS)
# ────────────────────────────────────────────────────────────────────

cfn-lint: ## Lint CloudFormation templates
	cfn-lint infrastructure/cloudformation/*.yaml

package: clean ## Package Lambda function for deployment
	mkdir -p $(BUILD_DIR)/lambda
	cp -r $(SRC_DIR)/* $(BUILD_DIR)/lambda/
	$(PIP) install -r requirements.txt -t $(BUILD_DIR)/lambda/ --quiet
	cd $(BUILD_DIR)/lambda && zip -r ../../lambda-package.zip . -x "*.pyc" "__pycache__/*"
	@echo "✅ Lambda package: lambda-package.zip"

# ────────────────────────────────────────────────────────────────────
# CloudSentinel — LOCAL MODE (no AWS required)
# ────────────────────────────────────────────────────────────────────

run: ## Launch the interactive CLI menu
	export $$(grep -v '^#' .env.local | grep -v '^$$' | xargs) && \
	$(PYTHON) cloudsentinel.py

train: ## Train the ML model (non-interactive)
	export $$(grep -v '^#' .env.local | grep -v '^$$' | xargs) && \
	$(PYTHON) cloudsentinel.py --no-menu --train

evaluate: ## Evaluate model performance (non-interactive)
	export $$(grep -v '^#' .env.local | grep -v '^$$' | xargs) && \
	$(PYTHON) cloudsentinel.py --no-menu --evaluate

# ────────────────────────────────────────────────────────────────────
# Cleanup
# ────────────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf $(BUILD_DIR)/ htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/ reports/
	rm -f lambda-package.zip coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete"

