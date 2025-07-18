.PHONY: help install install-dev test lint format clean build release

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests with coverage
	pytest tests/ -v --cov=tonal_hortator --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage (faster)
	pytest tests/ -v

lint: ## Run all linting checks
	flake8 tonal_hortator/
	black --check --diff .
	isort --check-only --diff .
	mypy tonal_hortator/ --ignore-missing-imports

format: ## Format code with black and isort
	black .
	isort .

security: ## Run security checks
	bandit -r tonal_hortator/

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.xml

build: ## Build the package
	python -m build

release: ## Build and check package for release
	python -m build
	twine check dist/*

check-all: ## Run all checks (lint, test, security)
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) security

dev-setup: ## Set up development environment
	$(MAKE) install-dev
	$(MAKE) format
	$(MAKE) test

ci: ## Run CI checks locally (matches GitHub Actions exactly)
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
	# Lint with flake8 (exact GitHub Actions commands)
	flake8 tonal_hortator/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 tonal_hortator/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	# Check code formatting with black
	black --check --diff .
	# Check import sorting with isort
	isort --check-only --diff .
	# Type check with mypy
	mypy tonal_hortator/ --ignore-missing-imports
	# Run tests with pytest (exact GitHub Actions command)
	pytest tonal_hortator/tests/ -v --cov=tonal_hortator --cov-report=xml --cov-report=term-missing
	# Security checks (from security job)
	bandit -r tonal_hortator/ -f json -o bandit-report.json --exit-zero 