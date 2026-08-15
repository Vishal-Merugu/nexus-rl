.DEFAULT_GOAL := help
.PHONY: help install lint format types test test-py test-web evaluate clean pr up down logs

# Scope a test run to one package: make test PKG=nexus-sim
PKG ?=
ifeq ($(PKG),)
  PYTEST_TARGET := packages apps
else
  PYTEST_TARGET := packages/$(PKG) apps/$(PKG)
endif

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the Python workspace and the dashboard
	uv sync --all-packages --all-extras
	pnpm --dir apps/dashboard install

lint: ## Lint (ruff)
	uv run ruff check .
	@[ -d apps/dashboard/node_modules ] && pnpm --dir apps/dashboard lint || echo "  (dashboard deps not installed — skipping)"

format: ## Format (ruff + prettier)
	uv run ruff format .
	uv run ruff check . --fix
	@[ -d apps/dashboard/node_modules ] && pnpm --dir apps/dashboard format || true

types: ## Type check the whole workspace (mypy --strict)
	uv run mypy packages apps

test: test-py test-web ## Run all tests

test-py: ## Run Python tests (make test PKG=nexus-sim to scope)
	uv run pytest $(wildcard $(PYTEST_TARGET))

test-web: ## Run dashboard tests
	@[ -d apps/dashboard/node_modules ] && pnpm --dir apps/dashboard test || echo "  (dashboard deps not installed — skipping)"

evaluate: ## Regenerate the baseline comparison table in docs/evaluation/results.md
	uv run nexus-eval --config experiments/configs/evaluation.yaml --write-report

up: ## Start the full stack
	docker compose -f .docker/docker-compose.yml up -d

down: ## Stop the stack
	docker compose -f .docker/docker-compose.yml down

logs: ## Tail service logs
	docker compose -f .docker/docker-compose.yml logs -f

pr: types lint test ## Full pre-commit gate
	@[ -d apps/dashboard/node_modules ] && pnpm --dir apps/dashboard build || true
	@echo "\n  All gates passed. Ready for commit/PR.\n"

clean: ## Remove caches and build output
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .mypy_cache -prune -exec rm -rf {} +
	rm -rf .ruff_cache apps/dashboard/dist
