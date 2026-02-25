.DEFAULT_GOAL := help

.PHONY: all
all: fmt lint

.PHONY: install
install:
	uv sync

.PHONY: fmt
fmt:
	uv run ruff format

.PHONY: lint
lint:
	uv run ruff check --fix
	uv run ty check

.PHONY: check
check:
	uv run ruff format --check
	uv run ruff check
	uv run ty check

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make help     - Show this help message (default)"
	@echo "  make all      - Format (ruff) and lint (ruff, ty)"
	@echo "  make install  - Install dependencies with uv"
	@echo "  make fmt      - Format code with ruff"
	@echo "  make lint     - Lint with ruff (--fix) and type-check with ty"
	@echo "  make check    - Check formatting (ruff), lint (ruff), and types (ty)"
