.DEFAULT_GOAL := help

.PHONY: all
all: fmt lint

.PHONY: fmt
fmt:
	ruff format

.PHONY: lint
lint:
	ruff check --fix
	basedpyright

.PHONY: check
check:
	ruff format --check
	ruff check
	basedpyright

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make help     - Show this help message (default)"
	@echo "  make all      - Format (ruff) and lint (ruff, basedpyright)"
	@echo "  make fmt      - Format code with ruff"
	@echo "  make lint     - Lint with ruff (--fix) and type-check with basedpyright"
	@echo "  make check    - Check formatting (ruff), lint (ruff), and types (basedpyright)"
