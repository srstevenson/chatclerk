.PHONY: help
help:
	@echo "Available targets:"
	@echo "  help  show this help message"
	@echo "  fmt   format with ruff"
	@echo "  lint  lint with ruff and type-check with ty"

.PHONY: fmt
fmt:
	uv run ruff check --fix
	uv run ruff format

.PHONY: lint
lint:
	uv run ruff check
	uv run ruff format --check
	uv run ty check
