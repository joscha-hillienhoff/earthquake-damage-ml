PYTHON_VERSION = 3.11

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "%-22s%s\n", $$1, $$2}'

.PHONY: create_environment
create_environment:  ## Create the Poetry virtual environment
	poetry env use $(PYTHON_VERSION)

.PHONY: requirements
requirements:  ## Install dependencies
	poetry install

.PHONY: lint
lint:  ## Lint and check formatting with ruff
	poetry run ruff check .
	poetry run ruff format --check .

.PHONY: format
format:  ## Apply ruff fixes and formatting
	poetry run ruff check --fix .
	poetry run ruff format .

.PHONY: test
test:  ## Run the test suite
	poetry run pytest

.PHONY: interim
interim:  ## Merge competition train values and labels into data/interim/*.parquet
	poetry run python -m src.dataset

.PHONY: clean
clean:  ## Delete compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
