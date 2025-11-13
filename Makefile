.PHONY: help install install-dev test lint format clean build

help:
	@echo "Available commands:"
	@echo "  make install      - Install package"
	@echo "  make install-dev  - Install package with dev dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with black and isort"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make build        - Build package"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src/ai_text_summarizer --cov-report=term-missing

lint:
	flake8 src/ tests/
	pylint src/ai_text_summarizer
	mypy src/ai_text_summarizer

format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build
