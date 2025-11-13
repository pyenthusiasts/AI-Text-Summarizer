# Contributing to AI Text Summarizer

Thank you for your interest in contributing to AI Text Summarizer! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)
- Any relevant code samples or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear description of the enhancement
- Why this enhancement would be useful
- Examples of how it would work
- Any alternative approaches you've considered

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Set up your development environment**:
   ```bash
   git clone https://github.com/yourusername/AI-Text-Summarizer.git
   cd AI-Text-Summarizer
   pip install -e ".[dev]"
   ```

3. **Make your changes**:
   - Write clear, readable code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Run tests and checks**:
   ```bash
   # Run tests
   make test

   # Run linting
   make lint

   # Format code
   make format
   ```

5. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference any related issues
   ```bash
   git commit -m "Add feature X that does Y (fixes #123)"
   ```

6. **Push to your fork** and submit a pull request

### Pull Request Guidelines

- Keep pull requests focused on a single feature or fix
- Include tests that cover your changes
- Update documentation if you're changing functionality
- Ensure all tests pass and code is properly formatted
- Write a clear PR description explaining what and why

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip
- virtualenv (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/pyenthusiasts/AI-Text-Summarizer.git
cd AI-Text-Summarizer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ai_text_summarizer --cov-report=html

# Run specific test file
pytest tests/test_summarizer.py

# Run with verbose output
pytest -v
```

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pylint**: Additional linting

Format your code before committing:
```bash
make format
```

Check your code:
```bash
make lint
```

### Project Structure

```
AI-Text-Summarizer/
├── src/ai_text_summarizer/   # Main package
│   ├── __init__.py
│   ├── summarizer.py          # Core summarization logic
│   ├── config.py              # Configuration classes
│   ├── utils.py               # Utility functions
│   └── cli.py                 # Command-line interface
├── tests/                     # Test suite
├── examples/                  # Example scripts
├── docs/                      # Documentation
└── README.md
```

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Use meaningful variable and function names

### Documentation

- Add docstrings to all public APIs
- Update README.md for user-facing changes
- Add comments for complex logic
- Include examples in docstrings

### Testing

- Write tests for all new features
- Maintain or improve code coverage
- Test edge cases and error conditions
- Use descriptive test names

Example test:
```python
def test_summarize_empty_text_raises_error():
    """Test that empty text raises ValueError."""
    summarizer = TextSummarizer()
    with pytest.raises(ValueError, match="Text cannot be empty"):
        summarizer.summarize("")
```

## Making Your First Contribution

Not sure where to start? Look for issues labeled:
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `documentation` - Documentation improvements

## Questions?

Feel free to open an issue with your question or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to AI Text Summarizer!
