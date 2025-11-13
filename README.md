# AI Text Summarizer

[![CI](https://github.com/pyenthusiasts/AI-Text-Summarizer/workflows/CI/badge.svg)](https://github.com/pyenthusiasts/AI-Text-Summarizer/actions)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional Python package for text summarization using state-of-the-art pre-trained models from Hugging Face Transformers. Supports multiple models, handles long texts automatically, and provides both Python API and command-line interface.

## Features

- **Multiple Model Support**: T5, BART, Pegasus, and more
- **Automatic Text Chunking**: Handles texts of any length by intelligently splitting into chunks
- **Smart Token-Based Splitting**: Proper token-aware chunking with configurable overlap
- **Batch Processing**: Summarize multiple texts efficiently
- **CLI Interface**: Easy-to-use command-line tool
- **Python API**: Clean, intuitive API for programmatic use
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Type Hints**: Full type annotation for better IDE support
- **Comprehensive Testing**: Extensive test suite with high coverage
- **Configurable**: Flexible configuration for all parameters

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/pyenthusiasts/AI-Text-Summarizer.git
cd AI-Text-Summarizer

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python 3.8 or higher
- PyTorch 2.0+
- Transformers 4.30+

## Quick Start

### Python API

```python
from ai_text_summarizer import summarize_text

# Simple usage with default settings
text = """
Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to
natural intelligence displayed by animals including humans. Leading AI textbooks define
the field as the study of intelligent agents: any system that perceives its environment
and takes actions that maximize its chance of achieving its goals.
"""

summary = summarize_text(text)
print(summary)
```

### Advanced Usage

```python
from ai_text_summarizer import TextSummarizer, SummarizerConfig

# Custom configuration
config = SummarizerConfig(
    model_name="facebook/bart-large-cnn",
    max_length=150,
    min_length=50,
    chunk_overlap=30
)

# Create summarizer instance
summarizer = TextSummarizer(config=config, verbose=True)

# Summarize single text
summary = summarizer.summarize(text)

# Batch summarize multiple texts
texts = ["Text 1...", "Text 2...", "Text 3..."]
summaries = summarizer.batch_summarize(texts)
```

### Command-Line Interface

```bash
# Summarize from stdin
echo "Your long text here..." | ai-text-summarizer

# Summarize a file
ai-text-summarizer -i input.txt -o summary.txt

# Use a different model
ai-text-summarizer -i input.txt --model facebook/bart-large-cnn

# Customize summary length
ai-text-summarizer -i input.txt --max-length 200 --min-length 50

# Enable verbose output
ai-text-summarizer -i input.txt -v

# List supported models
ai-text-summarizer --list-models

# Get help
ai-text-summarizer --help
```

## Supported Models

| Model | Type | Max Tokens | Best For |
|-------|------|------------|----------|
| `t5-small` | T5 | 512 | Fast, general purpose (default) |
| `t5-base` | T5 | 512 | Better quality, still fast |
| `t5-large` | T5 | 512 | High quality, slower |
| `facebook/bart-large-cnn` | BART | 1024 | News articles, longer texts |
| `google/pegasus-xsum` | Pegasus | 512 | Extreme summarization |
| `google/pegasus-cnn_dailymail` | Pegasus | 1024 | News, balanced summaries |

## Configuration Options

### SummarizerConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "t5-small" | Pre-trained model to use |
| `max_length` | int | 130 | Maximum summary length in tokens |
| `min_length` | int | 30 | Minimum summary length in tokens |
| `max_chunk_size` | int | None | Max chunk size (auto-detected if None) |
| `chunk_overlap` | int | 50 | Token overlap between chunks |
| `do_sample` | bool | False | Use sampling for generation |
| `temperature` | float | 1.0 | Sampling temperature |
| `device` | str | None | Device to use ('cpu', 'cuda', or None for auto) |

## How It Works

1. **Initialization**: Loads the specified pre-trained model and tokenizer
2. **Token Counting**: Tokenizes input text to determine length
3. **Smart Chunking**: If text exceeds model limits, splits into overlapping chunks
4. **Summarization**: Generates summaries for each chunk
5. **Combination**: Merges chunk summaries into final output
6. **Recursive Processing**: Re-summarizes if combined result is still too long

### Key Improvements Over Basic Implementation

- **Token-Based Chunking**: Uses actual tokens instead of characters for accurate splitting
- **Error Handling**: Comprehensive error checking and validation
- **Logging**: Detailed logging for debugging and monitoring
- **Type Safety**: Full type hints throughout
- **Configurability**: All parameters exposed for customization
- **Batch Processing**: Efficient handling of multiple texts
- **Device Management**: Automatic GPU detection and usage

## Examples

See the [examples/](examples/) directory for detailed examples:

- [`basic_usage.py`](examples/basic_usage.py) - Simple usage with default settings
- [`advanced_usage.py`](examples/advanced_usage.py) - Custom configuration and features
- [`file_processing.py`](examples/file_processing.py) - Reading and writing files

Run an example:
```bash
cd examples
python basic_usage.py
```

## Development

### Setup Development Environment

```bash
# Clone and install with dev dependencies
git clone https://github.com/pyenthusiasts/AI-Text-Summarizer.git
cd AI-Text-Summarizer
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
pytest --cov=src/ai_text_summarizer --cov-report=html
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run all checks
make format lint test
```

### Project Structure

```
AI-Text-Summarizer/
├── src/ai_text_summarizer/    # Main package
│   ├── __init__.py             # Package initialization
│   ├── summarizer.py           # Core summarization logic
│   ├── config.py               # Configuration classes
│   ├── utils.py                # Utility functions
│   └── cli.py                  # Command-line interface
├── tests/                      # Test suite
│   ├── test_summarizer.py
│   ├── test_config.py
│   └── test_utils.py
├── examples/                   # Example scripts
├── .github/workflows/          # CI/CD configuration
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development dependencies
├── setup.py                    # Package setup
├── pyproject.toml             # Build configuration
└── README.md                  # This file
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make test lint`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Testing

The project includes a comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ai_text_summarizer

# Run specific test file
pytest tests/test_summarizer.py

# Run with verbose output
pytest -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

## Acknowledgments

- Built with [Hugging Face Transformers](https://huggingface.co/transformers/)
- Powered by [PyTorch](https://pytorch.org/)
- Supports models from [Hugging Face Model Hub](https://huggingface.co/models)

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{ai_text_summarizer,
  title = {AI Text Summarizer},
  author = {Python Enthusiasts},
  year = {2024},
  url = {https://github.com/pyenthusiasts/AI-Text-Summarizer}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/pyenthusiasts/AI-Text-Summarizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pyenthusiasts/AI-Text-Summarizer/discussions)
- **Documentation**: See [examples/](examples/) and inline documentation

## Roadmap

- [ ] Web interface for easy usage
- [ ] REST API server
- [ ] Additional model support
- [ ] Multi-language support
- [ ] Model fine-tuning scripts
- [ ] Docker containerization
- [ ] Performance benchmarks
- [ ] Caching mechanisms

---

Made with ❤️ by Python Enthusiasts
