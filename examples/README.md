# Examples

This directory contains example scripts demonstrating various features of the AI Text Summarizer.

## Prerequisites

Make sure you have installed the package:

```bash
pip install -e ..
```

Or install the requirements:

```bash
pip install -r ../requirements.txt
```

## Examples

### 1. Basic Usage (`basic_usage.py`)

The simplest way to use the summarizer with default settings.

```bash
python basic_usage.py
```

This example shows:
- Simple function call with `summarize_text()`
- Default model and parameters
- Basic text summarization

### 2. Advanced Usage (`advanced_usage.py`)

Demonstrates advanced features and customization options.

```bash
python advanced_usage.py
```

This example shows:
- Custom configuration
- Batch processing multiple texts
- Using different models
- Handling long texts with automatic chunking

### 3. File Processing (`file_processing.py`)

Shows how to process text files and save summaries.

```bash
python file_processing.py
```

This example shows:
- Reading text from files
- Writing summaries to files
- Error handling
- File I/O operations

## CLI Usage

You can also use the command-line interface directly:

```bash
# Summarize from stdin
echo "Your long text here..." | ai-text-summarizer

# Summarize a file
ai-text-summarizer -i input.txt -o summary.txt

# Use a different model
ai-text-summarizer -i input.txt --model facebook/bart-large-cnn

# List available models
ai-text-summarizer --list-models

# Get help
ai-text-summarizer --help
```

## Notes

- First run may take time as models need to be downloaded
- GPU acceleration will be used automatically if available
- For production use, consider caching models locally
