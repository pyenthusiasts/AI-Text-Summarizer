"""Tests for utility functions."""

import pytest
from unittest.mock import Mock
from ai_text_summarizer.utils import (
    split_text_into_chunks,
    validate_text_input,
    estimate_summary_ratio,
)


def test_validate_text_input_valid():
    """Test validation with valid input."""
    validate_text_input("This is valid text")
    # Should not raise any exception


def test_validate_text_input_empty():
    """Test validation with empty text."""
    with pytest.raises(ValueError, match="Text cannot be empty"):
        validate_text_input("")


def test_validate_text_input_whitespace():
    """Test validation with whitespace-only text."""
    with pytest.raises(ValueError, match="Text cannot be empty"):
        validate_text_input("   \n\t  ")


def test_validate_text_input_non_string():
    """Test validation with non-string input."""
    with pytest.raises(TypeError, match="Text must be a string"):
        validate_text_input(123)


def test_estimate_summary_ratio():
    """Test summary ratio estimation."""
    ratio, expected_len = estimate_summary_ratio(1000, 200, 50)
    assert ratio > 0
    assert 50 <= expected_len <= 200
    assert expected_len == min(200, max(50, 1000 // 4))


def test_estimate_summary_ratio_short_text():
    """Test summary ratio with very short text."""
    ratio, expected_len = estimate_summary_ratio(50, 200, 100)
    assert expected_len >= 100  # Should use min_length


def test_split_text_into_chunks():
    """Test text chunking functionality."""
    # Create a mock tokenizer
    tokenizer = Mock()
    tokenizer.encode = Mock(return_value=list(range(1000)))  # 1000 tokens
    tokenizer.decode = Mock(side_effect=lambda tokens, **kwargs: f"chunk_{len(tokens)}_tokens")

    chunks = split_text_into_chunks("long text", tokenizer, max_chunk_size=300, overlap=50)

    assert len(chunks) > 1  # Should create multiple chunks
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_split_text_into_chunks_invalid_params():
    """Test chunking with invalid parameters."""
    tokenizer = Mock()
    tokenizer.encode = Mock(return_value=list(range(100)))

    with pytest.raises(ValueError, match="max_chunk_size .* must be greater than overlap"):
        split_text_into_chunks("text", tokenizer, max_chunk_size=50, overlap=100)
