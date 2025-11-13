"""Tests for configuration module."""

import pytest

from ai_text_summarizer.config import SummarizerConfig


def test_default_config():
    """Test default configuration values."""
    config = SummarizerConfig()
    assert config.model_name == "t5-small"
    assert config.max_length == 130
    assert config.min_length == 30
    assert config.chunk_overlap == 50
    assert config.do_sample is False


def test_custom_config():
    """Test custom configuration values."""
    config = SummarizerConfig(
        model_name="facebook/bart-large-cnn",
        max_length=200,
        min_length=50,
        chunk_overlap=100,
    )
    assert config.model_name == "facebook/bart-large-cnn"
    assert config.max_length == 200
    assert config.min_length == 50
    assert config.chunk_overlap == 100


def test_invalid_length_config():
    """Test that invalid length configuration raises error."""
    with pytest.raises(ValueError, match="max_length .* must be greater than min_length"):
        SummarizerConfig(max_length=50, min_length=100)


def test_invalid_overlap_config():
    """Test that negative overlap raises error."""
    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        SummarizerConfig(chunk_overlap=-10)


def test_invalid_temperature_config():
    """Test that invalid temperature raises error."""
    with pytest.raises(ValueError, match="temperature must be positive"):
        SummarizerConfig(temperature=0)


def test_supported_models():
    """Test that supported models dictionary is available."""
    assert "t5-small" in SummarizerConfig.SUPPORTED_MODELS
    assert "facebook/bart-large-cnn" in SummarizerConfig.SUPPORTED_MODELS
    assert SummarizerConfig.SUPPORTED_MODELS["t5-small"]["type"] == "t5"
