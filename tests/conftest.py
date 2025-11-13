"""Pytest configuration and fixtures."""

import pytest

from ai_text_summarizer.config import SummarizerConfig


@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return (
        "Artificial Intelligence (AI) is intelligence demonstrated by machines, "
        "as opposed to natural intelligence displayed by animals including humans. "
        "Leading AI textbooks define the field as the study of intelligent agents: "
        "any system that perceives its environment and takes actions that maximize "
        "its chance of achieving its goals. Some popular accounts use the term "
        "artificial intelligence to describe machines that mimic cognitive functions "
        "that humans associate with the human mind, such as learning and problem solving."
    )


@pytest.fixture
def long_text():
    """Provide a longer text for chunking tests."""
    return " ".join(
        [
            "This is a test sentence that will be repeated many times to create a long text. "
            "The purpose is to test the chunking functionality of the text summarizer. "
            "We need to ensure that it can handle texts that exceed the model's token limit. "
        ]
        * 100  # Repeat to create a very long text
    )


@pytest.fixture
def default_config():
    """Provide default configuration."""
    return SummarizerConfig()


@pytest.fixture
def custom_config():
    """Provide custom configuration."""
    return SummarizerConfig(
        model_name="t5-small",
        max_length=100,
        min_length=20,
        chunk_overlap=30,
    )
