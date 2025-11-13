"""Tests for summarizer module."""

from unittest.mock import Mock, patch

import pytest

from ai_text_summarizer.config import SummarizerConfig
from ai_text_summarizer.summarizer import TextSummarizer, summarize_text


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.model_max_length = 512
    tokenizer.encode = Mock(return_value=list(range(100)))  # Short text
    tokenizer.decode = Mock(return_value="mocked decoded text")
    return tokenizer


@pytest.fixture
def mock_pipeline():
    """Create a mock summarization pipeline."""
    pipeline = Mock()
    pipeline.return_value = [{"summary_text": "This is a mocked summary."}]
    return pipeline


@patch("ai_text_summarizer.summarizer.AutoTokenizer")
@patch("ai_text_summarizer.summarizer.pipeline")
def test_text_summarizer_init(mock_pipeline_func, mock_tokenizer_class):
    """Test TextSummarizer initialization."""
    mock_tokenizer_class.from_pretrained = Mock(return_value=Mock(model_max_length=512))
    mock_pipeline_func.return_value = Mock()

    config = SummarizerConfig()
    summarizer = TextSummarizer(config=config)

    assert summarizer.config == config
    mock_tokenizer_class.from_pretrained.assert_called_once()
    mock_pipeline_func.assert_called_once()


@patch("ai_text_summarizer.summarizer.AutoTokenizer")
@patch("ai_text_summarizer.summarizer.pipeline")
def test_summarize_short_text(mock_pipeline_func, mock_tokenizer_class, sample_text):
    """Test summarizing short text."""
    # Setup mocks
    mock_tokenizer = Mock()
    mock_tokenizer.model_max_length = 512
    mock_tokenizer.encode = Mock(return_value=list(range(50)))  # Short text
    mock_tokenizer_class.from_pretrained = Mock(return_value=mock_tokenizer)

    mock_summarizer = Mock()
    mock_summarizer.return_value = [{"summary_text": "Short summary."}]
    mock_pipeline_func.return_value = mock_summarizer

    summarizer = TextSummarizer()
    result = summarizer.summarize(sample_text)

    assert isinstance(result, str)
    assert len(result) > 0
    mock_summarizer.assert_called_once()


@patch("ai_text_summarizer.summarizer.AutoTokenizer")
@patch("ai_text_summarizer.summarizer.pipeline")
def test_summarize_empty_text(mock_pipeline_func, mock_tokenizer_class):
    """Test that empty text raises ValueError."""
    mock_tokenizer_class.from_pretrained = Mock(return_value=Mock(model_max_length=512))
    mock_pipeline_func.return_value = Mock()

    summarizer = TextSummarizer()

    with pytest.raises(ValueError, match="Text cannot be empty"):
        summarizer.summarize("")


@patch("ai_text_summarizer.summarizer.AutoTokenizer")
@patch("ai_text_summarizer.summarizer.pipeline")
def test_summarize_non_string(mock_pipeline_func, mock_tokenizer_class):
    """Test that non-string input raises TypeError."""
    mock_tokenizer_class.from_pretrained = Mock(return_value=Mock(model_max_length=512))
    mock_pipeline_func.return_value = Mock()

    summarizer = TextSummarizer()

    with pytest.raises(TypeError, match="Text must be a string"):
        summarizer.summarize(123)


@patch("ai_text_summarizer.summarizer.AutoTokenizer")
@patch("ai_text_summarizer.summarizer.pipeline")
def test_batch_summarize(mock_pipeline_func, mock_tokenizer_class):
    """Test batch summarization."""
    # Setup mocks
    mock_tokenizer = Mock()
    mock_tokenizer.model_max_length = 512
    mock_tokenizer.encode = Mock(return_value=list(range(50)))
    mock_tokenizer_class.from_pretrained = Mock(return_value=mock_tokenizer)

    mock_summarizer = Mock()
    mock_summarizer.return_value = [{"summary_text": "Summary."}]
    mock_pipeline_func.return_value = mock_summarizer

    summarizer = TextSummarizer()
    texts = ["Text 1", "Text 2", "Text 3"]
    results = summarizer.batch_summarize(texts)

    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)


@patch("ai_text_summarizer.summarizer.TextSummarizer")
def test_summarize_text_convenience_function(mock_summarizer_class):
    """Test the convenience function."""
    mock_instance = Mock()
    mock_instance.summarize = Mock(return_value="Summary text")
    mock_summarizer_class.return_value = mock_instance

    result = summarize_text("Test text", max_length=100, min_length=20)

    assert result == "Summary text"
    mock_summarizer_class.assert_called_once()
    mock_instance.summarize.assert_called_once_with("Test text")
