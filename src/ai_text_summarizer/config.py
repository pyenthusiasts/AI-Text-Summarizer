"""
Configuration module for AI Text Summarizer.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SummarizerConfig:
    """Configuration for text summarization.

    Attributes:
        model_name: The pre-trained model to use for summarization.
        max_length: Maximum length of the summary in tokens.
        min_length: Minimum length of the summary in tokens.
        max_chunk_size: Maximum size of text chunks in tokens.
        chunk_overlap: Number of tokens to overlap between chunks.
        do_sample: Whether to use sampling for generation.
        temperature: Sampling temperature (only used if do_sample=True).
        device: Device to use for inference ('cpu', 'cuda', or None for auto).
    """

    model_name: str = "t5-small"
    max_length: int = 130
    min_length: int = 30
    max_chunk_size: Optional[int] = None
    chunk_overlap: int = 50
    do_sample: bool = False
    temperature: float = 1.0
    device: Optional[str] = None

    # Supported models with their characteristics
    SUPPORTED_MODELS = {
        "t5-small": {"type": "t5", "max_tokens": 512},
        "t5-base": {"type": "t5", "max_tokens": 512},
        "t5-large": {"type": "t5", "max_tokens": 512},
        "facebook/bart-large-cnn": {"type": "bart", "max_tokens": 1024},
        "google/pegasus-xsum": {"type": "pegasus", "max_tokens": 512},
        "google/pegasus-cnn_dailymail": {"type": "pegasus", "max_tokens": 1024},
    }

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_length <= self.min_length:
            raise ValueError(
                f"max_length ({self.max_length}) must be greater than "
                f"min_length ({self.min_length})"
            )

        if self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be non-negative, got {self.chunk_overlap}"
            )

        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {self.temperature}"
            )
