"""
Utility functions for AI Text Summarizer.
"""

import logging
from typing import List, Tuple
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration.

    Args:
        level: Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def split_text_into_chunks(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_chunk_size: int,
    overlap: int = 50,
) -> List[str]:
    """Split text into chunks based on token count with overlap.

    This function properly splits text by tokens, not characters, ensuring
    that each chunk fits within the model's token limit.

    Args:
        text: The input text to split.
        tokenizer: The tokenizer to use for tokenization.
        max_chunk_size: Maximum number of tokens per chunk.
        overlap: Number of tokens to overlap between chunks.

    Returns:
        List of text chunks.

    Raises:
        ValueError: If max_chunk_size <= overlap.
    """
    if max_chunk_size <= overlap:
        raise ValueError(
            f"max_chunk_size ({max_chunk_size}) must be greater than "
            f"overlap ({overlap})"
        )

    # Tokenize the entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)

    logger.debug(f"Total tokens in text: {total_tokens}")
    logger.debug(f"Splitting into chunks of max {max_chunk_size} tokens")

    chunks = []
    start_idx = 0

    while start_idx < total_tokens:
        # Get chunk tokens
        end_idx = min(start_idx + max_chunk_size, total_tokens)
        chunk_tokens = tokens[start_idx:end_idx]

        # Decode chunk back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

        logger.debug(
            f"Created chunk {len(chunks)} with {len(chunk_tokens)} tokens "
            f"(positions {start_idx} to {end_idx})"
        )

        # Move to next chunk with overlap
        if end_idx >= total_tokens:
            break
        start_idx = end_idx - overlap

    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


def validate_text_input(text: str) -> None:
    """Validate that the input text is suitable for summarization.

    Args:
        text: The input text to validate.

    Raises:
        ValueError: If text is empty or invalid.
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Text must be a string, got {type(text).__name__}")

    if not text or not text.strip():
        raise ValueError("Text cannot be empty or whitespace only")


def estimate_summary_ratio(
    input_tokens: int, max_length: int, min_length: int
) -> Tuple[float, int]:
    """Estimate the compression ratio and expected summary length.

    Args:
        input_tokens: Number of tokens in input text.
        max_length: Maximum summary length.
        min_length: Minimum summary length.

    Returns:
        Tuple of (compression_ratio, expected_length)
    """
    expected_length = min(max_length, max(min_length, input_tokens // 4))
    compression_ratio = input_tokens / expected_length if expected_length > 0 else 0
    return compression_ratio, expected_length
