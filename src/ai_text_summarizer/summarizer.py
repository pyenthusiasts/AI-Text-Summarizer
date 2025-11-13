"""
Main text summarization module using Hugging Face Transformers.
"""

import logging
import warnings
from typing import List, Optional

from transformers import AutoTokenizer, pipeline
from transformers.utils import logging as transformers_logging

from .config import SummarizerConfig
from .utils import (
    estimate_summary_ratio,
    setup_logging,
    split_text_into_chunks,
    validate_text_input,
)

logger = logging.getLogger(__name__)


class TextSummarizer:
    """Text summarization using pre-trained Hugging Face models.

    This class provides an enhanced interface for text summarization with support
    for large texts, multiple models, and comprehensive error handling.

    Attributes:
        config: Configuration object for the summarizer.
        tokenizer: The tokenizer for the selected model.
        summarizer: The Hugging Face summarization pipeline.
    """

    def __init__(
        self,
        config: Optional[SummarizerConfig] = None,
        verbose: bool = False,
    ):
        """Initialize the TextSummarizer.

        Args:
            config: Configuration object. If None, uses default configuration.
            verbose: Whether to enable verbose logging.

        Raises:
            ValueError: If model initialization fails.
        """
        self.config = config or SummarizerConfig()

        # Set up logging
        log_level = logging.DEBUG if verbose else logging.INFO
        setup_logging(log_level)

        # Suppress transformers logging unless verbose
        if not verbose:
            transformers_logging.set_verbosity_error()
            warnings.filterwarnings("ignore")

        logger.info(f"Initializing TextSummarizer with model: {self.config.model_name}")

        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            logger.debug("Tokenizer loaded successfully")

            # Initialize summarization pipeline
            device = self._get_device()
            self.summarizer = pipeline(
                "summarization",
                model=self.config.model_name,
                device=device,
            )
            logger.info(f"Model loaded successfully on device: {device}")

            # Get max token length
            self.max_tokens = self.tokenizer.model_max_length
            if self.max_tokens > 1_000_000:  # Some models return very large values
                self.max_tokens = self.config.SUPPORTED_MODELS.get(self.config.model_name, {}).get(
                    "max_tokens", 512
                )

            logger.debug(f"Model max tokens: {self.max_tokens}")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise ValueError(f"Model initialization failed: {e}") from e

    def _get_device(self) -> int:
        """Determine the device to use for inference.

        Returns:
            Device ID (-1 for CPU, 0+ for GPU).
        """
        if self.config.device is not None:
            if self.config.device == "cpu":
                return -1
            elif self.config.device.startswith("cuda"):
                try:
                    import torch

                    if torch.cuda.is_available():
                        device_id = (
                            int(self.config.device.split(":")[-1])
                            if ":" in self.config.device
                            else 0
                        )
                        return device_id
                except (ImportError, ValueError):
                    logger.warning("CUDA requested but not available, using CPU")
                    return -1
            return -1
        else:
            # Auto-detect
            try:
                import torch

                return 0 if torch.cuda.is_available() else -1
            except ImportError:
                return -1

    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> str:
        """Summarize the input text.

        Handles both short and long texts by automatically chunking when necessary.

        Args:
            text: The input text to summarize.
            max_length: Override the configured max_length.
            min_length: Override the configured min_length.

        Returns:
            The summarized text.

        Raises:
            ValueError: If input text is invalid.
            RuntimeError: If summarization fails.
        """
        # Validate input
        validate_text_input(text)

        # Use config values if not overridden
        max_len = max_length or self.config.max_length
        min_len = min_length or self.config.min_length

        logger.info("Starting text summarization")
        logger.debug(f"Input text length: {len(text)} characters")

        try:
            # Tokenize to check length
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            num_tokens = len(tokens)

            logger.info(f"Input has {num_tokens} tokens (max: {self.max_tokens})")

            # Estimate compression
            ratio, expected_len = estimate_summary_ratio(num_tokens, max_len, min_len)
            logger.debug(
                f"Expected compression ratio: {ratio:.2f}x, target length: {expected_len} tokens"
            )

            # Determine chunk size
            max_chunk_size = self.config.max_chunk_size or (self.max_tokens - 100)

            if num_tokens <= max_chunk_size:
                # Text fits within limit, summarize directly
                logger.info("Text fits in single chunk, summarizing directly")
                summary = self._summarize_chunk(text, max_len, min_len)
            else:
                # Text too long, split into chunks
                logger.info("Text exceeds chunk size, splitting into chunks")
                summary = self._summarize_long_text(text, max_len, min_len, max_chunk_size)

            logger.info("Summarization completed successfully")
            logger.debug(f"Summary length: {len(summary)} characters")

            return summary

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise RuntimeError(f"Failed to summarize text: {e}") from e

    def _summarize_chunk(
        self,
        text: str,
        max_length: int,
        min_length: int,
    ) -> str:
        """Summarize a single chunk of text.

        Args:
            text: The text chunk to summarize.
            max_length: Maximum length of the summary.
            min_length: Minimum length of the summary.

        Returns:
            The summarized text.
        """
        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature if self.config.do_sample else None,
        )
        return result[0]["summary_text"]

    def _summarize_long_text(
        self,
        text: str,
        max_length: int,
        min_length: int,
        max_chunk_size: int,
    ) -> str:
        """Summarize long text by splitting into chunks.

        Args:
            text: The input text to summarize.
            max_length: Maximum length of each chunk summary.
            min_length: Minimum length of each chunk summary.
            max_chunk_size: Maximum size of each chunk in tokens.

        Returns:
            The combined summary of all chunks.
        """
        # Split text into chunks
        chunks = split_text_into_chunks(
            text,
            self.tokenizer,
            max_chunk_size,
            self.config.chunk_overlap,
        )

        logger.info(f"Processing {len(chunks)} chunks")

        # Summarize each chunk
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Summarizing chunk {i}/{len(chunks)}")
            try:
                summary = self._summarize_chunk(chunk, max_length, min_length)
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"Failed to summarize chunk {i}: {e}")
                # Continue with other chunks
                continue

        if not summaries:
            raise RuntimeError("Failed to summarize any chunks")

        # Combine summaries
        combined = " ".join(summaries)

        # If combined summary is still too long, recursively summarize
        combined_tokens = len(self.tokenizer.encode(combined, add_special_tokens=True))
        if combined_tokens > max_chunk_size:
            logger.info("Combined summary too long, summarizing recursively")
            return self._summarize_long_text(combined, max_length, min_length, max_chunk_size)

        return combined

    def batch_summarize(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> List[str]:
        """Summarize multiple texts.

        Args:
            texts: List of texts to summarize.
            max_length: Override the configured max_length.
            min_length: Override the configured min_length.

        Returns:
            List of summaries.
        """
        logger.info(f"Batch summarizing {len(texts)} texts")
        summaries = []

        for i, text in enumerate(texts, 1):
            logger.debug(f"Processing text {i}/{len(texts)}")
            try:
                summary = self.summarize(text, max_length, min_length)
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"Failed to summarize text {i}: {e}")
                summaries.append("")  # Add empty summary for failed texts

        return summaries


def summarize_text(
    text: str,
    max_length: int = 130,
    min_length: int = 30,
    model_name: str = "t5-small",
    verbose: bool = False,
) -> str:
    """Convenience function to summarize text with default settings.

    This function provides a simple interface similar to the original implementation
    but with all the enhancements.

    Args:
        text: The text to summarize.
        max_length: Maximum length of the summary (default: 130).
        min_length: Minimum length of the summary (default: 30).
        model_name: The model to use (default: 't5-small').
        verbose: Whether to enable verbose logging.

    Returns:
        The summarized text.

    Example:
        >>> summary = summarize_text("Your long article or text here...")
        >>> print(summary)
    """
    config = SummarizerConfig(
        model_name=model_name,
        max_length=max_length,
        min_length=min_length,
    )

    summarizer = TextSummarizer(config=config, verbose=verbose)
    return summarizer.summarize(text)
