"""
AI Text Summarizer - A Python package for text summarization using Hugging Face Transformers.
"""

from .config import SummarizerConfig
from .summarizer import TextSummarizer, summarize_text

__version__ = "1.0.0"
__all__ = ["TextSummarizer", "summarize_text", "SummarizerConfig"]
