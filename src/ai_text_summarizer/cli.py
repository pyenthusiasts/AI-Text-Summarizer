"""
Command-line interface for AI Text Summarizer.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import SummarizerConfig
from .summarizer import TextSummarizer

logger = logging.getLogger(__name__)


def read_input_text(input_path: Optional[str] = None) -> str:
    """Read input text from file or stdin.

    Args:
        input_path: Path to input file, or None to read from stdin.

    Returns:
        The input text.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        IOError: If reading fails.
    """
    if input_path:
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Reading input from: {input_path}")
        return input_file.read_text(encoding="utf-8")
    else:
        logger.info("Reading input from stdin")
        return sys.stdin.read()


def write_output_text(text: str, output_path: Optional[str] = None) -> None:
    """Write output text to file or stdout.

    Args:
        text: The text to write.
        output_path: Path to output file, or None to write to stdout.
    """
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing output to: {output_path}")
        output_file.write_text(text, encoding="utf-8")
    else:
        print(text)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="ai-text-summarizer",
        description="Summarize text using pre-trained Hugging Face models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize text from stdin
  echo "Your long text here..." | ai-text-summarizer

  # Summarize a file
  ai-text-summarizer -i input.txt -o summary.txt

  # Use a different model
  ai-text-summarizer -i input.txt --model facebook/bart-large-cnn

  # Customize summary length
  ai-text-summarizer -i input.txt --max-length 200 --min-length 50

  # Enable verbose output
  ai-text-summarizer -i input.txt -v
        """,
    )

    # Input/Output options
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input file path (default: read from stdin)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path (default: write to stdout)",
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="t5-small",
        help="Model name (default: t5-small)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=130,
        help="Maximum summary length in tokens (default: 130)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=30,
        help="Minimum summary length in tokens (default: 30)",
    )

    # Advanced options
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        help="Maximum chunk size in tokens (default: auto-detect)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Number of tokens to overlap between chunks (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device to use for inference (default: auto-detect)",
    )

    # Sampling options
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0, only used with --do-sample)",
    )

    # Utility options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported models and exit",
    )

    return parser


def list_supported_models() -> None:
    """Print list of supported models."""
    print("Supported models:")
    print()
    for model_name, info in SummarizerConfig.SUPPORTED_MODELS.items():
        print(f"  {model_name}")
        print(f"    Type: {info['type']}")
        print(f"    Max tokens: {info['max_tokens']}")
        print()


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        list_supported_models()
        return 0

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        # Read input
        input_text = read_input_text(args.input)

        if not input_text.strip():
            logger.error("Input text is empty")
            return 1

        # Create configuration
        config = SummarizerConfig(
            model_name=args.model,
            max_length=args.max_length,
            min_length=args.min_length,
            max_chunk_size=args.max_chunk_size,
            chunk_overlap=args.chunk_overlap,
            do_sample=args.do_sample,
            temperature=args.temperature,
            device=args.device,
        )

        # Initialize summarizer
        logger.info("Initializing text summarizer...")
        summarizer = TextSummarizer(config=config, verbose=args.verbose)

        # Summarize
        logger.info("Summarizing text...")
        summary = summarizer.summarize(input_text)

        # Write output
        write_output_text(summary, args.output)

        if args.output:
            logger.info("Summarization completed successfully")
        else:
            # Add a newline after the summary when writing to stdout
            if not summary.endswith("\n"):
                print()

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            logger.exception("Detailed error information:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
