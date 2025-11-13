"""
File processing example for AI Text Summarizer.

This example demonstrates how to read text from files and save summaries.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_text_summarizer import SummarizerConfig, TextSummarizer  # noqa: E402


def summarize_file(input_path: str, output_path: str, verbose: bool = False):
    """
    Summarize text from a file and save to another file.

    Args:
        input_path: Path to input text file
        output_path: Path to output summary file
        verbose: Whether to print verbose output
    """
    # Read input file
    print(f"Reading from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Original text length: {len(text)} characters")

    # Create summarizer
    config = SummarizerConfig(
        model_name="t5-small",
        max_length=150,
        min_length=50,
    )
    summarizer = TextSummarizer(config=config, verbose=verbose)

    # Summarize
    print("Summarizing...")
    summary = summarizer.summarize(text)

    print(f"Summary length: {len(summary)} characters")

    # Write output file
    print(f"Writing to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print("Done!")


def main():
    """Main function."""
    print("=" * 80)
    print("AI Text Summarizer - File Processing Example")
    print("=" * 80)
    print()

    # Create sample input file
    sample_text = """
    Artificial Intelligence has made significant advances in recent years. Machine learning,
    a subset of AI, has become particularly prominent with the development of deep learning
    techniques. These methods have achieved remarkable success in various domains including
    computer vision, natural language processing, and speech recognition.

    Deep neural networks, inspired by the structure of the human brain, can learn complex
    patterns from large amounts of data. This has led to breakthroughs in image classification,
    object detection, machine translation, and many other tasks. Companies and researchers
    worldwide are investing heavily in AI research and development.

    However, AI also raises important ethical considerations. Issues such as bias in algorithms,
    privacy concerns, and the potential impact on employment need to be carefully addressed.
    As AI systems become more powerful and ubiquitous, ensuring they are developed and deployed
    responsibly becomes increasingly important.

    The future of AI holds great promise. Researchers are working on more advanced techniques
    such as reinforcement learning, few-shot learning, and neural architecture search. These
    methods aim to create AI systems that can learn more efficiently and generalize better to
    new situations. As the field continues to evolve, AI is likely to have an increasingly
    profound impact on society.
    """

    # Save sample text
    input_file = "sample_input.txt"
    output_file = "sample_output.txt"

    print(f"Creating sample input file: {input_file}")
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(sample_text)

    print()

    # Summarize
    try:
        summarize_file(input_file, output_file, verbose=False)

        print()
        print("Summary content:")
        print("-" * 80)
        with open(output_file, "r", encoding="utf-8") as f:
            print(f.read())
        print("-" * 80)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Cleanup
        print()
        print("Cleaning up sample files...")
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
